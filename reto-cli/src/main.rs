#![forbid(unsafe_code)]
//! Thin CLI front-end for Reto-Split.
//!
//! Handles CLI argument parsing, input path discovery/verification, progress reporting,
//! dual-sink logging management, and passes verified file lists to `reto-core::run_batch`.

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use reto_core::{
    clear_retinaface_model_cache, clear_superpoint_model_cache, is_supported_image, run_batch,
    BatchProcessingRequest, ProgressEvent, ProgressObserver,
};
use std::fs::File;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{fmt, EnvFilter, Layer};

/// Command-line arguments for reto-cli.
#[allow(clippy::struct_excessive_bools)]
#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "3D film camera image splitting and alignment tool",
    long_about = None
)]
pub struct Cli {
    /// Path to the input image file or directory for batch processing
    #[arg(short, long, value_name = "PATH")]
    pub input: PathBuf,

    /// Output directory for results and debug visual artifacts
    #[arg(short, long, value_name = "DIR")]
    pub output: PathBuf,

    /// Enable debug mode
    #[arg(long)]
    pub debug: bool,

    /// Inter-frame delay for Wiggle GIF in milliseconds (default: 100ms)
    #[arg(long, default_value_t = reto_core::DEFAULT_FRAME_DELAY_MS, value_name = "MS")]
    pub gif_delay: u32,

    /// Disable Floyd-Steinberg dithering during GIF color quantization
    #[arg(long)]
    pub no_dither: bool,

    /// Enable 24-bit `TrueColor` HEVC MP4 video generation alongside GIF
    #[arg(long)]
    pub enable_mp4: bool,

    /// Force NVIDIA NVENC hardware encoder for HEVC MP4 generation (requires NVIDIA driver)
    #[arg(long)]
    pub enable_nvenc: bool,

    /// Number of continuous ping-pong wiggle loop cycles encoded into the MP4 video (default: 4)
    #[arg(long, default_value_t = reto_core::DEFAULT_MP4_LOOPS, value_name = "COUNT")]
    pub mp4_loops: usize,

    /// Quality / Constant Rate Factor for HEVC video encoding (0-51, lower is higher quality, default: 18)
    #[arg(long, default_value_t = reto_core::DEFAULT_MP4_CRF, value_name = "CRF")]
    pub mp4_crf: u32,

    /// Disable terminal progress bar
    #[arg(long)]
    pub no_progress: bool,

    /// Custom file path for writing detailed execution logs
    #[arg(long, value_name = "FILE")]
    pub log_file: Option<PathBuf>,

    /// Quiet mode (suppress all non-error output)
    #[arg(short, long)]
    pub quiet: bool,

    /// Verbose logging level (-v for debug, -vv for trace)
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,
}

/// Discovers and validates image files from an input path using [`reto_core::is_supported_image`].
///
/// # Arguments
/// * `input_path` - Path to an individual image file or a directory containing scan files.
#[must_use]
pub fn collect_verified_images(input_path: &Path) -> Vec<PathBuf> {
    if input_path.is_file() {
        if is_supported_image(input_path) {
            vec![input_path.to_path_buf()]
        } else {
            Vec::new()
        }
    } else if input_path.is_dir() {
        let mut files = Vec::new();
        if let Ok(entries) = std::fs::read_dir(input_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() && is_supported_image(&path) {
                    files.push(path);
                }
            }
        }
        files.sort();
        files
    } else {
        Vec::new()
    }
}

/// Interactive progress observer backed by `indicatif::ProgressBar`.
#[derive(Debug, Clone)]
struct IndicatifObserver {
    pb: ProgressBar,
}

impl IndicatifObserver {
    #[inline]
    const fn new(pb: ProgressBar) -> Self {
        Self { pb }
    }
}

impl ProgressObserver for IndicatifObserver {
    fn on_progress(&self, event: ProgressEvent<'_>) {
        match event {
            ProgressEvent::ItemStarted { file_stem, .. } => {
                self.pb.set_message(format!("{file_stem}.jpg"));
            }
            ProgressEvent::ItemCompleted { .. } => {
                self.pb.inc(1);
            }
        }
    }
}

/// Non-interactive line-based progress observer for CI/CD and piped streams.
const PROGRESS_STEP_PCT: usize = 25;

#[derive(Debug)]
struct NonTtyProgressObserver {
    completed_count: AtomicUsize,
    last_logged_pct: AtomicUsize,
}

impl NonTtyProgressObserver {
    #[inline]
    const fn new() -> Self {
        Self {
            completed_count: AtomicUsize::new(0),
            last_logged_pct: AtomicUsize::new(0),
        }
    }
}

impl ProgressObserver for NonTtyProgressObserver {
    fn on_progress(&self, event: ProgressEvent<'_>) {
        if let ProgressEvent::ItemCompleted { total, .. } = event {
            let done = self.completed_count.fetch_add(1, Ordering::Relaxed) + 1;
            let pct = (done * 100) / total.max(1);
            let last = self.last_logged_pct.load(Ordering::Relaxed);
            if pct >= last + PROGRESS_STEP_PCT || done == total {
                self.last_logged_pct.store(
                    (pct / PROGRESS_STEP_PCT) * PROGRESS_STEP_PCT,
                    Ordering::Relaxed,
                );
                tracing::info!("Progress: {}/{} ({}%) completed", done, total, pct);
            }
        }
    }
}

fn init_logging(cli: &Cli, is_tty: bool) -> Option<PathBuf> {
    let file_filter_directive = match cli.verbose {
        0 => {
            if cli.debug {
                "debug,ort=info"
            } else {
                "info,ort=warn"
            }
        }
        1 => "debug,ort=info",
        _ => "trace,ort=info",
    };

    let term_filter_directive = if cli.quiet {
        "error"
    } else if cli.verbose > 0 || cli.debug {
        file_filter_directive
    } else if is_tty {
        "warn"
    } else {
        file_filter_directive
    };

    let log_path = cli
        .log_file
        .clone()
        .unwrap_or_else(|| cli.output.join("reto_run.log"));

    let (file_layer, actual_log_file) =
        match std::fs::create_dir_all(&cli.output).and_then(|()| File::create(&log_path)) {
            Ok(file) => {
                let layer = fmt::layer()
                    .with_writer(std::sync::Mutex::new(file))
                    .with_ansi(false)
                    .with_timer(fmt::time::UtcTime::rfc_3339())
                    .with_span_events(fmt::format::FmtSpan::CLOSE)
                    .with_target(true)
                    .with_filter(EnvFilter::new(file_filter_directive));
                (Some(layer), Some(log_path))
            }
            Err(e) => {
                eprintln!(
                    "[WARN] Failed to initialize file logger at {}: {}",
                    log_path.display(),
                    e
                );
                (None, None)
            }
        };

    let (indicatif_layer, tty_term_layer, non_tty_term_layer) = if is_tty {
        let ind = tracing_indicatif::IndicatifLayer::new().with_max_progress_bars(0, None);
        let stderr_writer = ind.get_stderr_writer();
        let ind_layer = ind.with_filter(tracing_indicatif::filter::IndicatifFilter::new(false));
        let term = fmt::layer().with_writer(stderr_writer).with_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new(term_filter_directive)),
        );
        (Some(ind_layer), Some(term), None)
    } else {
        let term = fmt::layer().with_writer(std::io::stderr).with_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new(term_filter_directive)),
        );
        (None, None, Some(term))
    };

    tracing_subscriber::registry()
        .with(indicatif_layer)
        .with(tty_term_layer)
        .with(non_tty_term_layer)
        .with(file_layer)
        .init();

    actual_log_file
}

fn build_batch_request(
    cli: &Cli,
    files: Vec<PathBuf>,
    is_tty: bool,
) -> anyhow::Result<(BatchProcessingRequest, Option<ProgressBar>)> {
    let total_files = files.len();
    let gif_config = reto_core::WiggleGifConfig::new(cli.gif_delay).with_dither(!cli.no_dither);
    let video_config = if cli.enable_mp4 || cli.enable_nvenc {
        let v_cfg = reto_core::WiggleVideoConfig::new()
            .with_loops(cli.mp4_loops)
            .with_crf(cli.mp4_crf)
            .with_nvenc(cli.enable_nvenc);

        match reto_core::probe_video_encoder_backend(&v_cfg) {
            Ok(backend) => {
                tracing::info!(
                    backend = %backend,
                    loops = v_cfg.loops,
                    crf = v_cfg.crf,
                    "Hardware HEVC MP4 video encoding enabled"
                );
            }
            Err(e) => {
                anyhow::bail!(
                    "MP4 video export requested (--enable-mp4), but no available HEVC video encoder backend was found on this host: {e}"
                );
            }
        }
        Some(v_cfg)
    } else {
        None
    };

    let mut request = BatchProcessingRequest::new(files, cli.output.clone(), cli.debug)
        .with_gif_config(gif_config)
        .with_video_config(video_config);

    let progress_bar = if is_tty {
        let pb = ProgressBar::new(total_files as u64);
        pb.set_draw_target(indicatif::ProgressDrawTarget::stderr_with_hz(20));
        let style = ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:24.cyan/blue}] {pos}/{len} ({percent}%) ETA: {eta} | {wide_msg}")
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("━╸─")
            .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏");
        pb.set_style(style);
        pb.enable_steady_tick(std::time::Duration::from_millis(120));
        request = request.with_progress_observer(Arc::new(IndicatifObserver::new(pb.clone())));
        Some(pb)
    } else if !cli.quiet && !cli.no_progress {
        request = request.with_progress_observer(Arc::new(NonTtyProgressObserver::new()));
        None
    } else {
        None
    };

    Ok((request, progress_bar))
}

fn print_summary(
    cli: &Cli,
    summary: &reto_core::ProcessSummary,
    elapsed_secs: f64,
    is_tty: bool,
    actual_log_file: Option<&Path>,
) {
    if !cli.quiet {
        let prefix = if is_tty { "✔ [OK]" } else { "[OK]" };
        println!(
            "{prefix} Processed {} images ({} succeeded, {} failed) in {:.1}s.",
            summary.total_input, summary.successful_count, summary.failed_count, elapsed_secs
        );
        println!("  • Output Directory: {}", cli.output.display());
        if let Some(log_file) = actual_log_file {
            println!("  • Execution Log:    {}", log_file.display());
        }
    }

    if summary.failed_count > 0 {
        tracing::warn!(
            succeeded = summary.successful_count,
            failed = summary.failed_count,
            "Completed with some failures"
        );
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let is_tty = std::io::stderr().is_terminal() && !cli.no_progress && !cli.quiet;
    let actual_log_file = init_logging(&cli, is_tty);

    let files = collect_verified_images(&cli.input);
    if files.is_empty() {
        tracing::warn!(input = ?cli.input, "No valid image files found matching supported extensions");
        return Ok(());
    }

    tracing::info!(
        discovered_count = files.len(),
        input = ?cli.input,
        "Verified input files for processing"
    );

    let (request, progress_bar) = build_batch_request(&cli, files, is_tty)?;
    let start_time = Instant::now();
    let summary = run_batch(&request)?;
    let elapsed = start_time.elapsed();

    if let Some(pb) = progress_bar {
        pb.finish_and_clear();
    }

    print_summary(
        &cli,
        &summary,
        elapsed.as_secs_f64(),
        is_tty,
        actual_log_file.as_deref(),
    );

    clear_superpoint_model_cache();
    clear_retinaface_model_cache();

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parsing_options() {
        let args = vec![
            "reto-cli",
            "-i",
            "scans/",
            "-o",
            "output/",
            "--no-progress",
            "--quiet",
            "--log-file",
            "custom.log",
        ];
        let cli = Cli::try_parse_from(args).expect("Should parse valid arguments");
        assert_eq!(cli.input, PathBuf::from("scans/"));
        assert_eq!(cli.output, PathBuf::from("output/"));
        assert!(cli.no_progress);
        assert!(cli.quiet);
        assert_eq!(cli.log_file, Some(PathBuf::from("custom.log")));
        assert!(!cli.enable_mp4);
        assert!(!cli.enable_nvenc);
        assert_eq!(cli.mp4_loops, 4);
        assert_eq!(cli.mp4_crf, 18);
    }

    #[test]
    fn test_cli_parsing_mp4_options() {
        let args = vec![
            "reto-cli",
            "-i",
            "scans/",
            "-o",
            "output/",
            "--enable-mp4",
            "--enable-nvenc",
            "--mp4-loops",
            "6",
            "--mp4-crf",
            "22",
        ];
        let cli = Cli::try_parse_from(args).expect("Should parse MP4 arguments");
        assert!(cli.enable_mp4);
        assert!(cli.enable_nvenc);
        assert_eq!(cli.mp4_loops, 6);
        assert_eq!(cli.mp4_crf, 22);
    }

    #[test]
    fn test_collect_verified_images_empty_dir() {
        let temp_dir = std::env::temp_dir().join("reto_cli_test_collect");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        let files = collect_verified_images(&temp_dir);
        assert!(files.is_empty());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
