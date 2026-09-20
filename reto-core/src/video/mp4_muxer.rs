//! Pure Rust ISO Base Media File Format (MP4) container multiplexer for HEVC (H.265) video streams.
//!
//! Encapsulates HEVC parameter sets (VPS, SPS, PPS) and sample frames into standard
//! `mp42`/`isom`/`hvc1` containers with variable $\mathrm{SE}(3)$ timing preservation (`stts`).

#![allow(
    clippy::similar_names,
    clippy::cast_possible_truncation,
    clippy::unreadable_literal,
    clippy::doc_markdown,
    clippy::trivially_copy_pass_by_ref
)]

use crate::error::{Error, Result};
use std::io::Write;

/// HEVC NAL Unit Types per ITU-T H.265 / ISO/IEC 23008-2.
/// Trailing non-reference picture slice.
pub const NAL_TRAIL_N: u8 = 0;
/// Trailing reference picture slice (standard P/B frame).
pub const NAL_TRAIL_R: u8 = 1;
/// IDR picture with RADL slices.
pub const NAL_IDR_W_RADL: u8 = 19;
/// IDR picture with leading picture slices.
pub const NAL_IDR_N_LP: u8 = 20;
/// Clean Random Access (CRA) picture slice.
pub const NAL_CRA_NUT: u8 = 21;
/// Video Parameter Set (VPS).
pub const NAL_VPS: u8 = 32;
/// Sequence Parameter Set (SPS).
pub const NAL_SPS: u8 = 33;
/// Picture Parameter Set (PPS).
pub const NAL_PPS: u8 = 34;
/// Access Unit Delimiter (AUD).
pub const NAL_AUD: u8 = 35;
/// Prefix Supplemental Enhancement Information (SEI).
pub const NAL_PREFIX_SEI: u8 = 39;
/// Suffix Supplemental Enhancement Information (SEI).
pub const NAL_SUFFIX_SEI: u8 = 40;

/// Individual HEVC Network Abstraction Layer (NAL) Unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HevcNalUnit {
    /// NAL unit type (0..=63).
    pub nal_type: u8,
    /// Raw NAL unit payload bytes (without Annex B start code prefix).
    pub data: Vec<u8>,
    /// Whether this NAL unit corresponds to an IDR / Instantaneous Decoder Refresh keyframe.
    pub is_keyframe: bool,
}

impl HevcNalUnit {
    /// Creates a new NAL unit by parsing the NAL unit header byte.
    #[must_use]
    pub fn from_bytes(data: Vec<u8>) -> Self {
        if data.is_empty() {
            return Self {
                nal_type: 0,
                data,
                is_keyframe: false,
            };
        }
        // HEVC NAL header: forbidden_zero_bit (1 bit), nal_unit_type (6 bits), nuh_layer_id (6 bits), nuh_temporal_id_plus1 (3 bits)
        let nal_type = (data[0] >> 1) & 0x3F;
        let is_keyframe = matches!(nal_type, NAL_IDR_W_RADL | NAL_IDR_N_LP | NAL_CRA_NUT);
        Self {
            nal_type,
            data,
            is_keyframe,
        }
    }
}

/// Helper function to split Annex B formatted byte streams (`00 00 01` or `00 00 00 01`) into distinct NAL units.
#[must_use]
pub fn parse_annex_b_nalus(stream: &[u8]) -> Vec<HevcNalUnit> {
    let mut nalus = Vec::new();
    let len = stream.len();
    let mut i = 0;

    while i < len {
        // Find start code prefix
        let start_len = if i + 4 <= len
            && stream[i] == 0
            && stream[i + 1] == 0
            && stream[i + 2] == 0
            && stream[i + 3] == 1
        {
            4
        } else if i + 3 <= len && stream[i] == 0 && stream[i + 1] == 0 && stream[i + 2] == 1 {
            3
        } else {
            i += 1;
            continue;
        };

        let nal_start = i + start_len;
        let mut next_i = nal_start;

        while next_i < len {
            if (next_i + 3 <= len
                && stream[next_i] == 0
                && stream[next_i + 1] == 0
                && stream[next_i + 2] == 1)
                || (next_i + 4 <= len
                    && stream[next_i] == 0
                    && stream[next_i + 1] == 0
                    && stream[next_i + 2] == 0
                    && stream[next_i + 3] == 1)
            {
                break;
            }
            next_i += 1;
        }

        if nal_start < next_i {
            let nal_bytes = stream[nal_start..next_i].to_vec();
            nalus.push(HevcNalUnit::from_bytes(nal_bytes));
        }

        i = next_i;
    }

    nalus
}

/// Single encoded video frame consisting of one or more NAL units and display duration.
#[derive(Debug, Clone)]
pub struct EncodedVideoSample {
    /// NAL units associated with this frame sample (e.g. SEI, IDR/P slice).
    pub nalus: Vec<HevcNalUnit>,
    /// Duration of this frame sample in milliseconds ($\Delta t_i$).
    pub duration_ms: u32,
    /// Whether this sample is a random access sync sample (keyframe).
    pub is_sync: bool,
}

/// Pure Rust MP4 Container Multiplexer.
pub struct Mp4Muxer;

impl Mp4Muxer {
    /// Multiplexes HEVC parameter sets and video samples into an ISO BMFF (`.mp4`) container.
    ///
    /// # Arguments
    /// * `width` - Video width in pixels.
    /// * `height` - Video height in pixels.
    /// * `vps` - Video Parameter Set NAL unit data.
    /// * `sps` - Sequence Parameter Set NAL unit data.
    /// * `pps` - Picture Parameter Set NAL unit data.
    /// * `samples` - Ordered list of encoded video samples with non-uniform display durations.
    /// * `writer` - Output byte stream writer.
    ///
    /// # Errors
    /// Returns [`Error`] if required parameter sets are missing or writing fails.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn mux_hevc<W: Write>(
        width: u32,
        height: u32,
        vps: &[u8],
        sps: &[u8],
        pps: &[u8],
        samples: &[EncodedVideoSample],
        writer: &mut W,
    ) -> Result<()> {
        if samples.is_empty() {
            return Err(Error::Unknown(
                "Cannot mux MP4 with zero video samples".into(),
            ));
        }

        let total_duration_ms: u64 = samples.iter().map(|s| u64::from(s.duration_ms)).sum();
        let ftyp_box = build_ftyp_box();
        let (mdat_box, sample_sizes, sync_samples) = build_mdat_and_sample_tables(samples);

        let timescale = 1000u32;
        let movie_duration = total_duration_ms as u32;

        let mvhd_box = build_mvhd_box(timescale, movie_duration);
        let tkhd_box = build_tkhd_box(width, height, movie_duration);
        let mdhd_box = build_mdhd_box(timescale, movie_duration);
        let hdlr_box = build_hdlr_box();
        let vmhd_box = build_vmhd_box();
        let dinf_box = build_dinf_box();
        let hvcc_box = build_hvcc_box(vps, sps, pps);
        let stsd_box = build_stsd_box(width, height, &hvcc_box);
        let stts_box = build_stts_box(samples);
        let stss_box = build_stss_box(&sync_samples);
        let stsc_box = build_stsc_box(samples.len());
        let stsz_box = build_stsz_box(&sample_sizes);

        let stbl_boxes = [
            &stsd_box[..],
            &stts_box[..],
            &stss_box[..],
            &stsc_box[..],
            &stsz_box[..],
        ];

        let ftyp_len = ftyp_box.len() as u32;
        let dummy_moov = assemble_moov_container(
            &mvhd_box,
            &tkhd_box,
            &mdhd_box,
            &hdlr_box,
            &vmhd_box,
            &dinf_box,
            &stbl_boxes,
            0,
        );
        let moov_len = dummy_moov.len() as u32;
        let mdat_payload_offset = ftyp_len + moov_len + 8;

        let final_moov = assemble_moov_container(
            &mvhd_box,
            &tkhd_box,
            &mdhd_box,
            &hdlr_box,
            &vmhd_box,
            &dinf_box,
            &stbl_boxes,
            mdat_payload_offset,
        );

        writer.write_all(&ftyp_box)?;
        writer.write_all(&final_moov)?;
        writer.write_all(&mdat_box)?;
        writer.flush()?;

        Ok(())
    }
}

fn build_ftyp_box() -> Vec<u8> {
    let mut ftyp = Vec::new();
    ftyp.extend_from_slice(b"mp42"); // major brand
    ftyp.extend_from_slice(&0u32.to_be_bytes()); // minor version
    ftyp.extend_from_slice(b"isom");
    ftyp.extend_from_slice(b"mp42");
    ftyp.extend_from_slice(b"hvc1");
    write_box(b"ftyp", &ftyp)
}

#[allow(clippy::cast_possible_truncation)]
fn build_mdat_and_sample_tables(samples: &[EncodedVideoSample]) -> (Vec<u8>, Vec<u32>, Vec<u32>) {
    let mut mdat_payload = Vec::new();
    let mut sample_sizes = Vec::with_capacity(samples.len());
    let mut sync_samples = Vec::new();

    for (idx, sample) in samples.iter().enumerate() {
        let sample_offset_start = mdat_payload.len();
        if sample.is_sync || sample.nalus.iter().any(|n| n.is_keyframe) {
            sync_samples.push((idx + 1) as u32); // 1-based index for stss
        }

        for nalu in &sample.nalus {
            if nalu.nal_type == NAL_VPS
                || nalu.nal_type == NAL_SPS
                || nalu.nal_type == NAL_PPS
                || nalu.nal_type == NAL_AUD
            {
                continue;
            }
            let nalu_len = nalu.data.len() as u32;
            mdat_payload.extend_from_slice(&nalu_len.to_be_bytes());
            mdat_payload.extend_from_slice(&nalu.data);
        }

        let sample_size = (mdat_payload.len() - sample_offset_start) as u32;
        sample_sizes.push(sample_size);
    }

    let mdat_box = write_box(b"mdat", &mdat_payload);
    (mdat_box, sample_sizes, sync_samples)
}

fn build_mvhd_box(timescale: u32, movie_duration: u32) -> Vec<u8> {
    let mut mvhd = Vec::new();
    mvhd.push(0); // version 0
    mvhd.extend_from_slice(&[0, 0, 0]); // flags
    mvhd.extend_from_slice(&0u32.to_be_bytes()); // creation time
    mvhd.extend_from_slice(&0u32.to_be_bytes()); // modification time
    mvhd.extend_from_slice(&timescale.to_be_bytes()); // timescale
    mvhd.extend_from_slice(&movie_duration.to_be_bytes()); // duration
    mvhd.extend_from_slice(&0x00010000u32.to_be_bytes()); // rate = 1.0
    mvhd.extend_from_slice(&0x0100u16.to_be_bytes()); // volume = 1.0 (full volume)
    mvhd.extend_from_slice(&[0u8; 10]); // reserved
    let matrix: [u32; 9] = [0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000];
    for val in matrix {
        mvhd.extend_from_slice(&val.to_be_bytes());
    }
    mvhd.extend_from_slice(&[0u8; 24]); // pre_defined
    mvhd.extend_from_slice(&2u32.to_be_bytes()); // next_track_ID = 2
    write_box(b"mvhd", &mvhd)
}

fn build_tkhd_box(width: u32, height: u32, movie_duration: u32) -> Vec<u8> {
    let mut tkhd = Vec::new();
    tkhd.push(0); // version 0
    tkhd.extend_from_slice(&[0, 0, 7]); // flags: track enabled (1) | in movie (2) | in preview (4)
    tkhd.extend_from_slice(&0u32.to_be_bytes()); // creation time
    tkhd.extend_from_slice(&0u32.to_be_bytes()); // modification time
    tkhd.extend_from_slice(&1u32.to_be_bytes()); // track ID = 1
    tkhd.extend_from_slice(&0u32.to_be_bytes()); // reserved
    tkhd.extend_from_slice(&movie_duration.to_be_bytes()); // duration
    tkhd.extend_from_slice(&[0u8; 8]); // reserved
    tkhd.extend_from_slice(&0u16.to_be_bytes()); // layer = 0
    tkhd.extend_from_slice(&0u16.to_be_bytes()); // alternate group = 0
    tkhd.extend_from_slice(&0u16.to_be_bytes()); // volume = 0 (video)
    tkhd.extend_from_slice(&0u16.to_be_bytes()); // reserved
    let matrix: [u32; 9] = [0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000];
    for val in matrix {
        tkhd.extend_from_slice(&val.to_be_bytes());
    }
    tkhd.extend_from_slice(&(width << 16).to_be_bytes()); // width in 16.16 fixed point
    tkhd.extend_from_slice(&(height << 16).to_be_bytes()); // height in 16.16 fixed point
    write_box(b"tkhd", &tkhd)
}

fn build_mdhd_box(timescale: u32, movie_duration: u32) -> Vec<u8> {
    let mut mdhd = Vec::new();
    mdhd.push(0); // version 0
    mdhd.extend_from_slice(&[0, 0, 0]); // flags
    mdhd.extend_from_slice(&0u32.to_be_bytes()); // creation time
    mdhd.extend_from_slice(&0u32.to_be_bytes()); // modification time
    mdhd.extend_from_slice(&timescale.to_be_bytes()); // timescale = 1000
    mdhd.extend_from_slice(&movie_duration.to_be_bytes()); // duration
    mdhd.extend_from_slice(&0x55C4u16.to_be_bytes()); // language = und (undefined)
    mdhd.extend_from_slice(&0u16.to_be_bytes()); // pre_defined
    write_box(b"mdhd", &mdhd)
}

fn build_hdlr_box() -> Vec<u8> {
    let mut hdlr = Vec::new();
    hdlr.push(0); // version
    hdlr.extend_from_slice(&[0, 0, 0]); // flags
    hdlr.extend_from_slice(&0u32.to_be_bytes()); // pre_defined
    hdlr.extend_from_slice(b"vide"); // handler_type
    hdlr.extend_from_slice(&[0u8; 12]); // reserved
    hdlr.extend_from_slice(b"VideoHandler\0"); // name
    write_box(b"hdlr", &hdlr)
}

fn build_vmhd_box() -> Vec<u8> {
    let mut vmhd = Vec::new();
    vmhd.push(0); // version
    vmhd.extend_from_slice(&[0, 0, 1]); // flags = 1
    vmhd.extend_from_slice(&0u16.to_be_bytes()); // graphicsmode = copy
    vmhd.extend_from_slice(&[0u8; 6]); // opcolor = [0, 0, 0]
    write_box(b"vmhd", &vmhd)
}

fn build_dinf_box() -> Vec<u8> {
    let mut dref_inner = Vec::new();
    dref_inner.push(0); // version
    dref_inner.extend_from_slice(&[0, 0, 0]); // flags
    dref_inner.extend_from_slice(&1u32.to_be_bytes()); // entry count = 1
    let mut url_box = Vec::new();
    url_box.push(0); // version
    url_box.extend_from_slice(&[0, 0, 1]); // flags = 1 (self-contained)
    dref_inner.extend_from_slice(&write_box(b"url ", &url_box));
    let dref_box = write_box(b"dref", &dref_inner);
    write_box(b"dinf", &dref_box)
}

#[allow(clippy::cast_possible_truncation)]
fn build_hvcc_box(vps: &[u8], sps: &[u8], pps: &[u8]) -> Vec<u8> {
    let mut hvcc = Vec::new();
    hvcc.push(1); // configurationVersion = 1
    let profile_idc = if sps.len() > 1 {
        (sps[1] >> 1) & 0x1F
    } else {
        1
    };
    hvcc.push(profile_idc.clamp(1, 2)); // Main Profile
    hvcc.extend_from_slice(&0x60000000u32.to_be_bytes()); // general_profile_compatibility_flags
    hvcc.extend_from_slice(&[0u8; 6]); // general_constraint_indicator_flags
    let level_idc = if sps.len() > 12 { sps[12] } else { 120 };
    hvcc.push(level_idc); // general_level_idc
    hvcc.extend_from_slice(&0xF000u16.to_be_bytes()); // min_spatial_segmentation_idc
    hvcc.push(0xFC); // parallelismType
    hvcc.push(0xFD); // chroma_format_idc (reserved 6 bits + 1 for 4:2:0)
    hvcc.push(0xF8); // bit_depth_luma_minus8
    hvcc.push(0xF8); // bit_depth_chroma_minus8
    hvcc.extend_from_slice(&0u16.to_be_bytes()); // avgFrameRate = 0
    hvcc.push(0x0F);
    hvcc.push(3); // Arrays count: 3 (VPS, SPS, PPS)

    // Array 1: VPS (NAL 32)
    hvcc.push(0x80 | NAL_VPS);
    hvcc.extend_from_slice(&1u16.to_be_bytes());
    hvcc.extend_from_slice(&(vps.len() as u16).to_be_bytes());
    hvcc.extend_from_slice(vps);

    // Array 2: SPS (NAL 33)
    hvcc.push(0x80 | NAL_SPS);
    hvcc.extend_from_slice(&1u16.to_be_bytes());
    hvcc.extend_from_slice(&(sps.len() as u16).to_be_bytes());
    hvcc.extend_from_slice(sps);

    // Array 3: PPS (NAL 34)
    hvcc.push(0x80 | NAL_PPS);
    hvcc.extend_from_slice(&1u16.to_be_bytes());
    hvcc.extend_from_slice(&(pps.len() as u16).to_be_bytes());
    hvcc.extend_from_slice(pps);

    write_box(b"hvcC", &hvcc)
}

#[allow(clippy::cast_possible_truncation)]
fn build_stsd_box(width: u32, height: u32, hvcc_box: &[u8]) -> Vec<u8> {
    let mut hvc1 = Vec::new();
    hvc1.extend_from_slice(&[0u8; 6]); // reserved
    hvc1.extend_from_slice(&1u16.to_be_bytes()); // data_reference_index = 1
    hvc1.extend_from_slice(&0u16.to_be_bytes()); // pre_defined
    hvc1.extend_from_slice(&0u16.to_be_bytes()); // reserved
    hvc1.extend_from_slice(&[0u8; 12]); // pre_defined
    hvc1.extend_from_slice(&(width as u16).to_be_bytes());
    hvc1.extend_from_slice(&(height as u16).to_be_bytes());
    hvc1.extend_from_slice(&0x00480000u32.to_be_bytes()); // 72 dpi horiz
    hvc1.extend_from_slice(&0x00480000u32.to_be_bytes()); // 72 dpi vert
    hvc1.extend_from_slice(&0u32.to_be_bytes()); // reserved
    hvc1.extend_from_slice(&1u16.to_be_bytes()); // frame_count = 1
    hvc1.extend_from_slice(&[0u8; 32]); // compressorname
    hvc1.extend_from_slice(&0x0018u16.to_be_bytes()); // depth = 24-bit TrueColor
    hvc1.extend_from_slice(&(-1i16).to_be_bytes()); // pre_defined = -1
    hvc1.extend_from_slice(hvcc_box);
    let hvc1_box = write_box(b"hvc1", &hvc1);

    let mut stsd = Vec::new();
    stsd.push(0); // version
    stsd.extend_from_slice(&[0, 0, 0]); // flags
    stsd.extend_from_slice(&1u32.to_be_bytes()); // entry_count = 1
    stsd.extend_from_slice(&hvc1_box);
    write_box(b"stsd", &stsd)
}

#[allow(clippy::cast_possible_truncation)]
fn build_stts_box(samples: &[EncodedVideoSample]) -> Vec<u8> {
    let mut stts_entries: Vec<(u32, u32)> = Vec::new(); // (sample_count, sample_delta)
    for sample in samples {
        let dur = sample.duration_ms.max(1);
        if let Some(last) = stts_entries.last_mut() {
            if last.1 == dur {
                last.0 += 1;
                continue;
            }
        }
        stts_entries.push((1, dur));
    }

    let mut stts = Vec::new();
    stts.push(0); // version
    stts.extend_from_slice(&[0, 0, 0]); // flags
    stts.extend_from_slice(&(stts_entries.len() as u32).to_be_bytes());
    for (cnt, delta) in stts_entries {
        stts.extend_from_slice(&cnt.to_be_bytes());
        stts.extend_from_slice(&delta.to_be_bytes());
    }
    write_box(b"stts", &stts)
}

#[allow(clippy::cast_possible_truncation)]
fn build_stss_box(sync_samples: &[u32]) -> Vec<u8> {
    let mut stss = Vec::new();
    stss.push(0); // version
    stss.extend_from_slice(&[0, 0, 0]); // flags
    let effective_sync = if sync_samples.is_empty() {
        vec![1u32]
    } else {
        sync_samples.to_vec()
    };
    stss.extend_from_slice(&(effective_sync.len() as u32).to_be_bytes());
    for sync_idx in effective_sync {
        stss.extend_from_slice(&sync_idx.to_be_bytes());
    }
    write_box(b"stss", &stss)
}

#[allow(clippy::cast_possible_truncation)]
fn build_stsc_box(total_samples: usize) -> Vec<u8> {
    let mut stsc = Vec::new();
    stsc.push(0); // version
    stsc.extend_from_slice(&[0, 0, 0]); // flags
    stsc.extend_from_slice(&1u32.to_be_bytes()); // entry count = 1
    stsc.extend_from_slice(&1u32.to_be_bytes()); // first_chunk = 1
    stsc.extend_from_slice(&(total_samples as u32).to_be_bytes()); // samples_per_chunk = total
    stsc.extend_from_slice(&1u32.to_be_bytes()); // sample_description_index = 1
    write_box(b"stsc", &stsc)
}

#[allow(clippy::cast_possible_truncation)]
fn build_stsz_box(sample_sizes: &[u32]) -> Vec<u8> {
    let mut stsz = Vec::new();
    stsz.push(0); // version
    stsz.extend_from_slice(&[0, 0, 0]); // flags
    stsz.extend_from_slice(&0u32.to_be_bytes()); // sample_size = 0 (variable)
    stsz.extend_from_slice(&(sample_sizes.len() as u32).to_be_bytes());
    for &sz in sample_sizes {
        stsz.extend_from_slice(&sz.to_be_bytes());
    }
    write_box(b"stsz", &stsz)
}

#[allow(clippy::too_many_arguments)]
fn assemble_moov_container(
    mvhd_box: &[u8],
    tkhd_box: &[u8],
    mdhd_box: &[u8],
    hdlr_box: &[u8],
    vmhd_box: &[u8],
    dinf_box: &[u8],
    stbl_inner_boxes: &[&[u8]],
    chunk_offset: u32,
) -> Vec<u8> {
    let mut stco = Vec::new();
    stco.push(0); // version
    stco.extend_from_slice(&[0, 0, 0]); // flags
    stco.extend_from_slice(&1u32.to_be_bytes()); // entry count = 1 chunk
    stco.extend_from_slice(&chunk_offset.to_be_bytes());
    let stco_box = write_box(b"stco", &stco);

    let mut stbl = Vec::new();
    for &box_data in stbl_inner_boxes {
        stbl.extend_from_slice(box_data);
    }
    stbl.extend_from_slice(&stco_box);
    let stbl_box = write_box(b"stbl", &stbl);

    let mut minf = Vec::new();
    minf.extend_from_slice(vmhd_box);
    minf.extend_from_slice(dinf_box);
    minf.extend_from_slice(&stbl_box);
    let minf_box = write_box(b"minf", &minf);

    let mut mdia = Vec::new();
    mdia.extend_from_slice(mdhd_box);
    mdia.extend_from_slice(hdlr_box);
    mdia.extend_from_slice(&minf_box);
    let mdia_box = write_box(b"mdia", &mdia);

    let mut trak = Vec::new();
    trak.extend_from_slice(tkhd_box);
    trak.extend_from_slice(&mdia_box);
    let trak_box = write_box(b"trak", &trak);

    let mut moov = Vec::new();
    moov.extend_from_slice(mvhd_box);
    moov.extend_from_slice(&trak_box);
    write_box(b"moov", &moov)
}

/// Helper function to create an ISO BMFF Box with 4-byte big-endian size header.
fn write_box(box_type: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let size = (payload.len() + 8) as u32;
    let mut out = Vec::with_capacity(size as usize);
    out.extend_from_slice(&size.to_be_bytes());
    out.extend_from_slice(box_type);
    out.extend_from_slice(payload);
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_annex_b_parsing() {
        let stream = vec![
            0, 0, 0, 1, 0x40, 0x01, 0x0C, // VPS (32)
            0, 0, 0, 1, 0x42, 0x01, 0x01, // SPS (33)
            0, 0, 1, 0x44, 0x01, // PPS (34)
            0, 0, 0, 1, 0x26, 0x01, // IDR_W_RADL (19)
        ];
        let nalus = parse_annex_b_nalus(&stream);
        assert_eq!(nalus.len(), 4);
        assert_eq!(nalus[0].nal_type, NAL_VPS);
        assert_eq!(nalus[1].nal_type, NAL_SPS);
        assert_eq!(nalus[2].nal_type, NAL_PPS);
        assert_eq!(nalus[3].nal_type, NAL_IDR_W_RADL);
        assert!(nalus[3].is_keyframe);
    }

    #[test]
    fn test_mp4_muxing_structure() {
        let vps = vec![
            0x40, 0x01, 0x0C, 0x01, 0xFF, 0xFF, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x00,
        ];
        let sps = vec![
            0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03, 0x00, 0x78,
        ];
        let pps = vec![0x44, 0x01, 0xC0, 0xF3, 0xC0];

        let samples = vec![
            EncodedVideoSample {
                nalus: vec![HevcNalUnit::from_bytes(vec![0x26, 0x01, 0xAA, 0xBB])],
                duration_ms: 100,
                is_sync: true,
            },
            EncodedVideoSample {
                nalus: vec![HevcNalUnit::from_bytes(vec![0x02, 0x01, 0xCC, 0xDD])],
                duration_ms: 100,
                is_sync: false,
            },
        ];

        let mut out = Vec::new();
        Mp4Muxer::mux_hevc(640, 480, &vps, &sps, &pps, &samples, &mut out)
            .expect("Muxing should succeed");

        assert!(out.len() > 100);
        assert_eq!(&out[4..8], b"ftyp");
        // Verify moov presence
        let out_str = String::from_utf8_lossy(&out);
        assert!(out_str.contains("moov"));
        assert!(out_str.contains("mdat"));
    }
}
