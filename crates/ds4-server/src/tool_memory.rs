const KTM_HEADER: &[u8; 4] = b"KTM\x01";
const DEFAULT_MAX_IDS: usize = 100_000;
const MAX_ID_BYTES: usize = 256;
const MAX_DSML_BYTES: usize = 512 * 1024 * 1024;

fn wire_lengths(id: &[u8], dsml: &[u8]) -> Option<(u32, u32)> {
    if id.is_empty() || dsml.is_empty() || id.contains(&0) || dsml.contains(&0) {
        return None;
    }
    Some((
        u32::try_from(id.len()).ok()?,
        u32::try_from(dsml.len()).ok()?,
    ))
}

pub(crate) fn encode_ktm(entries: &[(&[u8], &[u8])]) -> Option<Vec<u8>> {
    let mut count = 0u32;
    let mut bytes = 8u64;
    for &(id, dsml) in entries {
        let Some((id_len, dsml_len)) = wire_lengths(id, dsml) else {
            continue;
        };
        count = count.checked_add(1)?;
        bytes = bytes
            .checked_add(8)?
            .checked_add(u64::from(id_len))?
            .checked_add(u64::from(dsml_len))?;
    }
    if count == 0 {
        return Some(Vec::new());
    }

    let mut out = Vec::new();
    out.try_reserve_exact(usize::try_from(bytes).ok()?).ok()?;
    out.extend_from_slice(KTM_HEADER);
    out.extend_from_slice(&count.to_le_bytes());
    for &(id, dsml) in entries {
        let Some((id_len, dsml_len)) = wire_lengths(id, dsml) else {
            continue;
        };
        out.extend_from_slice(&id_len.to_le_bytes());
        out.extend_from_slice(&dsml_len.to_le_bytes());
        out.extend_from_slice(id);
        out.extend_from_slice(dsml);
    }
    Some(out)
}

pub(crate) fn decode_ktm<F>(bytes: &[u8], max_ids: usize, mut accept: F) -> usize
where
    F: FnMut(&[u8], &[u8]) -> bool,
{
    if bytes.len() < 8 || &bytes[..4] != KTM_HEADER {
        return 0;
    }
    let count = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let max_ids = if max_ids == 0 {
        DEFAULT_MAX_IDS
    } else {
        max_ids
    };
    if u64::from(count) > u64::try_from(max_ids).unwrap_or(u64::MAX).saturating_mul(4) {
        return 0;
    }

    let mut loaded = 0;
    let mut pos = 8usize;
    for _ in 0..count {
        let Some(lens_end) = pos.checked_add(8).filter(|&end| end <= bytes.len()) else {
            return loaded;
        };
        let id_len = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        let dsml_len = u32::from_le_bytes(bytes[pos + 4..lens_end].try_into().unwrap()) as usize;
        if id_len == 0 || id_len > MAX_ID_BYTES || dsml_len == 0 || dsml_len > MAX_DSML_BYTES {
            return loaded;
        }
        pos = lens_end;
        let Some(id_end) = pos.checked_add(id_len).filter(|&end| end <= bytes.len()) else {
            return loaded;
        };
        let Some(dsml_end) = id_end
            .checked_add(dsml_len)
            .filter(|&end| end <= bytes.len())
        else {
            return loaded;
        };
        let id = &bytes[pos..id_end];
        let id = &id[..id.iter().position(|&byte| byte == 0).unwrap_or(id.len())];
        let dsml = &bytes[id_end..dsml_end];
        let dsml = &dsml[..dsml
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(dsml.len())];
        if accept(id, dsml) {
            loaded += 1;
        }
        pos = dsml_end;
    }
    loaded
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};
    use std::path::PathBuf;
    use std::process::Command;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn oracle() -> PathBuf {
        std::env::var("DS4_KV_C_ORACLE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/kv_c_oracle")
            })
    }

    #[test]
    fn ktm_wire_uses_binary_version_and_little_endian_lengths() {
        let got = encode_ktm(&[(b"call_a", b"<tool>")]).unwrap();
        let mut expected = b"KTM\x01\x01\0\0\0\x06\0\0\0\x06\0\0\0".to_vec();
        expected.extend_from_slice(b"call_a<tool>");
        assert_eq!(got, expected);
        assert_eq!(encode_ktm(&[]).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn ktm_codec_matches_c_oracle() {
        let entries = [
            (b"call_a".as_slice(), b"<tool>one</tool>".as_slice()),
            (b"call_b".as_slice(), b"<tool>two</tool>".as_slice()),
        ];
        let rust = encode_ktm(&entries).unwrap();
        let output = Command::new(oracle())
            .args([
                "ktm-encode",
                &hex(entries[0].0),
                &hex(entries[0].1),
                &hex(entries[1].0),
                &hex(entries[1].1),
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(String::from_utf8(output.stdout).unwrap().trim(), hex(&rust));

        let output = Command::new(oracle())
            .args(["ktm-decode", &hex(&rust), "100000"])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            format!(
                "{}:{}\n{}:{}\nloaded=2\n",
                hex(entries[0].0),
                hex(entries[0].1),
                hex(entries[1].0),
                hex(entries[1].1)
            )
        );
    }

    #[test]
    fn ktm_decode_keeps_valid_prefix_and_filters_wanted() {
        let mut bytes = encode_ktm(&[(b"keep", b"one"), (b"want", b"two")]).unwrap();
        bytes.pop();
        let mut got = Vec::new();
        assert_eq!(
            decode_ktm(&bytes, 100_000, |id, dsml| {
                got.push((id.to_vec(), dsml.to_vec()));
                true
            }),
            1
        );
        assert_eq!(got, vec![(b"keep".to_vec(), b"one".to_vec())]);

        let bytes = encode_ktm(&[(b"skip", b"one"), (b"want", b"two")]).unwrap();
        let wanted = HashSet::from([b"want".as_slice()]);
        let mut got = Vec::new();
        assert_eq!(
            decode_ktm(&bytes, 100_000, |id, dsml| {
                if !wanted.contains(id) {
                    return false;
                }
                got.push((id.to_vec(), dsml.to_vec()));
                true
            }),
            1
        );
        assert_eq!(got, vec![(b"want".to_vec(), b"two".to_vec())]);
    }

    #[test]
    fn ktm_decode_rejects_bad_header_count_and_lengths() {
        assert_eq!(decode_ktm(b"KTM1\0\0\0\0", 100_000, |_, _| true), 0);

        let mut over_count = b"KTM\x01".to_vec();
        over_count.extend_from_slice(&400_001u32.to_le_bytes());
        assert_eq!(decode_ktm(&over_count, 0, |_, _| true), 0);

        let mut zero_id = b"KTM\x01\x01\0\0\0".to_vec();
        zero_id.extend_from_slice(&0u32.to_le_bytes());
        zero_id.extend_from_slice(&1u32.to_le_bytes());
        zero_id.push(b'x');
        assert_eq!(decode_ktm(&zero_id, 100_000, |_, _| true), 0);

        let mut over_id = b"KTM\x01\x01\0\0\0".to_vec();
        over_id.extend_from_slice(&257u32.to_le_bytes());
        over_id.extend_from_slice(&1u32.to_le_bytes());
        assert_eq!(decode_ktm(&over_id, 100_000, |_, _| true), 0);

        let mut over_dsml = b"KTM\x01\x01\0\0\0".to_vec();
        over_dsml.extend_from_slice(&1u32.to_le_bytes());
        over_dsml.extend_from_slice(&(512u32 * 1024 * 1024 + 1).to_le_bytes());
        assert_eq!(decode_ktm(&over_dsml, 100_000, |_, _| true), 0);
    }

    #[test]
    fn ktm_decode_preserves_order_for_duplicate_last_wins() {
        let bytes = encode_ktm(&[(b"same", b"old"), (b"same", b"new")]).unwrap();
        let mut by_id = HashMap::new();
        assert_eq!(
            decode_ktm(&bytes, 100_000, |id, dsml| {
                by_id.insert(id.to_vec(), dsml.to_vec());
                true
            }),
            2
        );
        assert_eq!(
            by_id.get(b"same".as_slice()).map(Vec::as_slice),
            Some(b"new".as_slice())
        );
    }

    #[test]
    fn ktm_decode_matches_c_string_nul_truncation() {
        let bytes = b"KTM\x01\x01\0\0\0\x03\0\0\0\x03\0\0\0a\0bx\0y";
        let mut got = None;
        assert_eq!(
            decode_ktm(bytes, 100_000, |id, dsml| {
                got = Some((id.to_vec(), dsml.to_vec()));
                true
            }),
            1
        );
        assert_eq!(got, Some((b"a".to_vec(), b"x".to_vec())));

        let output = Command::new(oracle())
            .args(["ktm-decode", &hex(bytes), "100000"])
            .output()
            .unwrap();
        assert!(output.status.success());
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            "61:78\nloaded=1\n"
        );
    }
}
