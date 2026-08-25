//! Blocking frame I/O. Same layout as `dist_write_frame_header` / `dist_read_frame_header`.

use std::io::{self, Read, Write};

use crate::codec::{decode_frame_header, encode_frame_header, CodecError, FRAME_HEADER_BYTES};

pub fn write_frame<W: Write>(w: &mut W, typ: u32, payload: &[u8]) -> io::Result<()> {
    let bytes = u32::try_from(payload.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "distributed frame too large"))?;
    w.write_all(&encode_frame_header(typ, bytes))?;
    w.write_all(payload)?;
    Ok(())
}

pub fn read_frame<R: Read>(r: &mut R) -> io::Result<(u32, Vec<u8>)> {
    let mut hdr = [0u8; FRAME_HEADER_BYTES];
    r.read_exact(&mut hdr)?;
    let h = decode_frame_header(&hdr).map_err(|e| match e {
        CodecError::BadMagic(m) => io::Error::new(
            io::ErrorKind::InvalidData,
            format!("bad frame magic 0x{m:08x}"),
        ),
        other => io::Error::new(io::ErrorKind::InvalidData, other.to_string()),
    })?;
    let mut payload = vec![0u8; h.bytes as usize];
    if h.bytes != 0 {
        r.read_exact(&mut payload)?;
    }
    Ok((h.typ, payload))
}
