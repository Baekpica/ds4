//! Worker HELLO record from live slice metadata.

use crate::session_exec::SliceMeta;
use ds4_dist::{Hello, MAX_MODEL_NAME};

pub const UNKNOWN_MODEL_NAME: &str = "unknown";

pub fn worker_model_name(name: &str) -> &str {
    if name.is_empty() {
        UNKNOWN_MODEL_NAME
    } else {
        name
    }
}

pub fn worker_hello(
    meta: &SliceMeta,
    quant_bits: u32,
    listen_port: u32,
    model_name: &str,
) -> Hello {
    let name = worker_model_name(model_name);
    let name_len = name.len().min(MAX_MODEL_NAME as usize) as u32;
    Hello {
        model_id: meta.model_id,
        quant_bits,
        layer_start: meta.layer_start,
        layer_end: meta.layer_end,
        has_output: u32::from(meta.has_output),
        has_hidden: 1,
        ctx_size: meta.ctx_size,
        n_layers: meta.n_layers,
        listen_port,
        model_name_len: name_len,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session_exec::slice_meta;
    use ds4_core::SHAPE_FLASH;
    use ds4_dist::{decode_hello_payload, encode_hello_payload, Layers};

    #[test]
    fn worker_hello_matches_c_field_assignment() {
        let layers = Layers {
            start: 20,
            end: 20,
            has_output: true,
            set: true,
        };
        let meta = slice_meta(7, &SHAPE_FLASH, 4096, &layers);
        let hello = worker_hello(&meta, 2, 7100, SHAPE_FLASH.name);
        assert_eq!(hello.model_id, 7);
        assert_eq!(hello.quant_bits, 2);
        assert_eq!(hello.layer_start, 20);
        assert_eq!(hello.layer_end, 42);
        assert_eq!(hello.has_output, 1);
        assert_eq!(hello.has_hidden, 1);
        assert_eq!(hello.ctx_size, 4096);
        assert_eq!(hello.n_layers, 43);
        assert_eq!(hello.listen_port, 7100);
        assert_eq!(hello.model_name_len, SHAPE_FLASH.name.len() as u32);

        let empty = worker_hello(&meta, 2, 7100, "");
        assert_eq!(empty.model_name_len, UNKNOWN_MODEL_NAME.len() as u32);
        let payload = encode_hello_payload(&hello, SHAPE_FLASH.name).unwrap();
        let (decoded, name) = decode_hello_payload(&payload).unwrap();
        assert_eq!(decoded, hello);
        assert_eq!(name, SHAPE_FLASH.name);
    }
}
