use ds4_core::Model;

fn routed_quant_bits(model: &Model) {
    let _ = model.routed_quant_bits();
}

#[test]
fn model_exposes_routed_quant_bits() {
    let _ = routed_quant_bits as fn(&Model);
}
