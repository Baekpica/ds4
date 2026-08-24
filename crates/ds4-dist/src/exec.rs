//! Slice evaluation boundary. Production later calls `ds4_session_eval_layer_slice`
//! through the FFI; tests and the host runtime use this trait.

#[derive(Debug, Clone)]
pub struct WorkRequest {
    pub session_id: u64,
    pub request_id: u64,
    pub tokens: Vec<i32>,
    pub pos0: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    pub reset: bool,
    pub produce_hidden: bool,
    pub produce_logits: bool,
    pub input_hc: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct WorkOutput {
    pub hidden: Option<Vec<f32>>,
    pub logits: Option<Vec<f32>>,
}

pub trait SliceExec {
    fn model_id(&self) -> u32;
    fn n_layers(&self) -> u32;
    fn vocab(&self) -> u32;
    fn ctx_size(&self) -> u32;
    fn hidden_values(&self) -> u64;
    fn has_output(&self) -> bool;
    fn layer_start(&self) -> u32;
    fn layer_end(&self) -> u32;
    fn eval(&mut self, req: &WorkRequest) -> Result<WorkOutput, String>;
}
