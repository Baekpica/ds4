//! Static-lane owner around `BatchCtx::generate_static` (C `n>=2`).

use crate::generate::GenerateError;

/// C `ds4_bridge_batch_ctx_generate_static` when `n` is not a legal width.
pub const STATIC_WIDTH_ERR: &str = "static batch request count is out of range";

/// C `generate_batch_jobs` when the batched `err` buffer is empty.
pub const STATIC_FALLBACK_ERR: &str = "out of memory";

/// C `generate_batch_jobs` admission: coalesced group only.
pub const STATIC_N_MIN: usize = 2;

/// One greedy static row. Tokens are borrowed only for the call.
#[derive(Clone, Copy)]
pub struct StaticJob<'a> {
    pub tokens: &'a [i32],
    pub max_new_tokens: i32,
    pub eos: i32,
}

/// Owned sibling waiting to coalesce with the next static-routed request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnedStaticJob {
    pub tokens: Vec<i32>,
    pub max_new_tokens: i32,
    pub eos: i32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StaticRow {
    pub tokens: Vec<i32>,
}

/// Trait seam so tests can spy on `generate_static` without a GGUF.
pub trait StaticExec {
    fn generate_static(&mut self, jobs: &[StaticJob<'_>]) -> Result<Vec<StaticRow>, GenerateError>;

    /// Extra rows already waiting (coalesce). Default: none.
    fn pending_siblings(&self) -> &[OwnedStaticJob] {
        &[]
    }

    /// C per-call fallback after `generate_static` fails. Default: C err text.
    fn fallback_static(&mut self, err: GenerateError) -> Result<Vec<StaticRow>, GenerateError> {
        Err(static_fallback_error(err))
    }
}

/// Map a failed `generate_static` onto C `generate_batch_jobs` err text.
pub fn static_fallback_error(err: GenerateError) -> GenerateError {
    match err {
        GenerateError::Engine(msg) if !msg.is_empty() => GenerateError::Engine(msg),
        GenerateError::Engine(_)
        | GenerateError::Unsupported(_)
        | GenerateError::Io
        | GenerateError::ContinuationHold { .. } => {
            GenerateError::Engine(STATIC_FALLBACK_ERR.to_string())
        }
    }
}

/// Production owner: `BatchCtx::generate_static` with no GGUF in this crate.
#[cfg(feature = "native")]
pub struct BatchStatic<'a, 'm> {
    ctx: &'a mut ds4_core::BatchCtx<'m>,
}

#[cfg(feature = "native")]
impl<'a, 'm> BatchStatic<'a, 'm> {
    pub fn new(ctx: &'a mut ds4_core::BatchCtx<'m>) -> Self {
        Self { ctx }
    }
}

#[cfg(feature = "native")]
impl StaticExec for BatchStatic<'_, '_> {
    fn generate_static(&mut self, jobs: &[StaticJob<'_>]) -> Result<Vec<StaticRow>, GenerateError> {
        let requests: Vec<ds4_core::StaticBatchRequest<'_>> = jobs
            .iter()
            .map(|job| ds4_core::StaticBatchRequest {
                tokens: job.tokens,
                max_new_tokens: job.max_new_tokens,
                eos: job.eos,
            })
            .collect();
        self.ctx
            .generate_static(&requests)
            .map(|rows| {
                rows.into_iter()
                    .map(|row| StaticRow { tokens: row.tokens })
                    .collect()
            })
            .map_err(|err| GenerateError::Engine(err.message))
    }
}

/// Used when the route is static but no owner is attached (n<2 still refuses).
pub struct DetachedStatic;

impl StaticExec for DetachedStatic {
    fn generate_static(
        &mut self,
        _jobs: &[StaticJob<'_>],
    ) -> Result<Vec<StaticRow>, GenerateError> {
        Err(GenerateError::Unsupported("static owner is not attached"))
    }
}

/// `None` means `n` is admitted at the owner boundary.
pub const fn static_width_error(n: usize) -> Option<&'static str> {
    if n < STATIC_N_MIN {
        Some(STATIC_WIDTH_ERR)
    } else {
        None
    }
}

/// Owner entry: refuse `n<2` with the C width string; otherwise call
/// [`StaticExec::generate_static`]. On err, C fallback text — never serial.
pub fn run_static(
    exec: &mut dyn StaticExec,
    jobs: &[StaticJob<'_>],
) -> Result<Vec<StaticRow>, GenerateError> {
    if let Some(msg) = static_width_error(jobs.len()) {
        return Err(GenerateError::Engine(msg.to_string()));
    }
    match exec.generate_static(jobs) {
        Ok(rows) => Ok(rows),
        Err(err) => exec.fallback_static(err),
    }
}

/// Routed owner: current request plus any [`StaticExec::pending_siblings`].
pub fn run_static_routed(
    exec: &mut dyn StaticExec,
    current: StaticJob<'_>,
) -> Result<Vec<StaticRow>, GenerateError> {
    let siblings = exec.pending_siblings().to_vec();
    let mut jobs = Vec::with_capacity(siblings.len() + 1);
    for sibling in &siblings {
        jobs.push(StaticJob {
            tokens: &sibling.tokens,
            max_new_tokens: sibling.max_new_tokens,
            eos: sibling.eos,
        });
    }
    jobs.push(current);
    run_static(exec, &jobs)
}

#[cfg(test)]
#[path = "serve_static_harness.rs"]
mod harness;

#[cfg(test)]
#[path = "serve_static_test.rs"]
mod tests;

#[cfg(test)]
#[path = "serve_static_fallback_test.rs"]
mod fallback_tests;
