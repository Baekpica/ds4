//! Live CUDA memgov census via the narrow bridge ABI.

use ds4_sys::{
    ds4_bridge_mem_census_snap, ds4_bridge_mem_observe_snap, ds4_bridge_mem_substrate_outstanding,
    DS4_BRIDGE_MEMC_COUNT, DS4_BRIDGE_MEMD_COUNT,
};

pub const MEMC_COUNT: usize = DS4_BRIDGE_MEMC_COUNT;
pub const MEMD_COUNT: usize = DS4_BRIDGE_MEMD_COUNT;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemCell {
    pub requested: u64,
    pub committed: u64,
    pub freed_requested: u64,
    pub freed_committed: u64,
    pub alloc_calls: u64,
    pub free_calls: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemCensus {
    pub supported: bool,
    pub faults: u64,
    pub epoch: u64,
    pub torn_fallbacks: u64,
    pub cells: [[MemCell; MEMD_COUNT]; MEMC_COUNT],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemObserve {
    pub status: i32,
    pub source: i32,
    pub free_bytes: u64,
    pub total_bytes: u64,
    pub cuda_free_bytes: u64,
    pub meminfo_avail_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemSnap {
    pub census: MemCensus,
    pub observe: MemObserve,
    pub substrate_outstanding: u64,
}

/// Process-global CUDA census + observation. Safe to call before
/// `Model::open`; unsupported backends return `supported = false`.
pub fn snapshot_mem() -> MemSnap {
    let mut raw = ds4_sys::ds4_bridge_mem_census {
        supported: 0,
        faults: 0,
        epoch: 0,
        torn_fallbacks: 0,
        cells: [[ds4_sys::ds4_bridge_mem_cell {
            requested: 0,
            committed: 0,
            freed_requested: 0,
            freed_committed: 0,
            alloc_calls: 0,
            free_calls: 0,
        }; MEMD_COUNT]; MEMC_COUNT],
    };
    let mut obs = ds4_sys::ds4_bridge_mem_observe {
        status: 1,
        source: 0,
        free_bytes: 0,
        total_bytes: 0,
        cuda_free_bytes: 0,
        meminfo_avail_bytes: 0,
    };
    let rc = unsafe { ds4_bridge_mem_census_snap(&mut raw) };
    if rc != 0 {
        return MemSnap::default();
    }
    let _ = unsafe { ds4_bridge_mem_observe_snap(&mut obs) };
    let substrate_outstanding = unsafe { ds4_bridge_mem_substrate_outstanding() };
    let mut census = MemCensus {
        supported: raw.supported != 0,
        faults: raw.faults,
        epoch: raw.epoch,
        torn_fallbacks: raw.torn_fallbacks,
        cells: [[MemCell::default(); MEMD_COUNT]; MEMC_COUNT],
    };
    for c in 0..MEMC_COUNT {
        for d in 0..MEMD_COUNT {
            let s = raw.cells[c][d];
            census.cells[c][d] = MemCell {
                requested: s.requested,
                committed: s.committed,
                freed_requested: s.freed_requested,
                freed_committed: s.freed_committed,
                alloc_calls: s.alloc_calls,
                free_calls: s.free_calls,
            };
        }
    }
    MemSnap {
        census,
        observe: MemObserve {
            status: obs.status,
            source: obs.source,
            free_bytes: obs.free_bytes,
            total_bytes: obs.total_bytes,
            cuda_free_bytes: obs.cuda_free_bytes,
            meminfo_avail_bytes: obs.meminfo_avail_bytes,
        },
        substrate_outstanding,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn census_abi_layout() {
        assert_eq!(size_of::<ds4_sys::ds4_bridge_mem_cell>(), 48);
        assert_eq!(align_of::<ds4_sys::ds4_bridge_mem_census>(), 8);
        assert_eq!(MEMC_COUNT, 17);
        assert_eq!(MEMD_COUNT, 2);
    }
}
