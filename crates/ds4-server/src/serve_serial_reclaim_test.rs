use crate::serve_cont_roll::RejectReason;
use crate::serve_serial_reclaim::{
    serial_capacity_refuse_msg, serial_reclaim_gate, serial_reclaim_rank, AvailBytes,
    HeadroomBytes, MemFloor, NeedBytes, ReclaimBank, ReclaimableBytes, SerialReclaimAsk,
    SerialReclaimOutcome,
};

const GIB: u64 = 1 << 30;

fn ask(avail_gib: u64, need_gib: u64, floor_gb: u64, reclaim_gib: u64) -> SerialReclaimAsk {
    SerialReclaimAsk {
        avail: AvailBytes::from_raw(avail_gib * GIB),
        need: NeedBytes::from_raw(need_gib * GIB),
        floor: MemFloor::from_gb(floor_gb),
        reclaimable: ReclaimableBytes::from_raw(reclaim_gib * GIB),
        headroom: HeadroomBytes::from_raw(0),
    }
}

fn remaining_after(ask: SerialReclaimAsk, out: SerialReclaimOutcome) -> u64 {
    let after = ask.avail.raw().saturating_add(out.reclaimed());
    if out.admitted() {
        after.saturating_sub(ask.need.raw())
    } else {
        after
    }
}

#[test]
fn mem_floor_default_is_c_four_gib() {
    // Given: no DS4_MEM_FLOOR_GB
    // When: parse the C floor helper
    let floor = MemFloor::from_env_gb(None);

    // Then: 4 GiB, same as ds4_mem_floor_bytes
    assert_eq!(floor, MemFloor::from_gb(4));
    assert_eq!(floor.bytes(), 4 * GIB);
}

#[test]
fn mem_floor_env_zero_is_c_kill_switch() {
    // Given: DS4_MEM_FLOOR_GB=0
    // When: parse
    let floor = MemFloor::from_env_gb(Some(b"0"));

    // Then: floor is 0 (C A/B kill switch)
    assert_eq!(floor.bytes(), 0);
}

#[test]
fn reclaim_then_admit_when_idle_banks_cover_deficit() {
    // Given: 5 GiB free, 4 GiB floor, 2 GiB serial need, 3 GiB idle
    let ask = ask(5, 2, 4, 3);

    // When: the serial reclaim gate runs
    let out = serial_reclaim_gate(ask);

    // Then: reclaim the 1 GiB deficit, then admit; floor intact
    assert_eq!(out, SerialReclaimOutcome::Admit { reclaimed: GIB });
    assert!(remaining_after(ask, out) >= ask.floor.bytes());
}

#[test]
fn reclaim_runs_before_typed_refuse_when_still_short() {
    // Given: reclaim cannot cover need above the floor
    let ask = ask(5, 3, 4, 1);

    // When: the serial reclaim gate runs
    let out = serial_reclaim_gate(ask);

    // Then: reclaim still ran; typed refuse is C live_headroom
    assert_eq!(
        out,
        SerialReclaimOutcome::Refuse {
            reclaimed: GIB,
            reason: RejectReason::LiveHeadroom,
        }
    );
    assert_eq!(RejectReason::LiveHeadroom.name(), "live_headroom");
    assert!(remaining_after(ask, out) >= ask.floor.bytes());
}

#[test]
fn refuse_does_not_admit_below_mem_floor() {
    // Given: already at the floor, reclaim cannot fund the graph
    let ask = ask(4, 2, 4, 1);

    // When: the serial reclaim gate runs
    let out = serial_reclaim_gate(ask);

    // Then: still refuse; never admit a spend that crosses the floor
    assert!(!out.admitted());
    match out {
        SerialReclaimOutcome::Refuse { reason, reclaimed } => {
            assert_eq!(reason, RejectReason::LiveHeadroom);
            assert_eq!(reclaimed, GIB);
        }
        SerialReclaimOutcome::Admit { .. } => panic!("admitted below --mem-floor-gb"),
    }
    assert!(remaining_after(ask, out) >= ask.floor.bytes());
}

#[test]
fn already_fits_does_not_reclaim() {
    // Given: free already covers need + floor
    let ask = ask(7, 2, 4, 8);

    // When: the serial reclaim gate runs
    let out = serial_reclaim_gate(ask);

    // Then: admit with no reclaim
    assert_eq!(out, SerialReclaimOutcome::Admit { reclaimed: 0 });
}

#[test]
fn refuse_body_matches_c_serial_capacity() {
    // Given: C serial_session_ensure_fit 503
    // When: format the refuse
    let msg = serial_capacity_refuse_msg(26);

    // Then: exact C body
    assert_eq!(
        msg,
        "Server is temporarily at capacity for a 26-token serial request \
         (no session graph fits beside the batch banks); retry shortly"
    );
}

#[test]
fn rank_skips_protected_and_deep_pin() {
    // Given: C serial_reclaim_rank hard exclusions
    let banks = [
        ReclaimBank {
            id: 0,
            valid: false,
            last_use: 10,
            committed: 0,
            protected: true,
            superseded: false,
        },
        ReclaimBank {
            id: 1,
            valid: false,
            last_use: 1,
            committed: 0,
            protected: false,
            superseded: false,
        },
        ReclaimBank {
            id: 2,
            valid: true,
            last_use: 2,
            committed: 40,
            protected: false,
            superseded: true,
        },
        ReclaimBank {
            id: 3,
            valid: true,
            last_use: 3,
            committed: 50,
            protected: false,
            superseded: false,
        },
        ReclaimBank {
            id: 4,
            valid: true,
            last_use: 4,
            committed: 192,
            protected: false,
            superseded: false,
        },
    ];

    // When: rank with warm_pin_min=192
    let ranked = serial_reclaim_rank(&banks, 192);

    // Then: protected 0 and deep 4 stay intact; cheapest first
    assert_eq!(ranked, vec![1, 2, 3]);
}

#[test]
fn rank_orders_superseded_by_lru() {
    // Given: two shallow superseded banks
    let banks = [
        ReclaimBank {
            id: 8,
            valid: true,
            last_use: 30,
            committed: 10,
            protected: false,
            superseded: true,
        },
        ReclaimBank {
            id: 7,
            valid: true,
            last_use: 10,
            committed: 10,
            protected: false,
            superseded: true,
        },
    ];

    // When: rank
    let ranked = serial_reclaim_rank(&banks, 192);

    // Then: older last_use first (C insertion sort)
    assert_eq!(ranked, vec![7, 8]);
}
