//! Host-owned memgov D0b evaluator: lease, claim, ledger, quote.
//!
//! Copied from `ds4_mem_gov.h` at v0.6.3-dfm. Pure: (ledger, observation,
//! claim) -> quote. No GPU, no clock, no allocation. Census numbers still
//! come from the native snap; this module owns the decision arithmetic.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::mem::{MEMC_COUNT, MEMD_COUNT};

/// C `DS4_GOVC__COUNT`. Growing this is a commit-visible act.
pub const GOVC_COUNT: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum GovConsumer {
    EngineBoot = 0,
    Prewarm = 1,
    BatchBankPlan = 2,
    SerialSession = 3,
    StaticBatch = 4,
}

impl GovConsumer {
    pub const fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::EngineBoot),
            1 => Some(Self::Prewarm),
            2 => Some(Self::BatchBankPlan),
            3 => Some(Self::SerialSession),
            4 => Some(Self::StaticBatch),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GovLease {
    pub intent: u64,
    pub resident: u64,
    pub reservation: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GovLedger {
    pub lease: [GovLease; GOVC_COUNT],
    pub floor_bytes: u64,
    pub substrate_outstanding: u64,
    pub epoch: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GovClaim {
    pub requester: i32,
    pub memc: i32,
    pub domain: i32,
    pub proposed_outstanding: u64,
    pub operation_transient: u64,
    pub class_limit: u64,
}

/// C `ds4_mem_obs_status`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum MemObsStatus {
    Ok = 0,
    Unsupported = 1,
    QueryError = 2,
}

/// C `ds4_mem_obs_source`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum MemObsSource {
    None = 0,
    CudaFree = 1,
    MeminfoAvailable = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemObservation {
    pub status: MemObsStatus,
    pub source: MemObsSource,
    pub free_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum GovStatus {
    Admit = 0,
    RefuseClass = 1,
    RefuseLive = 2,
    RetryObs = 3,
    Unsupported = 4,
    Fault = 5,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum GovCmp {
    Agree = 0,
    LiveStricter = 1,
    ShadowStricter = 2,
    VerdictClass = 3,
    ObsPolicy = 4,
    Fault = 5,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum GovMode {
    Off = 0,
    Observe = 1,
    Enforce = 2,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GovQuote {
    pub status: i32,
    pub retryable: i32,
    pub requester: i32,
    pub memc: i32,
    pub domain: i32,
    pub obs_status: i32,
    pub obs_source: i32,
    pub obs_free: u64,
    pub obs_total: u64,
    pub class_limit: u64,
    pub proposed_class_bytes: u64,
    pub floor_bytes: u64,
    pub other_reservations: u64,
    pub substrate_outstanding: u64,
    pub old_intent: u64,
    pub proposed_intent: u64,
    pub total_prospective_intent: u64,
    pub operation_transient: u64,
    pub available: u64,
    pub required: u64,
    pub deficit: u64,
    pub epoch: u64,
}

impl GovQuote {
    pub fn status(self) -> GovStatus {
        match self.status {
            0 => GovStatus::Admit,
            1 => GovStatus::RefuseClass,
            2 => GovStatus::RefuseLive,
            3 => GovStatus::RetryObs,
            4 => GovStatus::Unsupported,
            _ => GovStatus::Fault,
        }
    }
}

/// C `ds4_gov_compare`: closed reason set, no OTHER bucket.
pub fn gov_compare(live_status: GovStatus, shadow_status: GovStatus) -> GovCmp {
    if shadow_status == GovStatus::Fault {
        return GovCmp::Fault;
    }
    if shadow_status == GovStatus::RetryObs || shadow_status == GovStatus::Unsupported {
        return GovCmp::ObsPolicy;
    }
    if live_status == shadow_status {
        return GovCmp::Agree;
    }
    if live_status == GovStatus::Admit {
        return GovCmp::ShadowStricter;
    }
    if shadow_status == GovStatus::Admit {
        return GovCmp::LiveStricter;
    }
    GovCmp::VerdictClass
}

/// C `ds4_gov_mode_parse`: exact words only, no prefix or case slack.
pub fn gov_mode_parse(s: &str) -> Option<GovMode> {
    match s {
        "off" => Some(GovMode::Off),
        "observe" => Some(GovMode::Observe),
        "enforce" => Some(GovMode::Enforce),
        _ => None,
    }
}

pub fn gov_mode_name(mode: GovMode) -> &'static str {
    match mode {
        GovMode::Off => "off",
        GovMode::Observe => "observe",
        GovMode::Enforce => "enforce",
    }
}

fn sat_add(a: u64, b: u64, faults: &mut u64) -> u64 {
    match a.checked_add(b) {
        Some(sum) => sum,
        None => {
            *faults += 1;
            u64::MAX
        }
    }
}

/// C `ds4_gov_lease_publish`. `resident > intent` is refused so the
/// ledger never holds a state the evaluator must fail on.
pub fn gov_lease_publish(
    lg: &mut GovLedger,
    consumer: i32,
    intent: u64,
    resident: u64,
    reservation: u64,
    faults: &mut u64,
) -> bool {
    if !(0..GOVC_COUNT as i32).contains(&consumer) || resident > intent {
        *faults += 1;
        return false;
    }
    lg.lease[consumer as usize] = GovLease {
        intent,
        resident,
        reservation,
    };
    true
}

/// C `ds4_gov_evaluate`. Class cap first, live floor second. Saturation
/// or an impossible lease is FAULT, never a wrapped ADMIT.
pub fn gov_evaluate(lg: &GovLedger, obs: &MemObservation, cl: &GovClaim) -> GovQuote {
    let mut q = GovQuote::default();
    if GovConsumer::from_i32(cl.requester).is_none()
        || cl.memc < 0
        || cl.memc >= MEMC_COUNT as i32
        || cl.domain < 0
        || cl.domain >= MEMD_COUNT as i32
    {
        q.status = GovStatus::Fault as i32;
        return q;
    }
    q.requester = cl.requester;
    q.memc = cl.memc;
    q.domain = cl.domain;
    q.epoch = lg.epoch;
    q.obs_status = obs.status as i32;
    q.obs_source = obs.source as i32;
    q.class_limit = cl.class_limit;
    q.proposed_class_bytes = cl.proposed_outstanding;
    q.floor_bytes = lg.floor_bytes;
    q.substrate_outstanding = lg.substrate_outstanding;
    q.operation_transient = cl.operation_transient;
    q.old_intent = lg.lease[cl.requester as usize].intent;
    q.proposed_intent = cl.proposed_outstanding;

    let mut ovf = 0u64;
    let mut prospective_unfunded = 0u64;
    for c in 0..GOVC_COUNT {
        let l = lg.lease[c];
        if l.resident > l.intent {
            q.status = GovStatus::Fault as i32;
            return q;
        }
        if c != cl.requester as usize {
            q.other_reservations = sat_add(q.other_reservations, l.reservation, &mut ovf);
            prospective_unfunded = sat_add(prospective_unfunded, l.intent - l.resident, &mut ovf);
        }
        let term = if c == cl.requester as usize {
            cl.proposed_outstanding
        } else {
            l.intent
        };
        q.total_prospective_intent = sat_add(q.total_prospective_intent, term, &mut ovf);
    }
    if ovf != 0 {
        q.status = GovStatus::Fault as i32;
        return q;
    }
    if obs.status == MemObsStatus::Unsupported {
        q.status = GovStatus::Unsupported as i32;
        return q;
    }
    if obs.status != MemObsStatus::Ok {
        q.status = GovStatus::RetryObs as i32;
        q.retryable = 1;
        return q;
    }
    q.obs_free = obs.free_bytes;
    q.obs_total = obs.total_bytes;
    q.available = obs.free_bytes;
    if cl.class_limit != 0 && cl.proposed_outstanding > cl.class_limit {
        q.status = GovStatus::RefuseClass as i32;
        return q;
    }
    let resident = lg.lease[cl.requester as usize].resident;
    let unfunded = cl.proposed_outstanding.saturating_sub(resident);
    let mut req = sat_add(lg.floor_bytes, lg.substrate_outstanding, &mut ovf);
    req = sat_add(req, q.other_reservations, &mut ovf);
    req = sat_add(req, prospective_unfunded, &mut ovf);
    req = sat_add(req, unfunded, &mut ovf);
    req = sat_add(req, cl.operation_transient, &mut ovf);
    q.required = req;
    if ovf != 0 {
        q.status = GovStatus::Fault as i32;
        return q;
    }
    if q.available >= q.required {
        q.status = GovStatus::Admit as i32;
        return q;
    }
    q.status = GovStatus::RefuseLive as i32;
    q.retryable = 1;
    q.deficit = q.required - q.available;
    q
}

/// C `ds4_gov_epoch_write_begin`: odd epoch marks an in-flight write.
pub fn gov_epoch_write_begin(epoch: &AtomicU64, faults: &mut u64) {
    let s = epoch.load(Ordering::Relaxed);
    if s & 1 != 0 {
        *faults += 1;
    }
    epoch.store(s.wrapping_add(1), Ordering::Relaxed);
    std::sync::atomic::fence(Ordering::SeqCst);
}

/// C `ds4_gov_epoch_write_end`.
pub fn gov_epoch_write_end(epoch: &AtomicU64, faults: &mut u64) {
    std::sync::atomic::fence(Ordering::SeqCst);
    let s = epoch.load(Ordering::Relaxed);
    if s & 1 == 0 {
        *faults += 1;
    }
    epoch.store(s.wrapping_add(1), Ordering::Relaxed);
}

pub fn gov_epoch_read_begin(epoch: &AtomicU64) -> u64 {
    epoch.load(Ordering::Acquire)
}

/// 1 = copy taken since `read_begin` is coherent (same epoch, even).
pub fn gov_epoch_read_verify(epoch: &AtomicU64, began: u64) -> bool {
    std::sync::atomic::fence(Ordering::SeqCst);
    let now = epoch.load(Ordering::Relaxed);
    now == began && (began & 1) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    const MEMC_BATCH_BANK: i32 = 12;
    const MEMC_SESSION_TENSORS: i32 = 10;
    const MEMD_UNIFIED: i32 = 0;

    fn test_ledger() -> GovLedger {
        let mut lg = GovLedger::default();
        let mut faults = 0;
        assert!(gov_lease_publish(
            &mut lg,
            GovConsumer::EngineBoot as i32,
            10000,
            10000,
            0,
            &mut faults
        ));
        assert!(gov_lease_publish(
            &mut lg,
            GovConsumer::Prewarm as i32,
            200,
            200,
            0,
            &mut faults
        ));
        assert!(gov_lease_publish(
            &mut lg,
            GovConsumer::BatchBankPlan as i32,
            1200,
            1000,
            0,
            &mut faults
        ));
        assert!(gov_lease_publish(
            &mut lg,
            GovConsumer::SerialSession as i32,
            0,
            0,
            200,
            &mut faults
        ));
        assert_eq!(faults, 0);
        lg.floor_bytes = 600;
        lg.substrate_outstanding = 50;
        lg.epoch = 42;
        lg
    }

    fn test_obs(free_bytes: u64) -> MemObservation {
        MemObservation {
            status: MemObsStatus::Ok,
            source: MemObsSource::MeminfoAvailable,
            free_bytes,
            total_bytes: 20000,
        }
    }

    fn bank_claim(proposed: u64, class_limit: u64) -> GovClaim {
        GovClaim {
            requester: GovConsumer::BatchBankPlan as i32,
            memc: MEMC_BATCH_BANK,
            domain: MEMD_UNIFIED,
            proposed_outstanding: proposed,
            operation_transient: 0,
            class_limit,
        }
    }

    #[test]
    fn evaluate_verdicts_match_c() {
        let lg = test_ledger();
        let mut cl = bank_claim(1500, 2000);
        let q = gov_evaluate(&lg, &test_obs(1350), &cl);
        assert_eq!(q.status(), GovStatus::Admit);
        assert_eq!((q.required, q.available, q.deficit), (1350, 1350, 0));
        assert_eq!((q.old_intent, q.proposed_intent), (1200, 1500));
        assert_eq!(q.other_reservations, 200);
        assert_eq!(q.total_prospective_intent, 10000 + 200 + 1500);
        assert_eq!(q.epoch, 42);
        assert_eq!(q.obs_source, MemObsSource::MeminfoAvailable as i32);

        let q = gov_evaluate(&lg, &test_obs(1349), &cl);
        assert_eq!(q.status(), GovStatus::RefuseLive);
        assert_eq!(q.retryable, 1);
        assert_eq!(q.deficit, 1);

        cl.proposed_outstanding = 2500;
        let q = gov_evaluate(&lg, &test_obs(u64::MAX / 2), &cl);
        assert_eq!(q.status(), GovStatus::RefuseClass);
        assert_eq!(q.retryable, 0);
        assert_eq!((q.proposed_class_bytes, q.class_limit), (2500, 2000));

        cl.class_limit = 0;
        let q = gov_evaluate(&lg, &test_obs(u64::MAX / 2), &cl);
        assert_eq!(q.status(), GovStatus::Admit);

        cl.proposed_outstanding = 1500;
        cl.class_limit = 2000;
        cl.operation_transient = 100;
        let q = gov_evaluate(&lg, &test_obs(1449), &cl);
        assert_eq!(q.status(), GovStatus::RefuseLive);
        assert_eq!(q.deficit, 1);
    }

    #[test]
    fn serial_window_lease_matches_c() {
        let mut lg = test_ledger();
        let mut faults = 0;
        let cl = bank_claim(1500, 0);
        let o = test_obs(1350);
        let q = gov_evaluate(&lg, &o, &cl);
        assert_eq!(q.status(), GovStatus::Admit);
        assert_eq!(q.required, 1350);

        assert!(gov_lease_publish(
            &mut lg,
            GovConsumer::SerialSession as i32,
            5000,
            0,
            200,
            &mut faults
        ));
        let q = gov_evaluate(&lg, &o, &cl);
        assert_eq!(q.status(), GovStatus::RefuseLive);
        assert_eq!((q.required, q.deficit), (1350 + 5000, 5000));

        assert!(gov_lease_publish(
            &mut lg,
            GovConsumer::SerialSession as i32,
            5000,
            5000,
            200,
            &mut faults
        ));
        let q = gov_evaluate(&lg, &o, &cl);
        assert_eq!(q.status(), GovStatus::Admit);
        assert_eq!(q.required, 1350);

        assert!(gov_lease_publish(
            &mut lg,
            GovConsumer::SerialSession as i32,
            0,
            0,
            200,
            &mut faults
        ));
        let q = gov_evaluate(&lg, &o, &cl);
        assert_eq!(q.status(), GovStatus::Admit);
        assert_eq!(q.required, 1350);
        assert_eq!(faults, 0);
    }

    #[test]
    fn requester_asymmetry_matches_c() {
        let lg = test_ledger();
        let mut cl = GovClaim {
            requester: GovConsumer::SerialSession as i32,
            memc: MEMC_SESSION_TENSORS,
            domain: MEMD_UNIFIED,
            proposed_outstanding: 700,
            operation_transient: 0,
            class_limit: 0,
        };
        let qs = gov_evaluate(&lg, &test_obs(1550), &cl);
        assert_eq!(qs.status(), GovStatus::Admit);
        assert_eq!(qs.required, 1550);
        assert_eq!(qs.other_reservations, 0);

        cl.requester = GovConsumer::StaticBatch as i32;
        let qb = gov_evaluate(&lg, &test_obs(1550), &cl);
        assert_eq!(qb.status(), GovStatus::RefuseLive);
        assert_eq!((qb.required, qb.deficit), (1550 + 200, 200));
        assert_eq!(qb.other_reservations, 200);
    }

    #[test]
    fn absolute_replacement_is_idempotent() {
        let lg = test_ledger();
        let mut cl = bank_claim(1500, 2000);
        let o = test_obs(1350);
        assert_eq!(gov_evaluate(&lg, &o, &cl), gov_evaluate(&lg, &o, &cl));

        cl.proposed_outstanding = 900;
        let qs = gov_evaluate(&lg, &test_obs(850), &cl);
        assert_eq!(qs.status(), GovStatus::Admit);
        assert_eq!(qs.required, 850);

        let mut after = test_ledger();
        let before = after;
        let mut faults = 0;
        assert!(gov_lease_publish(
            &mut after,
            GovConsumer::BatchBankPlan as i32,
            1200,
            1000,
            0,
            &mut faults
        ));
        assert_eq!(before, after);
        assert_eq!(faults, 0);
    }

    #[test]
    fn observation_policy_matches_c() {
        let lg = test_ledger();
        let cl = bank_claim(1500, 0);
        let mut o = test_obs(1350);
        o.status = MemObsStatus::Unsupported;
        let q = gov_evaluate(&lg, &o, &cl);
        assert_eq!(q.status(), GovStatus::Unsupported);
        assert_eq!(q.retryable, 0);
        assert_eq!((q.obs_free, q.available), (0, 0));
        assert_eq!(q.requester, GovConsumer::BatchBankPlan as i32);
        assert_eq!(q.epoch, 42);

        o.status = MemObsStatus::QueryError;
        let q = gov_evaluate(&lg, &o, &cl);
        assert_eq!(q.status(), GovStatus::RetryObs);
        assert_eq!(q.retryable, 1);
    }

    #[test]
    fn checked_arithmetic_fails_closed() {
        let cl = bank_claim(1500, 0);
        let o = test_obs(1350);
        let mut lg = test_ledger();
        assert_eq!(
            gov_evaluate(&lg, &o, &GovClaim { requester: 5, ..cl }).status(),
            GovStatus::Fault
        );
        assert_eq!(
            gov_evaluate(
                &lg,
                &o,
                &GovClaim {
                    memc: MEMC_COUNT as i32,
                    ..cl
                }
            )
            .status(),
            GovStatus::Fault
        );
        assert_eq!(
            gov_evaluate(&lg, &o, &GovClaim { domain: -1, ..cl }).status(),
            GovStatus::Fault
        );

        lg.lease[GovConsumer::EngineBoot as usize].resident =
            lg.lease[GovConsumer::EngineBoot as usize].intent + 1;
        assert_eq!(gov_evaluate(&lg, &o, &cl).status(), GovStatus::Fault);

        lg = test_ledger();
        lg.floor_bytes = u64::MAX;
        assert_eq!(
            gov_evaluate(&lg, &test_obs(u64::MAX), &cl).status(),
            GovStatus::Fault
        );
        lg = test_ledger();
        lg.lease[GovConsumer::EngineBoot as usize].intent = u64::MAX;
        lg.lease[GovConsumer::EngineBoot as usize].resident = u64::MAX;
        assert_eq!(
            gov_evaluate(&lg, &test_obs(u64::MAX), &cl).status(),
            GovStatus::Fault
        );

        let before = test_ledger();
        let mut after = before;
        let mut faults = 0;
        assert!(!gov_lease_publish(
            &mut after,
            GOVC_COUNT as i32,
            1,
            1,
            0,
            &mut faults
        ));
        assert!(!gov_lease_publish(
            &mut after,
            GovConsumer::Prewarm as i32,
            100,
            101,
            0,
            &mut faults
        ));
        assert_eq!(faults, 2);
        assert_eq!(before, after);
    }

    #[test]
    fn state_machine_matches_c() {
        let mut lg = test_ledger();
        let mut faults = 0;
        let cl = bank_claim(1500, 2000);
        let o = test_obs(1350);
        assert_eq!(gov_evaluate(&lg, &o, &cl).status(), GovStatus::Admit);
        assert!(gov_lease_publish(
            &mut lg,
            GovConsumer::BatchBankPlan as i32,
            1500,
            1500,
            0,
            &mut faults
        ));
        let q = gov_evaluate(&lg, &o, &cl);
        assert_eq!(q.status(), GovStatus::Admit);
        assert_eq!(q.required, 850);
        assert!(gov_lease_publish(
            &mut lg,
            GovConsumer::BatchBankPlan as i32,
            0,
            0,
            0,
            &mut faults
        ));
        let q = gov_evaluate(&lg, &o, &cl);
        assert_eq!((q.old_intent, q.required), (0, 850 + 1500));
        assert_eq!(q.status(), GovStatus::RefuseLive);
        assert_eq!(q.deficit, 1000);
        assert_eq!(faults, 0);
    }

    #[test]
    fn epoch_protocol_matches_c() {
        let epoch = AtomicU64::new(0);
        let mut faults = 0;
        let mut began = gov_epoch_read_begin(&epoch);
        assert!(gov_epoch_read_verify(&epoch, began));

        gov_epoch_write_begin(&epoch, &mut faults);
        began = gov_epoch_read_begin(&epoch);
        assert_ne!(began & 1, 0);
        assert!(!gov_epoch_read_verify(&epoch, began));
        gov_epoch_write_end(&epoch, &mut faults);
        assert_eq!(faults, 0);

        began = gov_epoch_read_begin(&epoch);
        gov_epoch_write_begin(&epoch, &mut faults);
        gov_epoch_write_end(&epoch, &mut faults);
        assert!(!gov_epoch_read_verify(&epoch, began));

        began = gov_epoch_read_begin(&epoch);
        assert!(gov_epoch_read_verify(&epoch, began));
        assert_eq!(faults, 0);

        gov_epoch_write_begin(&epoch, &mut faults);
        gov_epoch_write_begin(&epoch, &mut faults);
        assert_eq!(faults, 1);
        gov_epoch_write_end(&epoch, &mut faults);
        assert_eq!(faults, 2);
        gov_epoch_write_end(&epoch, &mut faults);
        assert_eq!(faults, 2);
        began = gov_epoch_read_begin(&epoch);
        assert_eq!(began & 1, 0);
        assert!(gov_epoch_read_verify(&epoch, began));
    }

    #[test]
    fn compare_classes_are_closed() {
        assert_eq!(
            gov_compare(GovStatus::Admit, GovStatus::Admit),
            GovCmp::Agree
        );
        assert_eq!(
            gov_compare(GovStatus::RefuseLive, GovStatus::RefuseLive),
            GovCmp::Agree
        );
        assert_eq!(
            gov_compare(GovStatus::RefuseLive, GovStatus::Admit),
            GovCmp::LiveStricter
        );
        assert_eq!(
            gov_compare(GovStatus::Admit, GovStatus::RefuseLive),
            GovCmp::ShadowStricter
        );
        assert_eq!(
            gov_compare(GovStatus::RefuseClass, GovStatus::RefuseLive),
            GovCmp::VerdictClass
        );
        assert_eq!(
            gov_compare(GovStatus::Admit, GovStatus::RetryObs),
            GovCmp::ObsPolicy
        );
        assert_eq!(
            gov_compare(GovStatus::RefuseLive, GovStatus::Unsupported),
            GovCmp::ObsPolicy
        );
        assert_eq!(
            gov_compare(GovStatus::Admit, GovStatus::Fault),
            GovCmp::Fault
        );
    }

    #[test]
    fn mode_parse_is_exact() {
        assert_eq!(gov_mode_parse("off"), Some(GovMode::Off));
        assert_eq!(gov_mode_parse("observe"), Some(GovMode::Observe));
        assert_eq!(gov_mode_parse("enforce"), Some(GovMode::Enforce));
        assert_eq!(gov_mode_parse(""), None);
        assert_eq!(gov_mode_parse("obs"), None);
        assert_eq!(gov_mode_parse("observed"), None);
        assert_eq!(gov_mode_parse("Enforce"), None);
        assert_eq!(gov_mode_parse("offx"), None);
        for mode in [GovMode::Off, GovMode::Observe, GovMode::Enforce] {
            assert_eq!(gov_mode_parse(gov_mode_name(mode)), Some(mode));
        }
    }
}
