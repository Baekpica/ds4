//! Rolling continuous admit table. The engine's `ContDriver::admit` pulls
//! the next pending job while others are still live; one-at-a-time is not
//! the only path.

use std::collections::{HashSet, VecDeque};

/// Banks already claimed by a prior rolling prepare. The next `prepare_slot`
/// ORs these onto the continuation-hold mask so fork/pin/evict cannot spend
/// a live target or drop protected saturation.
#[derive(Debug, Default)]
pub(crate) struct RollReserve {
    banks: Vec<usize>,
}

impl RollReserve {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Record the 1-based `ContAdmit::place_bank` (0 = unset).
    pub(crate) fn note_place(&mut self, place_bank: i32) {
        let Some(bank) = (place_bank > 0)
            .then(|| usize::try_from(place_bank - 1).ok())
            .flatten()
        else {
            return;
        };
        if !self.banks.contains(&bank) {
            self.banks.push(bank);
        }
    }

    pub(crate) fn protect(&self, hold: &[bool]) -> Vec<bool> {
        let mut protected = hold.to_vec();
        for &bank in &self.banks {
            if bank >= protected.len() {
                protected.resize(bank + 1, false);
            }
            protected[bank] = true;
        }
        protected
    }
}

/// Host-side rolling job ids. `admit` may return another job while one is
/// already generating; `complete` retires a live job.
#[derive(Debug, Default)]
pub(crate) struct ContRoll {
    pending: VecDeque<usize>,
    live: HashSet<usize>,
    done: Vec<usize>,
}

impl ContRoll {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn enqueue(&mut self, user: usize) {
        self.pending.push_back(user);
    }

    /// Pull the next pending job into the live set. Returns `Some` even when
    /// another job is already generating.
    pub(crate) fn admit(&mut self) -> Option<usize> {
        let user = self.pending.pop_front()?;
        self.live.insert(user);
        Some(user)
    }

    pub(crate) fn live_count(&self) -> usize {
        self.live.len()
    }

    pub(crate) fn complete(&mut self, user: usize) {
        if self.live.remove(&user) {
            self.done.push(user);
        }
    }

    pub(crate) fn completed(&self) -> &[usize] {
        &self.done
    }
}

#[cfg(test)]
mod tests {
    use super::ContRoll;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct FakeJob {
        id: usize,
    }

    #[test]
    fn admits_two_fake_jobs_and_both_complete() {
        let mut roll = ContRoll::new();
        let first_job = FakeJob { id: 1 };
        let second_job = FakeJob { id: 2 };
        roll.enqueue(first_job.id);
        roll.enqueue(second_job.id);

        let first = roll.admit().expect("first job admits");
        assert_eq!(first, first_job.id);
        let second = roll
            .admit()
            .expect("second job admits while first is generating");
        assert_eq!(second, second_job.id);
        assert_eq!(roll.live_count(), 2, "both jobs generate together");
        assert!(roll.admit().is_none());

        roll.complete(first);
        roll.complete(second);
        assert_eq!(roll.completed(), &[first_job.id, second_job.id]);
    }

    #[test]
    fn rolling_reserve_protects_the_live_place_bank() {
        // Given: first rolling admit placed on bank 2 (1-based place_bank = 3)
        let mut reserve = super::RollReserve::new();
        reserve.note_place(3);

        // When: the second admit merges that reserve onto an empty hold mask
        let protected = reserve.protect(&[false, false, false]);

        // Then: only the live target is protected
        assert_eq!(protected, vec![false, false, true]);
    }

    #[test]
    fn rolling_reserve_keeps_hold_saturation() {
        // Given: continuation hold already pins bank 0
        let mut reserve = super::RollReserve::new();
        reserve.note_place(2);

        // When: the second admit ORs reserved banks onto that hold
        let protected = reserve.protect(&[true, false, false]);

        // Then: hold is kept and the live target is also protected
        assert_eq!(protected, vec![true, true, false]);
    }

    #[test]
    fn rolling_reserve_ignores_unset_place_bank() {
        let mut reserve = super::RollReserve::new();
        reserve.note_place(0);
        assert_eq!(reserve.protect(&[false, false]), vec![false, false]);
    }
}
