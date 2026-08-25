//! Rolling continuous admit table. The engine's `ContDriver::admit` pulls
//! the next pending job while others are still live; one-at-a-time is not
//! the only path.

use std::collections::{HashSet, VecDeque};

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
}
