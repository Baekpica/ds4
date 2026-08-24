//! Inc 5a/5b/5c continuation registry from `ds4_server.c` at v0.6.3-dfm.
//! Host-owned: publish / resolve / hold / pin / TTL / bank claim.
//! Native session still executes prefill; this decides reuse vs 409/503.

use crate::route::Api;

pub const CONT_REGISTRY_MAX_DEFAULT: i32 = 64;
pub const CONT_GRACE_S: f64 = 60.0;
pub const CONT_TTL_S: f64 = 300.0;
pub const CONT_PIN_DEADLINE_S: f64 = 60.0;
pub const CONT_HOLD_SHED_S: f64 = 5.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContState {
    LiveFrontier = 0,
    ReplayOnly = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContOwner {
    SerialSession = 0,
    BatchBank = 1,
}

#[derive(Clone, Debug)]
pub struct ContRecord {
    pub state: ContState,
    pub owner: ContOwner,
    pub protocol: u8,
    pub owner_id: i32,
    pub owner_gen: u64,
    pub frontier: i32,
    pub call_ids: Vec<String>,
    pub publish_time: f64,
    pub hard_refs: i32,
    pub pin_expiry: f64,
}

#[derive(Clone, Debug)]
pub struct ContRegistry {
    records: Vec<ContRecord>,
    pub max_records: i32,
    pub grace_s: f64,
    pub ttl_s: f64,
    pub pin_deadline_s: f64,
    pub hold_shed_s: f64,
    pub serial_live: Option<usize>,
}

impl Default for ContRegistry {
    fn default() -> Self {
        Self {
            records: Vec::new(),
            max_records: CONT_REGISTRY_MAX_DEFAULT,
            grace_s: CONT_GRACE_S,
            ttl_s: CONT_TTL_S,
            pin_deadline_s: CONT_PIN_DEADLINE_S,
            hold_shed_s: CONT_HOLD_SHED_S,
            serial_live: None,
        }
    }
}

impl ContRegistry {
    pub fn n_records(&self) -> i32 {
        self.records.len() as i32
    }

    pub fn n_live(&self) -> i32 {
        self.records
            .iter()
            .filter(|r| r.state == ContState::LiveFrontier)
            .count() as i32
    }

    pub fn live_ids(&self, proto: Api) -> Vec<String> {
        let mut out = Vec::new();
        for rec in &self.records {
            if rec.state != ContState::LiveFrontier || rec.protocol != proto as u8 {
                continue;
            }
            for id in &rec.call_ids {
                if !id.is_empty() && !out.iter().any(|x| x == id) {
                    out.push(id.clone());
                }
            }
        }
        out
    }

    fn key(proto: u8, id: &str) -> String {
        let mut id = id;
        if id.len() > 94 {
            id = &id[..94];
        }
        let mut s = String::with_capacity(2 + id.len());
        s.push(char::from(b'0' + proto));
        s.push('\u{001f}');
        s.push_str(id);
        s
    }

    fn find_idx(&self, proto: u8, id: &str) -> Option<usize> {
        if id.is_empty() {
            return None;
        }
        let k = Self::key(proto, id);
        self.records.iter().position(|r| {
            r.call_ids
                .iter()
                .any(|cid| Self::key(r.protocol, cid) == k)
        })
    }

    fn set_eq(a: &[String], b: &[String]) -> bool {
        a.len() == b.len() && a.iter().all(|id| b.iter().any(|x| x == id))
    }

    fn demote_idx(&mut self, idx: usize) {
        if self.records[idx].state != ContState::LiveFrontier {
            return;
        }
        self.records[idx].state = ContState::ReplayOnly;
        if self.serial_live == Some(idx) {
            self.serial_live = None;
        }
    }

    fn prune(&mut self) {
        while self.n_records() > self.max_records {
            let oldest = self.records.iter().enumerate().rev().find(|(_, r)| {
                r.state == ContState::ReplayOnly && r.hard_refs <= 0
            });
            match oldest {
                Some((i, _)) => {
                    self.remove_idx(i);
                }
                None => break,
            }
        }
    }

    fn remove_idx(&mut self, idx: usize) {
        self.demote_idx(idx);
        self.records.remove(idx);
        if let Some(s) = self.serial_live {
            if s == idx {
                self.serial_live = None;
            } else if s > idx {
                self.serial_live = Some(s - 1);
            }
        }
    }

    pub fn expire(&mut self, now: f64) {
        if self.ttl_s <= 0.0 || self.n_live() == 0 {
            return;
        }
        let mut i = 0;
        while i < self.records.len() {
            if self.records[i].state == ContState::LiveFrontier
                && now - self.records[i].publish_time > self.ttl_s
            {
                self.demote_idx(i);
            }
            i += 1;
        }
    }

    pub fn publish(
        &mut self,
        proto: Api,
        ids: &[String],
        owner: ContOwner,
        owner_id: i32,
        gen: u64,
        frontier: i32,
        now: f64,
    ) {
        if ids.is_empty() || gen == 0 || frontier <= 0 {
            return;
        }
        let ids: Vec<String> = {
            let mut out = Vec::new();
            for id in ids {
                if !id.is_empty() && !out.iter().any(|x| x == id) {
                    out.push(id.clone());
                }
            }
            out
        };
        if ids.is_empty() {
            return;
        }
        if self.max_records <= 0 {
            self.max_records = CONT_REGISTRY_MAX_DEFAULT;
        }
        match owner {
            ContOwner::SerialSession => {
                if let Some(i) = self.serial_live {
                    self.demote_idx(i);
                }
            }
            ContOwner::BatchBank => {
                if let Some(i) = self.records.iter().position(|r| {
                    r.state == ContState::LiveFrontier
                        && r.owner == ContOwner::BatchBank
                        && r.owner_id == owner_id
                }) {
                    self.demote_idx(i);
                }
            }
        }
        let rec = ContRecord {
            state: ContState::LiveFrontier,
            owner,
            protocol: proto as u8,
            owner_id: if owner == ContOwner::BatchBank {
                owner_id
            } else {
                0
            },
            owner_gen: gen,
            frontier,
            call_ids: ids,
            publish_time: now,
            hard_refs: 0,
            pin_expiry: 0.0,
        };
        self.records.insert(0, rec);
        if owner == ContOwner::SerialSession {
            self.serial_live = Some(0);
        } else if let Some(s) = self.serial_live {
            self.serial_live = Some(s + 1);
        }
        self.prune();
    }

    pub fn publish_serial(&mut self, proto: Api, ids: &[String], gen: u64, frontier: i32, now: f64) {
        self.publish(proto, ids, ContOwner::SerialSession, 0, gen, frontier, now);
    }

    pub fn publish_bank(
        &mut self,
        proto: Api,
        ids: &[String],
        bank: i32,
        gen: u64,
        frontier: i32,
        now: f64,
    ) {
        if bank < 0 {
            return;
        }
        self.publish(proto, ids, ContOwner::BatchBank, bank, gen, frontier, now);
    }

    pub fn demote_serial(&mut self) {
        if let Some(i) = self.serial_live {
            self.demote_idx(i);
        }
    }

    pub fn live_has_id(&mut self, proto: Api, id: &str, now: f64) -> bool {
        self.expire(now);
        self.find_idx(proto as u8, id)
            .map(|i| self.records[i].state == ContState::LiveFrontier)
            .unwrap_or(false)
    }

    pub fn id_known(&self, id: &str) -> bool {
        for proto in [Api::Openai as u8, Api::Anthropic as u8, Api::Responses as u8] {
            if self.find_idx(proto, id).is_some() {
                return true;
            }
        }
        false
    }

    pub fn resolve_serial(
        &mut self,
        proto: Api,
        ids: &[String],
        session_gen: u64,
        live_pos: i32,
        now: f64,
    ) -> bool {
        if ids.is_empty() || session_gen == 0 {
            return false;
        }
        self.expire(now);
        let Some(i) = self.find_idx(proto as u8, &ids[0]) else {
            return false;
        };
        let rec = &self.records[i];
        rec.state == ContState::LiveFrontier
            && rec.owner == ContOwner::SerialSession
            && rec.protocol == proto as u8
            && Self::set_eq(&rec.call_ids, ids)
            && rec.owner_gen == session_gen
            && rec.frontier == live_pos
    }

    pub fn bank_claim(
        &mut self,
        proto: Api,
        ids: &[String],
        now: f64,
    ) -> Option<(i32, u64, i32)> {
        if ids.is_empty() {
            return None;
        }
        self.expire(now);
        let i = self.find_idx(proto as u8, &ids[0])?;
        let rec = &self.records[i];
        if rec.state == ContState::LiveFrontier
            && rec.owner == ContOwner::BatchBank
            && rec.protocol == proto as u8
            && Self::set_eq(&rec.call_ids, ids)
        {
            Some((rec.owner_id, rec.owner_gen, rec.frontier))
        } else {
            None
        }
    }

    pub fn pin_live(&mut self, proto: Api, id: &str, now: f64) -> Option<usize> {
        self.expire(now);
        let i = self.find_idx(proto as u8, id)?;
        if self.records[i].state != ContState::LiveFrontier {
            return None;
        }
        self.records[i].hard_refs += 1;
        if self.pin_deadline_s > 0.0 {
            let expiry = now + self.pin_deadline_s;
            if expiry > self.records[i].pin_expiry {
                self.records[i].pin_expiry = expiry;
            }
        }
        Some(i)
    }

    pub fn unpin(&mut self, pin: usize) {
        if let Some(rec) = self.records.get_mut(pin) {
            if rec.hard_refs > 0 {
                rec.hard_refs -= 1;
            }
        }
    }

    pub fn unpin_id(&mut self, proto: Api, id: &str) {
        if let Some(i) = self.find_idx(proto as u8, id) {
            self.unpin(i);
        }
    }

    pub fn serial_hold(
        &mut self,
        proto: Api,
        req_ids: &[String],
        now: f64,
    ) -> Option<i32> {
        self.expire(now);
        let Some(i) = self.serial_live else {
            return None;
        };
        let rec = &self.records[i];
        if rec.protocol == proto as u8 && Self::set_eq(&rec.call_ids, req_ids) && !req_ids.is_empty()
        {
            return None;
        }
        let shed_w = if self.hold_shed_s < self.grace_s {
            self.hold_shed_s
        } else {
            self.grace_s
        };
        let shed_left = if shed_w > 0.0 {
            shed_w - (now - rec.publish_time)
        } else {
            0.0
        };
        let pinned = rec.hard_refs > 0
            && self.pin_deadline_s > 0.0
            && now < rec.pin_expiry;
        if shed_left <= 0.0 && !pinned {
            return None;
        }
        let mut left = shed_left;
        if pinned && rec.pin_expiry - now > left {
            left = rec.pin_expiry - now;
        }
        let mut retry = (left + 0.999) as i32;
        if retry < 1 {
            retry = 1;
        }
        Some(retry)
    }

    pub fn serial_live_hard_refs(&self) -> i32 {
        self.serial_live
            .and_then(|i| self.records.get(i))
            .map(|r| r.hard_refs)
            .unwrap_or(0)
    }

    pub fn serial_live_state(&self) -> Option<ContState> {
        self.serial_live
            .and_then(|i| self.records.get(i))
            .map(|r| r.state)
    }

    pub fn set_serial_publish_time(&mut self, t: f64) {
        if let Some(i) = self.serial_live {
            self.records[i].publish_time = t;
        }
    }

    pub fn set_serial_pin_expiry(&mut self, t: f64) {
        if let Some(i) = self.serial_live {
            self.records[i].pin_expiry = t;
        }
    }

    pub fn rewind_live_publish(&mut self, delta: f64) {
        for rec in &mut self.records {
            if rec.state == ContState::LiveFrontier {
                rec.publish_time -= delta;
            }
        }
    }
}

fn push_msg_tool_ids(ids: &mut Vec<String>, m: &crate::parse::ChatMsg) {
    if !m.tool_call_id.is_empty() && !ids.iter().any(|x| x == &m.tool_call_id) {
        ids.push(m.tool_call_id.clone());
    }
    for id in &m.tool_call_ids {
        if !id.is_empty() && !ids.iter().any(|x| x == id) {
            ids.push(id.clone());
        }
    }
}

fn anthropic_tool_result_tail(m: &crate::parse::ChatMsg) -> bool {
    m.role == "user" && (!m.tool_call_id.is_empty() || !m.tool_call_ids.is_empty())
}

/// Call ids C `*_prepare_live_continuation` would bind (not the full history).
pub fn live_tool_result_ids(api: Api, messages: &[crate::parse::ChatMsg]) -> Vec<String> {
    match api {
        Api::Anthropic => {
            let mut tail_end = messages.len();
            while tail_end > 0 && crate::render::role_is_system(&messages[tail_end - 1].role) {
                tail_end -= 1;
            }
            let mut tail_start = tail_end;
            while tail_start > 0 && anthropic_tool_result_tail(&messages[tail_start - 1]) {
                tail_start -= 1;
            }
            if tail_start == tail_end {
                return Vec::new();
            }
            let mut ids = Vec::new();
            for m in &messages[tail_start..tail_end] {
                push_msg_tool_ids(&mut ids, m);
            }
            ids
        }
        Api::Responses => {
            let mut tail_start = messages.len();
            while tail_start > 0 {
                let role = messages[tail_start - 1].role.as_str();
                if role != "tool" && role != "function" {
                    break;
                }
                tail_start -= 1;
            }
            if tail_start == messages.len() {
                return Vec::new();
            }
            let mut ids = Vec::new();
            if tail_start > 0 {
                let assistant = &messages[tail_start - 1];
                if assistant.role != "assistant" || assistant.calls.is_empty() {
                    return Vec::new();
                }
                for c in &assistant.calls {
                    if !c.id.is_empty() && !ids.iter().any(|x| x == &c.id) {
                        ids.push(c.id.clone());
                    }
                }
                return ids;
            }
            for m in &messages[tail_start..] {
                push_msg_tool_ids(&mut ids, m);
            }
            ids
        }
        Api::Openai => Vec::new(),
    }
}

fn csv(ids: &[&str]) -> Vec<String> {
    ids.iter().map(|s| (*s).to_string()).collect()
}

fn hold_line(r: Option<i32>) -> String {
    match r {
        Some(n) => format!("HOLD 1 retry={n}"),
        None => "HOLD 0".into(),
    }
}

/// Tape matching `tests/parity/cont_c_oracle` scripts.
pub fn dump_script(name: &str) -> String {
    let mut out = String::new();
    match name {
        "publish-resolve-demote" => {
            let mut r = ContRegistry::default();
            let now = 1000.0;
            r.publish_serial(
                Api::Anthropic,
                &csv(&["toolu_regA", "toolu_regB"]),
                7,
                100,
                now,
            );
            out.push_str(&format!(
                "live_anth_a={}\n",
                u32::from(r.live_has_id(Api::Anthropic, "toolu_regA", now))
            ));
            out.push_str(&format!(
                "live_anth_b={}\n",
                u32::from(r.live_has_id(Api::Anthropic, "toolu_regB", now))
            ));
            out.push_str(&format!(
                "live_resp_a={}\n",
                u32::from(r.live_has_id(Api::Responses, "toolu_regA", now))
            ));
            let ids = csv(&["toolu_regA", "toolu_regB"]);
            out.push_str(&format!(
                "resolve_ok={}\n",
                u32::from(r.resolve_serial(Api::Anthropic, &ids, 7, 100, now))
            ));
            out.push_str(&format!(
                "resolve_gen={}\n",
                u32::from(r.resolve_serial(Api::Anthropic, &ids, 8, 100, now))
            ));
            out.push_str(&format!(
                "resolve_pos={}\n",
                u32::from(r.resolve_serial(Api::Anthropic, &ids, 7, 101, now))
            ));
            out.push_str(&format!(
                "resolve_proto={}\n",
                u32::from(r.resolve_serial(Api::Responses, &ids, 7, 100, now))
            ));
            out.push_str(&format!(
                "resolve_sub={}\n",
                u32::from(r.resolve_serial(Api::Anthropic, &csv(&["toolu_regA"]), 7, 100, now))
            ));
            out.push_str(&format!(
                "resolve_sup={}\n",
                u32::from(r.resolve_serial(
                    Api::Anthropic,
                    &csv(&["toolu_regA", "toolu_regB", "toolu_regC"]),
                    7,
                    100,
                    now
                ))
            ));
            r.demote_serial();
            out.push_str(&format!(
                "live_after_demote={}\n",
                u32::from(r.live_has_id(Api::Anthropic, "toolu_regA", now))
            ));
            out.push_str(&format!(
                "resolve_after_demote={}\n",
                u32::from(r.resolve_serial(Api::Anthropic, &ids, 7, 100, now))
            ));
            out.push_str(&format!(
                "known_after_demote={}\n",
                u32::from(r.id_known("toolu_regA"))
            ));
        }
        "supersede-cap" => {
            let mut r = ContRegistry {
                max_records: 4,
                ..ContRegistry::default()
            };
            let now = 1000.0;
            for t in 1..=2 {
                r.publish_serial(
                    Api::Anthropic,
                    &csv(&[&format!("toolu_turn{t}")]),
                    3,
                    50 * t,
                    now,
                );
            }
            out.push_str(&format!(
                "live1={} live2={} n_live={} n_rec={}\n",
                u32::from(r.live_has_id(Api::Anthropic, "toolu_turn1", now)),
                u32::from(r.live_has_id(Api::Anthropic, "toolu_turn2", now)),
                r.n_live(),
                r.n_records()
            ));
            for t in 3..=8 {
                r.publish_serial(
                    Api::Anthropic,
                    &csv(&[&format!("toolu_turn{t}")]),
                    3,
                    50 * t,
                    now,
                );
            }
            out.push_str(&format!(
                "n_rec={} known1={} known2={} live8={} n_live={}\n",
                r.n_records(),
                u32::from(r.id_known("toolu_turn1")),
                u32::from(r.id_known("toolu_turn2")),
                u32::from(r.live_has_id(Api::Anthropic, "toolu_turn8", now)),
                r.n_live()
            ));
        }
        "grace-hold" => {
            let mut r = ContRegistry::default();
            r.publish_serial(Api::Anthropic, &csv(&["toolu_hold"]), 4, 70, 1000.0);
            out.push_str(&format!("{}\n", hold_line(r.serial_hold(Api::Openai, &[], 1001.0))));
            out.push_str(&format!(
                "{}\n",
                hold_line(r.serial_hold(Api::Anthropic, &csv(&["toolu_hold"]), 1001.0))
            ));
            out.push_str(&format!("{}\n", hold_line(r.serial_hold(Api::Openai, &[], 1011.0))));
            out.push_str(&format!(
                "still_live={}\n",
                match r.serial_live_state() {
                    Some(ContState::LiveFrontier) => 1,
                    _ => 0,
                }
            ));
            out.push_str(&format!("{}\n", hold_line(r.serial_hold(Api::Openai, &[], 1131.0))));
            let pin = r.pin_live(Api::Anthropic, "toolu_hold", 1131.0);
            out.push_str(&format!("{}\n", hold_line(r.serial_hold(Api::Openai, &[], 1131.0))));
            r.set_serial_pin_expiry(1130.0);
            out.push_str(&format!("{}\n", hold_line(r.serial_hold(Api::Openai, &[], 1131.0))));
            if let Some(p) = pin {
                r.unpin(p);
            }
            out.push_str(&format!("hard_refs={}\n", r.serial_live_hard_refs()));
        }
        "ttl" => {
            let mut r = ContRegistry::default();
            r.publish_serial(Api::Anthropic, &csv(&["toolu_ttl"]), 4, 70, 1000.0);
            out.push_str(&format!(
                "live_before={}\n",
                u32::from(r.live_has_id(Api::Anthropic, "toolu_ttl", 1000.0))
            ));
            out.push_str(&format!(
                "live_after={}\n",
                u32::from(r.live_has_id(Api::Anthropic, "toolu_ttl", 1301.0))
            ));
            out.push_str(&format!("n_live={}\n", r.n_live()));
            out.push_str(&format!(
                "resolve={}\n",
                u32::from(r.resolve_serial(
                    Api::Anthropic,
                    &csv(&["toolu_ttl"]),
                    4,
                    70,
                    1301.0
                ))
            ));
            out.push_str(&format!("known={}\n", u32::from(r.id_known("toolu_ttl"))));
        }
        "bank-claim" => {
            let mut r = ContRegistry::default();
            let now = 1000.0;
            r.publish_bank(Api::Anthropic, &csv(&["toolu_bk1"]), 2, 7, 100, now);
            out.push_str(&format!(
                "live={} resp={} serial_live={}\n",
                u32::from(r.live_has_id(Api::Anthropic, "toolu_bk1", now)),
                u32::from(r.live_has_id(Api::Responses, "toolu_bk1", now)),
                u32::from(r.serial_live.is_some())
            ));
            let claim = r.bank_claim(Api::Anthropic, &csv(&["toolu_bk1"]), now);
            out.push_str(&format!(
                "claim={}\n",
                match claim {
                    Some((b, g, f)) => format!("{b},{g},{f}"),
                    None => "-".into(),
                }
            ));
            out.push_str(&format!(
                "claim_resp={}\n",
                u32::from(r.bank_claim(Api::Responses, &csv(&["toolu_bk1"]), now).is_some())
            ));
            out.push_str(&format!(
                "resolve_serial={}\n",
                u32::from(r.resolve_serial(Api::Anthropic, &csv(&["toolu_bk1"]), 7, 100, now))
            ));
            r.publish_serial(Api::Anthropic, &csv(&["toolu_ser1"]), 9, 40, now);
            out.push_str(&format!("n_live={}\n", r.n_live()));
            r.demote_serial();
            out.push_str(&format!(
                "n_live_after={} live_bk1={}\n",
                r.n_live(),
                u32::from(r.live_has_id(Api::Anthropic, "toolu_bk1", now))
            ));
            r.publish_bank(Api::Anthropic, &csv(&["toolu_bk2"]), 2, 8, 120, now);
            out.push_str(&format!(
                "live_bk1={} live_bk2={}\n",
                u32::from(r.live_has_id(Api::Anthropic, "toolu_bk1", now)),
                u32::from(r.live_has_id(Api::Anthropic, "toolu_bk2", now))
            ));
            r.publish_bank(Api::Responses, &csv(&["toolu_bk3"]), 3, 2, 80, now);
            out.push_str(&format!("n_live={}\n", r.n_live()));
            r.publish_bank(Api::Anthropic, &csv(&["toolu_bk_dead"]), 4, 0, 100, now);
            r.publish_bank(Api::Anthropic, &csv(&["toolu_bk_dead"]), 4, 5, 0, now);
            out.push_str(&format!(
                "known_dead={}\n",
                u32::from(r.id_known("toolu_bk_dead"))
            ));
            r.rewind_live_publish(301.0);
            out.push_str(&format!(
                "ttl_live={} n_live={} known_bk2={}\n",
                u32::from(r.live_has_id(Api::Anthropic, "toolu_bk2", now)),
                r.n_live(),
                u32::from(r.id_known("toolu_bk2"))
            ));
        }
        _ => out.push_str("ERROR unknown-script\n"),
    }
    out
}
