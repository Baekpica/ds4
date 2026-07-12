#!/usr/bin/env python3
"""Validate and replay DSpark terminal-yield-quench traces offline.

The server emits one ``DSPARK_TRACE`` line per request.  Its ``trace=`` tail is
a sequence of ``y:c:nd:live:spec:ms`` records: committed tokens, comparisons,
packed drafts, live banks, whether speculation ran, and wall milliseconds.
``CONT_MTP_ACCEPT(DSpark)`` lines are independent call-level totals, while the
immediately following ``DSPARK_SHADOW`` line records the engine's shadow-policy
state.  Unrelated log lines and the non-DSpark aggregate form are ignored.

The replay initializes yield EWMA to ``guard`` and debt to zero.  Only records
with ``spec=1`` supply evidence.  The first qualifying speculative observation
is reported as a zero-based *speculative-step ordinal*, matching the engine;
the code separately retains its full-record index.  The triggering step has
already executed speculatively, so it is included in the paid pre-quench prefix.
Later traced speculative observations are used for shadow/recovery analysis,
but terminal-quench economics regenerates every remaining token plainly.

Times are expressed in plain-step equivalents.  A speculative trace step costs
``C`` and a plain trace step costs one.  Always-plain time is the emitted-token
count.  After a quench, the observed prefix keeps its actual spec/plain costs
and all remaining tokens cost one each.  Speedup is always-plain time divided
by modeled time.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_GUARD = 2.285
DEFAULT_ALPHA = 0.125
DEFAULT_MINEV = 8
DEFAULT_BUDGET = 4.0
DEFAULT_COST = 2.17
FLOAT_TOLERANCE = 1.0e-3

GRID_GUARDS = (2.22, 2.285, 2.30)
GRID_ALPHAS = (0.083, 0.125)
GRID_MINEVS = (4, 8, 16)
GRID_BUDGETS = (2.0, 4.0)
# Debt floor sweep: 0 = engine-shadow reflected walk, mid values = bounded
# credit, 1e9 = effectively pure cumulative regret.
GRID_CREDIT_CAPS = (0.0, 4.0, 8.0, 16.0, 1e9)

TRACE_MARKER = "ds4: DSPARK_TRACE"
SHADOW_MARKER = "ds4: DSPARK_SHADOW"
AGGREGATE_MARKER = "ds4: CONT_MTP_ACCEPT(DSpark)"

_FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_UINT_RE = re.compile(r"\d+\Z")

TRACE_RE = re.compile(
    r"ds4:\s+DSPARK_TRACE\s+"
    r"bank=(?P<bank>\d+)\s+pos0=(?P<pos0>\d+)\s+D=(?P<D>\d+)\s+"
    r"steps=(?P<steps>\d+)\s+emit=(?P<emit>\d+)\s+"
    r"drafts=(?P<drafts>\d+)\s+hits=(?P<hits>\d+)\s+"
    r"trace=(?P<records>.*?)\s*$"
)

AGGREGATE_RE = re.compile(
    rf"ds4:\s+CONT_MTP_ACCEPT\(DSpark\)\s+"
    rf"D=(?P<D>\d+)\s+steps=(?P<steps>\d+)\s+emit=(?P<emit>\d+)\s+"
    rf"drafts=(?P<drafts>\d+)\s+hits=(?P<hits>\d+)\s+"
    rf"accept=(?P<accept>{_FLOAT})%\s+tok/step=(?P<tok_per_step>{_FLOAT})\s*$"
)

SHADOW_RE = re.compile(
    rf"ds4:\s+DSPARK_SHADOW\s+bank=(?P<bank>\d+)\s+"
    rf"guard=(?P<guard>{_FLOAT})\s+alpha=(?P<alpha>{_FLOAT})\s+"
    rf"minev=(?P<minev>\d+)\s+budget=(?P<budget>{_FLOAT})\s+"
    rf"(?:ccap=(?P<ccap>inf|{_FLOAT})\s+)?"
    rf"yewma=(?P<yewma>{_FLOAT})\s+debt=(?P<debt>{_FLOAT})\s+"
    rf"quench_step=(?P<quench_step>-?\d+)\s+"
    rf"post_quench_yield=(?P<post_quench_yield>{_FLOAT})\s*$"
)


@dataclass(frozen=True)
class TraceStep:
    y: int
    comparisons: int
    packed_drafts: int
    live_banks: int
    spec: bool
    milliseconds: float


@dataclass(frozen=True)
class PolicyParams:
    guard: float = DEFAULT_GUARD
    alpha: float = DEFAULT_ALPHA
    minev: int = DEFAULT_MINEV
    budget: float = DEFAULT_BUDGET
    # Debt floor: debt is clamped at -credit_cap. 0 reproduces the engine
    # shadow (reflected walk -- false-quenches long bursty winners); large
    # values approach pure cumulative regret (never quenches stationary
    # winners, slower to react to a mid-request regime change).
    credit_cap: float = 0.0


@dataclass(frozen=True)
class ShadowRecord:
    source: str
    line_number: int
    bank: int
    params: PolicyParams
    yewma: float
    debt: float
    quench_step: int
    post_quench_yield: float


@dataclass
class Request:
    source: str
    line_number: int
    bank: int
    pos0: int
    depth: int
    steps: int
    emit: int
    drafts: int
    hits: int
    records: Tuple[TraceStep, ...]
    shadow: Optional[ShadowRecord] = None


@dataclass(frozen=True)
class AggregateRecord:
    source: str
    line_number: int
    depth: int
    steps: int
    emit: int
    drafts: int
    hits: int
    accept_percent: float
    tokens_per_step: float


@dataclass(frozen=True)
class ParseIssue:
    source: str
    line_number: int
    message: str


@dataclass
class ParsedLog:
    source: str
    requests: List[Request]
    aggregates: List[AggregateRecord]
    shadows: List[ShadowRecord]
    issues: List[ParseIssue]


@dataclass(frozen=True)
class PolicyResult:
    params: PolicyParams
    yewma: float
    debt: float
    quench_step: int
    trigger_record_index: Optional[int]
    spec_steps: int
    post_quench_yield: float
    post_quench_spec_steps: int


@dataclass(frozen=True)
class EconomicResult:
    always_spec_time: float
    always_plain_time: float
    quenched_time: float
    oracle_time: float
    speedup_spec: float
    speedup_quenched: float
    oracle_speedup: float
    tokens_after_quench: int


@dataclass(frozen=True)
class ReplaySummary:
    params: PolicyParams
    cost: float
    n_requests: int
    mean_speedup_spec: float
    min_speedup_spec: float
    mean_speedup_quenched: float
    min_speedup_quenched: float
    oracle_retention: float
    mean_ratio_retention: float
    n_quenched: int
    false_quench_count: int
    mean_post_quench_yield: Optional[float]
    post_quench_observable_count: int
    recovery_count: int
    recovery_fraction: Optional[float]
    mean_tokens_after_quench: Optional[float]


def _finite_float(text: str, field: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{field} is not finite")
    return value


def _parse_trace_steps(text: str) -> Tuple[TraceStep, ...]:
    if not text.strip():
        return ()
    records: List[TraceStep] = []
    for record_number, token in enumerate(text.split(), 1):
        fields = token.split(":")
        if len(fields) != 6:
            raise ValueError(
                f"record {record_number} has {len(fields)} fields instead of 6"
            )
        if not all(_UINT_RE.fullmatch(field) for field in fields[:5]):
            raise ValueError(f"record {record_number} has a non-unsigned integer field")
        y, comparisons, packed, live, spec_int = (int(field) for field in fields[:5])
        milliseconds = _finite_float(fields[5], f"record {record_number} ms")
        if y < 1:
            raise ValueError(f"record {record_number} has y={y}; expected y >= 1")
        if spec_int not in (0, 1):
            raise ValueError(
                f"record {record_number} has spec={spec_int}; expected 0 or 1"
            )
        records.append(
            TraceStep(y, comparisons, packed, live, bool(spec_int), milliseconds)
        )
    return tuple(records)


def parse_lines(source: str, lines: Iterable[str]) -> ParsedLog:
    """Parse relevant records once, tolerating arbitrary unrelated lines."""
    requests: List[Request] = []
    aggregates: List[AggregateRecord] = []
    shadows: List[ShadowRecord] = []
    issues: List[ParseIssue] = []
    pending_by_bank: Dict[int, List[int]] = {}

    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.rstrip("\r\n")

        if TRACE_MARKER in line:
            match = TRACE_RE.search(line)
            if match is None:
                issues.append(ParseIssue(source, line_number, "malformed DSPARK_TRACE line"))
                continue
            try:
                records = _parse_trace_steps(match.group("records"))
                request = Request(
                    source=source,
                    line_number=line_number,
                    bank=int(match.group("bank")),
                    pos0=int(match.group("pos0")),
                    depth=int(match.group("D")),
                    steps=int(match.group("steps")),
                    emit=int(match.group("emit")),
                    drafts=int(match.group("drafts")),
                    hits=int(match.group("hits")),
                    records=records,
                )
            except ValueError as exc:
                issues.append(ParseIssue(source, line_number, f"malformed DSPARK_TRACE: {exc}"))
                continue
            requests.append(request)
            pending_by_bank.setdefault(request.bank, []).append(len(requests) - 1)
            continue

        if SHADOW_MARKER in line:
            match = SHADOW_RE.search(line)
            if match is None:
                issues.append(ParseIssue(source, line_number, "malformed DSPARK_SHADOW line"))
                continue
            try:
                # ccap absent (pre-Phase-2 logs) = the engine's original
                # zero-clamp; "inf" = pure cumulative regret (no debt floor).
                ccap_text = match.group("ccap")
                if ccap_text is None:
                    credit_cap = 0.0
                elif ccap_text == "inf":
                    credit_cap = math.inf
                else:
                    credit_cap = _finite_float(ccap_text, "ccap")
                params = PolicyParams(
                    guard=_finite_float(match.group("guard"), "guard"),
                    alpha=_finite_float(match.group("alpha"), "alpha"),
                    minev=int(match.group("minev")),
                    budget=_finite_float(match.group("budget"), "budget"),
                    credit_cap=credit_cap,
                )
                shadow = ShadowRecord(
                    source=source,
                    line_number=line_number,
                    bank=int(match.group("bank")),
                    params=params,
                    yewma=_finite_float(match.group("yewma"), "yewma"),
                    debt=_finite_float(match.group("debt"), "debt"),
                    quench_step=int(match.group("quench_step")),
                    post_quench_yield=_finite_float(
                        match.group("post_quench_yield"), "post_quench_yield"
                    ),
                )
            except ValueError as exc:
                issues.append(ParseIssue(source, line_number, f"malformed DSPARK_SHADOW: {exc}"))
                continue
            shadows.append(shadow)
            pending = pending_by_bank.get(shadow.bank, [])
            if not pending:
                issues.append(
                    ParseIssue(
                        source,
                        line_number,
                        f"DSPARK_SHADOW bank={shadow.bank} has no preceding unmatched trace",
                    )
                )
                continue
            # Flush prints TRACE then SHADOW.  LIFO keeps a missing older shadow
            # from mis-associating every later reuse of the same bank.
            request_index = pending.pop()
            requests[request_index].shadow = shadow
            continue

        if AGGREGATE_MARKER in line:
            match = AGGREGATE_RE.search(line)
            if match is None:
                issues.append(
                    ParseIssue(source, line_number, "malformed CONT_MTP_ACCEPT(DSpark) line")
                )
                continue
            try:
                aggregate = AggregateRecord(
                    source=source,
                    line_number=line_number,
                    depth=int(match.group("D")),
                    steps=int(match.group("steps")),
                    emit=int(match.group("emit")),
                    drafts=int(match.group("drafts")),
                    hits=int(match.group("hits")),
                    accept_percent=_finite_float(match.group("accept"), "accept"),
                    tokens_per_step=_finite_float(
                        match.group("tok_per_step"), "tok/step"
                    ),
                )
            except ValueError as exc:
                issues.append(
                    ParseIssue(
                        source, line_number, f"malformed CONT_MTP_ACCEPT(DSpark): {exc}"
                    )
                )
                continue
            if not (aggregate.steps == 0 and aggregate.emit == 0):
                aggregates.append(aggregate)

    return ParsedLog(source, requests, aggregates, shadows, issues)


def parse_log(path: str) -> ParsedLog:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return parse_lines(path, handle)


def request_consistency_mismatches(request: Request) -> List[Tuple[str, int, int]]:
    computed = {
        "steps": len(request.records),
        "emit": sum(step.y for step in request.records),
        "drafts": sum(step.comparisons for step in request.records),
        "hits": sum(step.y - 1 for step in request.records),
    }
    declared = {
        "steps": request.steps,
        "emit": request.emit,
        "drafts": request.drafts,
        "hits": request.hits,
    }
    return [
        (field, declared[field], computed[field])
        for field in ("steps", "emit", "drafts", "hits")
        if declared[field] != computed[field]
    ]


def replay_policy(request: Request, params: PolicyParams) -> PolicyResult:
    """Recompute engine shadow state and remember the terminal trigger location."""
    yewma = params.guard
    debt = 0.0
    spec_steps = 0
    quench_step = -1
    trigger_record_index: Optional[int] = None
    post_sum = 0.0
    post_count = 0

    for record_index, step in enumerate(request.records):
        if not step.spec:
            continue
        if quench_step >= 0:
            post_sum += step.y
            post_count += 1
        yewma = (1.0 - params.alpha) * yewma + params.alpha * step.y
        debt += params.guard - step.y
        if debt < -params.credit_cap:
            debt = -params.credit_cap
        spec_steps += 1
        if (
            quench_step < 0
            and spec_steps >= params.minev
            and yewma < params.guard
            and debt > params.budget
        ):
            quench_step = spec_steps - 1
            trigger_record_index = record_index

    post_yield = post_sum / post_count if quench_step >= 0 and post_count else -1.0
    return PolicyResult(
        params=params,
        yewma=yewma,
        debt=debt,
        quench_step=quench_step,
        trigger_record_index=trigger_record_index,
        spec_steps=spec_steps,
        post_quench_yield=post_yield,
        post_quench_spec_steps=post_count,
    )


def economic_model(
    request: Request, policy: PolicyResult, cost: float
) -> EconomicResult:
    """Apply the unitless always-spec/plain/terminal-quench time model."""
    spec_steps = sum(1 for step in request.records if step.spec)
    plain_steps = len(request.records) - spec_steps
    always_spec_time = spec_steps * cost + plain_steps
    always_plain_time = float(request.emit)
    tokens_after_quench = 0

    if policy.trigger_record_index is None:
        quenched_time = always_spec_time
    else:
        prefix = request.records[: policy.trigger_record_index + 1]
        prefix_spec = sum(1 for step in prefix if step.spec)
        prefix_plain = len(prefix) - prefix_spec
        prefix_tokens = sum(step.y for step in prefix)
        tokens_after_quench = request.emit - prefix_tokens
        quenched_time = prefix_spec * cost + prefix_plain + tokens_after_quench

    speedup_spec = always_plain_time / always_spec_time
    speedup_quenched = always_plain_time / quenched_time
    oracle_speedup = max(speedup_spec, 1.0)
    oracle_time = min(always_spec_time, always_plain_time)
    return EconomicResult(
        always_spec_time=always_spec_time,
        always_plain_time=always_plain_time,
        quenched_time=quenched_time,
        oracle_time=oracle_time,
        speedup_spec=speedup_spec,
        speedup_quenched=speedup_quenched,
        oracle_speedup=oracle_speedup,
        tokens_after_quench=tokens_after_quench,
    )


def summarize_replay(
    requests: Sequence[Request], params: PolicyParams, cost: float
) -> ReplaySummary:
    policies = [replay_policy(request, params) for request in requests]
    economics = [
        economic_model(request, policy, cost)
        for request, policy in zip(requests, policies)
    ]
    spec_speedups = [item.speedup_spec for item in economics]
    quenched_speedups = [item.speedup_quenched for item in economics]
    total_tokens = sum(request.emit for request in requests)
    total_quenched_time = sum(item.quenched_time for item in economics)
    total_oracle_time = sum(item.oracle_time for item in economics)
    quenched_throughput = total_tokens / total_quenched_time
    oracle_throughput = total_tokens / total_oracle_time

    quenched_indices = [
        index for index, policy in enumerate(policies) if policy.quench_step >= 0
    ]
    observable_post_yields = [
        policies[index].post_quench_yield
        for index in quenched_indices
        if policies[index].post_quench_spec_steps > 0
    ]
    recovery_count = sum(
        1
        for index in quenched_indices
        if policies[index].post_quench_spec_steps > 0
        and policies[index].post_quench_yield > params.guard
    )
    n_quenched = len(quenched_indices)

    return ReplaySummary(
        params=params,
        cost=cost,
        n_requests=len(requests),
        mean_speedup_spec=statistics.fmean(spec_speedups),
        min_speedup_spec=min(spec_speedups),
        mean_speedup_quenched=statistics.fmean(quenched_speedups),
        min_speedup_quenched=min(quenched_speedups),
        oracle_retention=quenched_throughput / oracle_throughput,
        mean_ratio_retention=statistics.fmean(
            item.speedup_quenched / item.oracle_speedup for item in economics
        ),
        n_quenched=n_quenched,
        false_quench_count=sum(
            1
            for index in quenched_indices
            if economics[index].speedup_spec > 1.0
        ),
        mean_post_quench_yield=(
            statistics.fmean(observable_post_yields) if observable_post_yields else None
        ),
        post_quench_observable_count=len(observable_post_yields),
        recovery_count=recovery_count,
        recovery_fraction=(recovery_count / n_quenched if n_quenched else None),
        mean_tokens_after_quench=(
            statistics.fmean(
                economics[index].tokens_after_quench for index in quenched_indices
            )
            if quenched_indices
            else None
        ),
    )


def _totals(items: Iterable[object]) -> Tuple[int, int, int, int]:
    emit = drafts = hits = steps = 0
    for item in items:
        emit += int(getattr(item, "emit"))
        drafts += int(getattr(item, "drafts"))
        hits += int(getattr(item, "hits"))
        steps += int(getattr(item, "steps"))
    return emit, drafts, hits, steps


def _totals_text(totals: Tuple[int, int, int, int]) -> str:
    emit, drafts, hits, steps = totals
    return f"emit={emit} drafts={drafts} hits={hits} steps={steps}"


def _float_matches(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= FLOAT_TOLERANCE + 1.0e-12


def _shadow_mismatches(request: Request) -> List[str]:
    shadow = request.shadow
    if shadow is None:
        return []
    result = replay_policy(request, shadow.params)
    mismatches: List[str] = []
    for field, actual, expected in (
        ("yewma", shadow.yewma, result.yewma),
        ("debt", shadow.debt, result.debt),
        (
            "post_quench_yield",
            shadow.post_quench_yield,
            result.post_quench_yield,
        ),
    ):
        if not _float_matches(actual, expected):
            mismatches.append(
                f"{field} engine={actual:.6f} recomputed={expected:.6f}"
            )
    if shadow.quench_step != result.quench_step:
        mismatches.append(
            f"quench_step engine={shadow.quench_step} recomputed={result.quench_step}"
        )
    return mismatches


def validate_parsed_log(parsed: ParsedLog) -> bool:
    print(f"VALIDATE {parsed.source}")
    parse_ok = not parsed.issues
    for issue in parsed.issues:
        print(f"{issue.source}:{issue.line_number}: PARSE FAIL: {issue.message}")
    print(f"PARSE: {'PASS' if parse_ok else 'FAIL'}")

    internal_ok = bool(parsed.requests)
    if not parsed.requests:
        print(f"{parsed.source}: TRACE FAIL: no DSPARK_TRACE requests found")
    for request in parsed.requests:
        mismatches = request_consistency_mismatches(request)
        if mismatches:
            internal_ok = False
            details = "; ".join(
                f"{field} header={declared} records={computed}"
                for field, declared, computed in mismatches
            )
            print(
                f"{request.source}:{request.line_number}: TRACE FAIL "
                f"bank={request.bank}: {details}"
            )
    print(
        f"TRACE INTERNAL: {'PASS' if internal_ok else 'FAIL'} "
        f"requests={len(parsed.requests)}"
    )

    trace_totals = _totals(parsed.requests)
    aggregate_totals = _totals(parsed.aggregates)
    totals_ok = trace_totals == aggregate_totals
    print(f"TRACE TOTALS: {_totals_text(trace_totals)}")
    print(
        f"DS4 AGG TOTALS: {_totals_text(aggregate_totals)} "
        f"runs={len(parsed.aggregates)}"
    )
    print(f"TOTALS: {'PASS' if totals_ok else 'FAIL'}")

    shadow_ok = True
    shadow_checked = 0
    for request in parsed.requests:
        if request.shadow is None:
            continue
        shadow_checked += 1
        mismatches = _shadow_mismatches(request)
        if mismatches:
            shadow_ok = False
            print(
                f"{request.shadow.source}:{request.shadow.line_number}: SHADOW FAIL "
                f"bank={request.bank}: {'; '.join(mismatches)}"
            )
    print(
        f"SHADOW: {'PASS' if shadow_ok else 'FAIL'} checked={shadow_checked} "
        f"missing={len(parsed.requests) - shadow_checked}"
    )

    passed = parse_ok and internal_ok and totals_ok and shadow_ok
    print(f"FILE: {'PASS' if passed else 'FAIL'}")
    return passed


def command_validate(paths: Sequence[str]) -> int:
    all_ok = True
    completed = 0
    for path in paths:
        try:
            parsed = parse_log(path)
        except OSError as exc:
            print(f"VALIDATE {path}")
            print(f"{path}: READ FAIL: {exc}")
            print("FILE: FAIL")
            all_ok = False
            completed += 1
            continue
        if not validate_parsed_log(parsed):
            all_ok = False
        completed += 1
    print(f"VALIDATE: {'PASS' if all_ok else 'FAIL'} files={completed}")
    return 0 if all_ok else 1


def _load_replayable(paths: Sequence[str]) -> Optional[List[Request]]:
    parsed_logs: List[ParsedLog] = []
    ok = True
    for path in paths:
        try:
            parsed = parse_log(path)
        except OSError as exc:
            print(f"{path}: READ FAIL: {exc}", file=sys.stderr)
            ok = False
            continue
        parsed_logs.append(parsed)
        for issue in parsed.issues:
            print(
                f"{issue.source}:{issue.line_number}: PARSE FAIL: {issue.message}",
                file=sys.stderr,
            )
            ok = False
        for request in parsed.requests:
            mismatches = request_consistency_mismatches(request)
            if mismatches:
                details = "; ".join(
                    f"{field} header={declared} records={computed}"
                    for field, declared, computed in mismatches
                )
                print(
                    f"{request.source}:{request.line_number}: TRACE FAIL: {details}",
                    file=sys.stderr,
                )
                ok = False
    # Empty traces (steps=0: requests that finished at seed, e.g. prefill probes)
    # are valid telemetry but carry no policy decisions -- skip them in replay
    # rather than failing; validate still checks them strictly.
    skipped_empty = 0
    requests = []
    for parsed in parsed_logs:
        for request in parsed.requests:
            if request.emit <= 0 or not request.records:
                skipped_empty += 1
                continue
            requests.append(request)
    if skipped_empty:
        print(f"skipped {skipped_empty} empty (steps=0) trace requests", file=sys.stderr)
    if not requests:
        print("no DSPARK_TRACE requests found", file=sys.stderr)
        ok = False
    return requests if ok else None


def _optional_float(value: Optional[float], digits: int = 6) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def print_single_summary(summary: ReplaySummary) -> None:
    params = summary.params
    print(
        "PARAMS "
        f"guard={params.guard:g} alpha={params.alpha:g} minev={params.minev} "
        f"budget={params.budget:g} cost={summary.cost:g}"
    )
    print(f"REQUESTS: n={summary.n_requests}")
    print(
        f"SPEEDUP SPEC: mean={summary.mean_speedup_spec:.6f} "
        f"floor={summary.min_speedup_spec:.6f}"
    )
    print(
        f"SPEEDUP QUENCHED: mean={summary.mean_speedup_quenched:.6f} "
        f"floor={summary.min_speedup_quenched:.6f}"
    )
    print(
        f"ORACLE RETENTION: throughput_weighted={summary.oracle_retention:.6f} "
        f"mean_of_ratios={summary.mean_ratio_retention:.6f}"
    )
    print(
        f"QUENCHES: n={summary.n_quenched} "
        f"false_quench={summary.false_quench_count}"
    )
    recovery_denominator = summary.n_quenched
    print(
        "RECOVERY: "
        f"mean_post_quench_yield={_optional_float(summary.mean_post_quench_yield)} "
        f"observable={summary.post_quench_observable_count}/{recovery_denominator} "
        f"terminal_recovery={summary.recovery_count}/{recovery_denominator} "
        f"fraction={_optional_float(summary.recovery_fraction)} "
        f"mean_tokens_after_quench={_optional_float(summary.mean_tokens_after_quench)}"
    )


def build_grid_summaries(
    requests: Sequence[Request], cost: float
) -> List[ReplaySummary]:
    indexed: List[Tuple[int, ReplaySummary]] = []
    original_index = 0
    for guard in GRID_GUARDS:
        for alpha in GRID_ALPHAS:
            for minev in GRID_MINEVS:
                for budget in GRID_BUDGETS:
                    for credit_cap in GRID_CREDIT_CAPS:
                        params = PolicyParams(guard, alpha, minev, budget, credit_cap)
                        indexed.append(
                            (original_index, summarize_replay(requests, params, cost))
                        )
                        original_index += 1
    indexed.sort(
        key=lambda item: (
            0 if item[1].min_speedup_quenched >= 0.99 else 1,
            -item[1].mean_speedup_quenched,
            item[0],
        )
    )
    return [summary for _, summary in indexed]


def print_grid(summaries: Sequence[ReplaySummary]) -> None:
    print(f"GRID: rows={len(summaries)} cost={summaries[0].cost:g}")
    header = (
        f"{'rank':>4} {'guard':>6} {'alpha':>6} {'minev':>5} {'budget':>6} {'ccap':>6} "
        f"{'n':>5} {'spec_mean':>9} {'spec_floor':>10} "
        f"{'q_mean':>9} {'q_floor':>9} {'oracle_ret':>10} {'mean_ret':>9} "
        f"{'nq':>5} {'false':>5} {'post_y':>8} {'recover':>9} "
        f"{'rec_frac':>8} {'tok_after':>9}"
    )
    print(header)
    for rank, summary in enumerate(summaries, 1):
        params = summary.params
        post_yield = _optional_float(summary.mean_post_quench_yield, 4)
        recovery_fraction = _optional_float(summary.recovery_fraction, 4)
        tokens_after = _optional_float(summary.mean_tokens_after_quench, 3)
        print(
            f"{rank:4d} {params.guard:6.3f} {params.alpha:6.3f} "
            f"{params.minev:5d} {params.budget:6.1f} "
            f"{('inf' if params.credit_cap >= 1e8 else format(params.credit_cap, '.1f')):>6} "
            f"{summary.n_requests:5d} "
            f"{summary.mean_speedup_spec:9.5f} {summary.min_speedup_spec:10.5f} "
            f"{summary.mean_speedup_quenched:9.5f} "
            f"{summary.min_speedup_quenched:9.5f} "
            f"{summary.oracle_retention:10.5f} "
            f"{summary.mean_ratio_retention:9.5f} {summary.n_quenched:5d} "
            f"{summary.false_quench_count:5d} {post_yield:>8} "
            f"{summary.recovery_count:4d}/{summary.n_quenched:<4d} "
            f"{recovery_fraction:>8} {tokens_after:>9}"
        )


CSV_FIELDS = (
    "rank",
    "guard",
    "alpha",
    "minev",
    "budget",
    "credit_cap",
    "cost",
    "n_requests",
    "mean_speedup_spec",
    "min_speedup_spec",
    "mean_speedup_quenched",
    "min_speedup_quenched",
    "oracle_retention",
    "mean_ratio_retention",
    "n_quenched",
    "false_quench_count",
    "mean_post_quench_yield",
    "post_quench_observable_count",
    "recovery_count",
    "recovery_fraction",
    "mean_tokens_after_quench",
)


def _csv_float(value: Optional[float]) -> object:
    return "" if value is None else f"{value:.12g}"


def write_grid_csv(path: str, summaries: Sequence[ReplaySummary]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for rank, summary in enumerate(summaries, 1):
            params = summary.params
            writer.writerow(
                {
                    "rank": rank,
                    "guard": f"{params.guard:.12g}",
                    "alpha": f"{params.alpha:.12g}",
                    "minev": params.minev,
                    "budget": f"{params.budget:.12g}",
                    "credit_cap": f"{params.credit_cap:.12g}",
                    "cost": f"{summary.cost:.12g}",
                    "n_requests": summary.n_requests,
                    "mean_speedup_spec": _csv_float(summary.mean_speedup_spec),
                    "min_speedup_spec": _csv_float(summary.min_speedup_spec),
                    "mean_speedup_quenched": _csv_float(
                        summary.mean_speedup_quenched
                    ),
                    "min_speedup_quenched": _csv_float(
                        summary.min_speedup_quenched
                    ),
                    "oracle_retention": _csv_float(summary.oracle_retention),
                    "mean_ratio_retention": _csv_float(
                        summary.mean_ratio_retention
                    ),
                    "n_quenched": summary.n_quenched,
                    "false_quench_count": summary.false_quench_count,
                    "mean_post_quench_yield": _csv_float(
                        summary.mean_post_quench_yield
                    ),
                    "post_quench_observable_count": (
                        summary.post_quench_observable_count
                    ),
                    "recovery_count": summary.recovery_count,
                    "recovery_fraction": _csv_float(summary.recovery_fraction),
                    "mean_tokens_after_quench": _csv_float(
                        summary.mean_tokens_after_quench
                    ),
                }
            )


def command_replay(
    paths: Sequence[str],
    grid: bool,
    params: PolicyParams,
    cost: float,
    csv_path: Optional[str],
) -> int:
    requests = _load_replayable(paths)
    if requests is None:
        return 1
    if grid:
        summaries = build_grid_summaries(requests, cost)
        print_grid(summaries)
        if csv_path is not None:
            try:
                write_grid_csv(csv_path, summaries)
            except OSError as exc:
                print(f"{csv_path}: CSV WRITE FAIL: {exc}", file=sys.stderr)
                return 1
            print(f"WROTE CSV {csv_path}")
    else:
        print_single_summary(summarize_replay(requests, params, cost))
    return 0


def command_inspect(path: str) -> int:
    requests = _load_replayable((path,))
    if requests is None:
        return 1
    for request in requests:
        yield_per_step = request.emit / request.steps
        accept = request.hits / request.drafts if request.drafts else None
        spec_fraction = sum(step.spec for step in request.records) / len(request.records)
        mean_ms = statistics.fmean(step.milliseconds for step in request.records)
        shadow_quench = (
            str(request.shadow.quench_step) if request.shadow is not None else "n/a"
        )
        print(
            f"{request.source}:{request.line_number} bank={request.bank} "
            f"pos0={request.pos0} D={request.depth} steps={request.steps} "
            f"emit={request.emit} yield={yield_per_step:.6f} "
            f"accept={_optional_float(accept)} spec_fraction={spec_fraction:.6f} "
            f"mean_step_ms={mean_ms:.3f} shadow_quench_step={shadow_quench}"
        )
    return 0


def _selftest_request(name: str, text: str) -> Tuple[Request, ParsedLog]:
    parsed = parse_lines(f"<selftest:{name}>", text.strip().splitlines())
    assert not parsed.issues, parsed.issues
    assert len(parsed.requests) == 1
    request = parsed.requests[0]
    assert not request_consistency_mismatches(request)
    return request, parsed


def _assert_close(actual: float, expected: float) -> None:
    assert math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-12), (
        actual,
        expected,
    )


def run_selftest() -> int:
    params = PolicyParams(guard=2.0, alpha=0.5, minev=2, budget=1.0)
    cost = 2.0
    try:
        all_accept, _ = _selftest_request(
            "all-accept",
            """
ds4: DSPARK_TRACE bank=0 pos0=10 D=2 steps=3 emit=9 drafts=6 hits=6 trace=3:2:2:1:1:1.0 3:2:2:1:1:1.0 3:2:2:1:1:1.0
""",
        )
        policy = replay_policy(all_accept, params)
        economics = economic_model(all_accept, policy, cost)
        assert policy.quench_step == -1
        _assert_close(policy.yewma, 2.875)
        _assert_close(policy.debt, 0.0)
        _assert_close(economics.speedup_spec, 1.5)
        _assert_close(economics.speedup_quenched, 1.5)
        print("selftest all-accept: PASS")

        all_reject, _ = _selftest_request(
            "all-reject",
            """
ds4: DSPARK_TRACE bank=1 pos0=20 D=3 steps=4 emit=4 drafts=4 hits=0 trace=1:1:3:1:1:1.0 1:1:3:1:1:1.0 1:1:3:1:1:1.0 1:1:3:1:1:1.0
""",
        )
        policy = replay_policy(all_reject, params)
        economics = economic_model(all_reject, policy, cost)
        assert policy.quench_step == 1
        assert policy.trigger_record_index == 1
        _assert_close(policy.yewma, 1.0625)
        _assert_close(policy.debt, 4.0)
        _assert_close(policy.post_quench_yield, 1.0)
        _assert_close(economics.speedup_spec, 0.5)
        _assert_close(economics.speedup_quenched, 2.0 / 3.0)
        assert economics.tokens_after_quench == 2
        print("selftest all-reject: PASS")

        recovery, recovery_parsed = _selftest_request(
            "recovery-after-slump",
            """
ds4: DSPARK_TRACE bank=2 pos0=30 D=3 steps=4 emit=10 drafts=8 hits=6 trace=1:1:3:1:1:1.0 1:1:3:1:1:1.0 4:3:3:1:1:1.0 4:3:3:1:1:1.0
ds4: DSPARK_SHADOW bank=2 guard=2.000 alpha=0.500 minev=2 budget=1.0 yewma=3.312 debt=0.000 quench_step=1 post_quench_yield=4.000
ds4: CONT_MTP_ACCEPT(DSpark) D=3 steps=4 emit=10 drafts=8 hits=6 accept=75.0% tok/step=2.5
""",
        )
        assert len(recovery_parsed.aggregates) == 1
        assert recovery.shadow is not None
        assert not _shadow_mismatches(recovery)
        policy = replay_policy(recovery, params)
        economics = economic_model(recovery, policy, cost)
        summary = summarize_replay((recovery,), params, cost)
        assert policy.quench_step == 1
        _assert_close(policy.yewma, 3.3125)
        _assert_close(policy.debt, 0.0)
        _assert_close(policy.post_quench_yield, 4.0)
        _assert_close(economics.speedup_spec, 1.25)
        _assert_close(economics.speedup_quenched, 5.0 / 6.0)
        _assert_close(summary.oracle_retention, 2.0 / 3.0)
        assert summary.false_quench_count == 1
        assert summary.recovery_count == 1
        assert economics.tokens_after_quench == 8
        print("selftest recovery-after-slump: PASS")

        eviction, _ = _selftest_request(
            "eviction-mid-accept",
            """
ds4: DSPARK_TRACE bank=3 pos0=40 D=4 steps=4 emit=9 drafts=7 hits=5 trace=3:3:4:2:1:1.0 1:0:0:3:0:2.0 3:3:4:1:1:3.0 2:1:4:1:1:4.0
""",
        )
        policy = replay_policy(eviction, params)
        economics = economic_model(eviction, policy, cost)
        assert policy.quench_step == -1
        assert policy.spec_steps == 3
        _assert_close(economics.always_spec_time, 7.0)
        _assert_close(economics.speedup_spec, 9.0 / 7.0)
        _assert_close(economics.speedup_quenched, 9.0 / 7.0)
        print("selftest eviction-mid-accept: PASS")
    except AssertionError as exc:
        print(f"SELFTEST: FAIL {exc}", file=sys.stderr)
        return 1

    print("SELFTEST: PASS (4 fixtures)")
    return 0


def _finite_cli_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("must be finite")
    return value


def _positive_cli_float(text: str) -> float:
    value = _finite_cli_float(text)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return value


def _unsigned_cli_int(text: str) -> int:
    if not _UINT_RE.fullmatch(text):
        raise argparse.ArgumentTypeError("must be an unsigned integer")
    return int(text)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and replay DSpark terminal-yield-quench traces."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="cross-check traces, aggregate telemetry, and shadows"
    )
    validate_parser.add_argument("logs", nargs="+", metavar="LOG")

    replay_parser = subparsers.add_parser(
        "replay", help="replay one policy or the built-in 36-point grid"
    )
    replay_parser.add_argument("logs", nargs="+", metavar="LOG")
    replay_parser.add_argument("--grid", action="store_true", help="sweep the full grid")
    replay_parser.add_argument("--guard", type=_finite_cli_float)
    replay_parser.add_argument("--alpha", type=_finite_cli_float)
    replay_parser.add_argument("--minev", type=_unsigned_cli_int)
    replay_parser.add_argument("--budget", type=_finite_cli_float)
    replay_parser.add_argument(
        "--credit-cap",
        type=_finite_cli_float,
        help="debt floor (-credit_cap); 0 = engine shadow, large = cumulative regret",
    )
    replay_parser.add_argument(
        "--cost",
        type=_positive_cli_float,
        default=DEFAULT_COST,
        help=f"spec-step cost in plain steps (default: {DEFAULT_COST})",
    )
    replay_parser.add_argument(
        "--csv", metavar="OUT", help="write all ranked grid rows as CSV"
    )

    inspect_parser = subparsers.add_parser(
        "inspect", help="print one line per traced request"
    )
    inspect_parser.add_argument("log", metavar="LOG")

    subparsers.add_parser("selftest", help="run four synthetic hand-checked fixtures")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.command == "validate":
        return command_validate(args.logs)
    if args.command == "inspect":
        return command_inspect(args.log)
    if args.command == "selftest":
        return run_selftest()
    if args.command == "replay":
        explicit_policy = any(
            value is not None
            for value in (args.guard, args.alpha, args.minev, args.budget, args.credit_cap)
        )
        if args.grid and explicit_policy:
            parser.error(
                "--grid cannot be combined with --guard, --alpha, --minev, "
                "--budget, or --credit-cap"
            )
        if args.csv is not None and not args.grid:
            parser.error("--csv requires --grid")
        params = PolicyParams(
            guard=DEFAULT_GUARD if args.guard is None else args.guard,
            alpha=DEFAULT_ALPHA if args.alpha is None else args.alpha,
            minev=DEFAULT_MINEV if args.minev is None else args.minev,
            budget=DEFAULT_BUDGET if args.budget is None else args.budget,
            credit_cap=0.0 if args.credit_cap is None else args.credit_cap,
        )
        return command_replay(args.logs, args.grid, params, args.cost, args.csv)

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    sys.exit(main())
