#!/usr/bin/env python3
"""Regression check: moving a proof runner must not change plan inputs."""

import copy
import importlib.util
import sys
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "ds4_proof", Path(__file__).with_name("ds4_proof.py")
)
assert spec and spec.loader
proof = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = proof
spec.loader.exec_module(proof)

oracle_plan = {
    "schema": "ds4-proof-expanded-plan-v1",
    "scenario": "cuda-opp-c-full",
    "bin_path": "/frozen/oracle/ds4",
    "tokens": 512,
    "profiles": [{"name": "cuda-default", "env": {}}],
}
candidate_plan = copy.deepcopy(oracle_plan)
candidate_plan["bin_path"] = "/tmp/ds4-proof/candidate"

assert proof.expanded_plan_inputs_sha256(oracle_plan) == proof.expanded_plan_inputs_sha256(
    candidate_plan
)
changed_plan = copy.deepcopy(candidate_plan)
changed_plan["tokens"] = 513
assert proof.expanded_plan_inputs_sha256(oracle_plan) != proof.expanded_plan_inputs_sha256(
    changed_plan
)

expected = {
    "expanded_plan_sha256": proof.expanded_plan_sha256(oracle_plan),
    "expanded_plan_inputs_sha256": proof.expanded_plan_inputs_sha256(oracle_plan),
    "cells": [],
}
passed, reasons = proof.validate_against_expected(expected, candidate_plan, {})
assert passed, reasons

legacy = {"expanded_plan_sha256": proof.expanded_plan_sha256(oracle_plan), "cells": []}
passed, reasons = proof.validate_against_expected(legacy, candidate_plan, {})
assert not passed and "SHA256 mismatch" in reasons[0]

print("ds4 proof plan hash smoke: PASS")
