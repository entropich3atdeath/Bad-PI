# TestSpec validation (v1)

This document describes Bad PI's executable validation protocol layer.

## Why this exists

A hypothesis can now be either:

- exploration hypothesis: narrative + broad evidence gathering
- validation hypothesis: narrative + strict executable `test_spec`

Validation hypotheses are analyzed deterministically and update belief **once per completed test**.

---

## Core model

```json
{
  "statement": "Depth and LR interact",
  "phase": "validation",
  "test_spec": {
    "type": "interaction_grid",
    "variables": ["DEPTH", "learning_rate"],
    "grid": {
      "DEPTH": [8, 12, 16],
      "learning_rate": [0.0001, 0.001, 0.01]
    },
    "min_runs_per_cell": 3,
    "decision_rule": {"threshold": 0.01}
  }
}
```

For validation hypotheses:

1. Engine validates `test_spec`
2. Scheduler fills missing arms/cells in queue
3. Analyzer computes deterministic statistic
4. Engine applies one win/loss vote when complete

---

## Supported test types (v1)

## `single_factor_effect`

Use when you want to verify a main effect of one variable.

Required fields:

- `variable`: string
- `values`: list with >=2 arms
- `min_runs_per_cell`: int >=1
- `decision_rule.threshold`: numeric

Example:

```json
{
  "type": "single_factor_effect",
  "variable": "DEPTH",
  "values": [8, 12],
  "min_runs_per_cell": 3,
  "decision_rule": {"threshold": 0.01}
}
```

Decision logic:

- compute mean `delta_bpb` for each arm
- `effect_size = worst_mean - best_mean`
- win if `effect_size >= threshold`

---

## `interaction_grid`

Use when you want to test non-additive interaction between two variables.

Required fields:

- `variables`: exactly 2 variable names
- `grid`: value lists for both variables (>=2 each)
- `min_runs_per_cell`: int >=1
- `decision_rule.threshold`: numeric

Example:

```json
{
  "type": "interaction_grid",
  "variables": ["DEPTH", "learning_rate"],
  "grid": {
    "DEPTH": [8, 12, 16],
    "learning_rate": [0.0001, 0.001, 0.01]
  },
  "min_runs_per_cell": 2,
  "decision_rule": {"threshold": 0.01}
}
```

Decision logic (deterministic):

- compute mean per cell
- compute additive expectation from marginals
- interaction strength = mean absolute deviation from additive expectation
- win if interaction strength >= threshold

---

## How `single_factor_effect` and `interaction_grid` work together

Typical pattern:

1. Run `single_factor_effect` for each variable separately.
   - confirms there is a main effect worth modeling.
2. Run `interaction_grid` to test whether combinations add extra signal.
   - confirms or rejects non-additive interaction.

This avoids jumping to interaction claims before main effects are verified.

---

## Scheduling behavior

For validation hypotheses, Bad PI enqueues targeted configs to fill the protocol:

- missing single-factor arms
- missing interaction grid cells

Each required arm/cell is repeated until `min_runs_per_cell` is met.

---

## Belief update semantics

Validation hypotheses do **not** update on each run.

Instead:

- all runs contribute to test completion and deterministic statistic
- when test completes, engine applies exactly one vote:
  - test win -> `alpha += 1`
  - test loss -> `beta += 1`

This reduces run-level noise and aligns belief updates with completed experimental protocols.

---

## Notes and current limits

- `stop_condition` fields are schema-validated in v1 but not yet used for confidence-based early completion.
- Current v1 focuses on minimal-risk deterministic validation primitives.
- Additional test types can be added later (`threshold_test`, `ranking_consistency`, `sensitivity_curve`).
