#!/usr/bin/env python3
"""Per-market volatility forecaster for the FAA outage attribution family.

For every market in a case we estimate how jumpy that market itself has
been lately, then draw a five-quantile fan around its checkpoint price
whose half-width grows like a (slightly damped) random walk.  The fans
are tied together by a soft family-coherence pull: the four attribution
markets are separate questions, so their prices are NOT forced to sum
to any fixed total.  Instead we watch the family's own price sum over
the recent past; when that sum has been hugging a steady level, each
market's centre is eased toward the level the family keeps returning
to, and when the sum has been wandering, the pull fades away on its
own.  Everything is a deterministic function of the case file.
"""

import json
import math
import os

# --- tuning constants (fitted on the six NVIDIA practice cases) -----------
SHORT_SPAN = 120        # minutes in the fast volatility window
LONG_SPAN = 720         # minutes in the slow volatility window
SHORT_WEIGHT = 0.95     # blend weight on the fast window
SIGMA_FLOOR = 0.0012    # clamp range for a history-derived sigma
SIGMA_CEIL = 0.006
THIN_LIMIT = 120        # fewer prints than this counts as thin history
THIN_SIGMA = 0.01       # fallback sigma for thin history (never clamped)
FAN_SHRINK = 0.68       # global width multiplier on the fan
GROWTH_POWER = 0.45     # fan half-width grows as minutes_ahead ** this
SUM_SPAN = 240          # common minutes used to judge the family sum
PULL_STRENGTH = 1.0     # coherence pull at full measured stability
PULL_SCALE = 0.04       # family-sum wobble at which the pull halves
PULL_RAMP = 120         # minutes over which the pull phases in
# Quantile shape: a unit-variance logistic, slightly heavier in the tails
# than a normal, which suits these step-and-jump price series.
LEVEL_SHAPE = (-1.6232, -0.6057, 0.0, 0.6057, 1.6232)


def window_rms(diffs, span):
    """Root-mean-square of the last `span` one-minute price changes."""
    seg = diffs[-span:] if len(diffs) > span else diffs
    if not seg:
        return 0.0
    return math.sqrt(sum(d * d for d in seg) / len(seg))


def market_sigma(prices):
    """Per-minute volatility of one market, from its own recent history.

    A market with too little history to judge gets a fixed, deliberately
    generous sigma; any other market blends a fast and a slow window of
    its realised minute-to-minute moves and is clamped to a sane band.
    """
    if len(prices) < THIN_LIMIT:
        return THIN_SIGMA
    diffs = [b - a for a, b in zip(prices, prices[1:])]
    fast = window_rms(diffs, SHORT_SPAN)
    slow = window_rms(diffs, LONG_SPAN)
    blended = SHORT_WEIGHT * fast + (1.0 - SHORT_WEIGHT) * slow
    return min(max(blended, SIGMA_FLOOR), SIGMA_CEIL)


def family_pull(series, sigmas, lasts):
    """How far each market's centre should ease toward family coherence.

    The family's price sum is tracked over the minutes every market
    shares.  Its average over that stretch is the level the family has
    been living at; the scatter around the average says how much that
    level can be trusted.  A trustworthy level earns a pull of the
    centres (split in proportion to each market's sigma) that would
    close the gap between the current sum and the level; an untrusty
    one earns almost none.  Returns one signed shift per market.
    """
    shared = min(len(s) for s in series)
    span = min(shared, SUM_SPAN)
    if span < 4:
        return [0.0] * len(series)
    sums = [0.0] * span
    for s in series:
        tail = s[-span:]
        for j, px in enumerate(tail):
            sums[j] += px
    level = sum(sums) / span
    scatter = math.sqrt(sum((s - level) ** 2 for s in sums) / span)
    trust = 1.0 / (1.0 + (scatter / PULL_SCALE) ** 2)
    gap = level - sum(lasts)
    total_sigma = sum(sigmas)
    shifts = []
    for sg in sigmas:
        share = sg / total_sigma if total_sigma > 0 else 1.0 / len(series)
        shifts.append(PULL_STRENGTH * trust * gap * share)
    return shifts


def forecast_case(case):
    """Build the {"forecasts": [...]} document for one parsed case."""
    markets = case["markets"]
    horizon = case["horizon_minutes"]
    # A market is only listed once it has a price, so its series is never
    # empty in a real case; the [0.5] stand-in is pure crash insurance,
    # since one crashed case would sink the whole entry.
    series = [m["prices"] if m["prices"] else [0.5] for m in markets]
    lasts = [s[-1] for s in series]
    sigmas = [market_sigma(s) for s in series]
    shifts = family_pull(series, sigmas, lasts)
    growth = [h ** GROWTH_POWER for h in range(1, horizon + 1)]

    rows_out = []
    for idx, m in enumerate(markets):
        half = FAN_SHRINK * sigmas[idx]
        rows = []
        for h in range(1, horizon + 1):
            phase = min(1.0, h / PULL_RAMP)
            centre = lasts[idx] + shifts[idx] * phase
            width = half * growth[h - 1]
            row = []
            prev = 0.0
            for z in LEVEL_SHAPE:
                q = centre + z * width
                q = min(max(q, 0.0), 1.0)
                q = max(q, prev)          # keep the five levels ordered
                prev = q
                row.append(round(q, 6))
            rows.append(row)
        rows_out.append({"id": m["id"], "q": rows})
    return {"forecasts": rows_out}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        doc = forecast_case(case)
        stem = name[:-3]  # strip ".in"
        target = os.path.join(out_dir, stem + ".out")
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, separators=(",", ":"))


if __name__ == "__main__":
    main()
