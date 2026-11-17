"""Hierarchical lognormal shrinkage quantiles — GTA VI Twitch viewers, hours 2-24.

Forecasts the Twitch category viewers of Grand Theft Auto VI for each hour of its
release day (2026-11-19 UTC) after the first hour, as five quantiles per hour.

Method (parametric, standard library only, deterministic):
  * For every past big launch in the case's `comparables`, take its release day and
    form the log-ratio of each hour's viewers to that day's hour-1 viewers:
    r[h] = ln(viewers[h] / viewers[1]).
  * Only launches whose hour-1 print reached 20,000 viewers enter the fit. A
    category with a few hundred viewers in its first hour is not a launch-scale
    audience, and its day curve follows a different regime: small categories
    typically grow through the day as streamers discover them, while big
    launches peak early and decay. Mixing the two regimes in one pool drags the
    centre the wrong way for a launch the size of GTA VI.
  * For each hour h = 2..24, the log-ratios across the fitted launches are
    modelled as normal (so the viewers ratio is lognormal): per-hour sample
    mean m[h] and variance v[h].
  * The per-hour mean is shrunk toward the pooled mean M (the mean of every
    log-ratio in the fit) with precision weighting: the weight on m[h] is
    tau^2 / (tau^2 + v[h]/n[h]), where tau^2 is the between-hour variance of
    the true means, estimated by the method of moments. Hours with few
    comparables shrink more; hours with many keep their own mean.
  * The spread is pooled: sigma[h]^2 blends the per-hour variance with the
    pooled variance V (prior weight 10), floored at 0.15 so no hour collapses
    to zero spread.
  * Quantiles: q_tau[h] = p1 * exp(mu[h] + z_tau * sigma[h]), p1 the case's
    hour-1 print and z_tau the standard normal quantile at the case's levels.
  * If a case carries no usable comparable, the forecast falls back to
    persistence: every hour's median is the hour-1 print, with a spread that
    widens like a random walk (0.15 * sqrt(h - 1)).

Validation (leave-one-out over the 37 comparable launches in the kit, each
held out in turn and fitted on the other 36; pinball loss on the log scale):
overall 0.8530, big hour-1 (>= 100,000 viewers) 0.2391, huge hour-1
(>= 250,000) 0.2158 — against the house baseline's 0.9297 / 0.2611 / 0.2255.

Run: python3 main.py — reads $MAKEFUN_INPUT_DIR (else /input) *.in files and
writes $MAKEFUN_OUTPUT_DIR (else /output) <id>.out per case. No network, no
randomness: two runs on the same case write the same bytes.
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Launch-scale floor for the fitting population (hour-1 viewers). See docstring.
MIN_HOUR1 = 20000.0
# Prior weight blending each hour's variance toward the pooled variance.
VAR_PRIOR_WEIGHT = 10.0
# Lowest log-spread any hour may take.
SIGMA_FLOOR = 0.15
# The contest caps every quantile at 1e9 viewers.
MAX_VIEWERS = 1_000_000_000.0
# Fallback spread (persistence), in log units per square-root hour.
FALLBACK_SIGMA = 0.15


def norm_ppf(p):
    """Standard normal quantile at p (Acklam's rational approximation)."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285984469534e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return num / den
    if p <= p_high:
        q = p - 0.5
        r = q * q
        num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        return num / den
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
    den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    return -num / den


def _parse_time(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")


def release_day_hours(comp):
    """{hour 1..24: viewers} for one comparable's release day (its day_start)."""
    try:
        day0 = _parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    hours = {}
    for row in comp.get("hourly") or []:
        try:
            moment = _parse_time(row["from"])
            viewers = row.get("viewers")
        except (KeyError, TypeError, ValueError):
            continue
        slot = int((moment - day0).total_seconds() // 3600) + 1
        if 1 <= slot <= 24 and viewers:
            hours[slot] = float(viewers)
    return hours


def fit_lognormal(day_hours_list):
    """Fit the hierarchical lognormal on a list of release-day hour dicts.

    Returns (per_hour, pooled_mean, pooled_var, tau2) or None when nothing in
    the list is usable. per_hour maps h -> (mean, variance-or-None, n).
    """
    ratios = {h: [] for h in range(2, 25)}
    for hours in day_hours_list:
        first = hours.get(1)
        if not first or first < MIN_HOUR1:
            continue
        base = math.log(first)
        for h in range(2, 25):
            value = hours.get(h)
            if value and value > 0:
                ratios[h].append(math.log(value) - base)
    per_hour = {}
    pooled = []
    for h in range(2, 25):
        vals = ratios[h]
        count = len(vals)
        if count == 0:
            per_hour[h] = (None, None, 0)
            continue
        mean = sum(vals) / count
        var = None
        if count > 1:
            var = sum((x - mean) ** 2 for x in vals) / (count - 1)
        per_hour[h] = (mean, var, count)
        pooled.extend(vals)
    if not pooled:
        return None
    pooled_mean = sum(pooled) / len(pooled)
    num = den = 0.0
    for h in range(2, 25):
        _mean, var, count = per_hour[h]
        if var is not None and count > 1:
            num += (count - 1) * var
            den += (count - 1)
    pooled_var = num / den if den > 0 else 0.25
    means = [per_hour[h][0] for h in range(2, 25) if per_hour[h][0] is not None]
    sampling = [per_hour[h][1] / per_hour[h][2] for h in range(2, 25)
                if per_hour[h][1] is not None and per_hour[h][2] > 0]
    grand = sum(means) / len(means)
    var_means = (sum((x - grand) ** 2 for x in means) / (len(means) - 1)
                 if len(means) > 1 else 0.0)
    mean_sampling = sum(sampling) / len(sampling) if sampling else 0.0
    tau2 = max(0.0, var_means - mean_sampling)
    return per_hour, pooled_mean, pooled_var, tau2


def _quantile_row(p1, mu, sigma, z_scores):
    row = []
    prev = 0.0
    for z in z_scores:
        q = p1 * math.exp(mu + z * sigma)
        q = min(max(q, 0.01), MAX_VIEWERS)
        if q < prev:
            q = prev
        row.append(q)
        prev = q
    return row


def forecast_hours(p1, fit, levels, hours):
    """Quantile rows for the requested hours from a fitted model."""
    per_hour, pooled_mean, pooled_var, tau2 = fit
    z_scores = [norm_ppf(t) for t in levels]
    rows = []
    for h in hours:
        mean, var, count = per_hour.get(h, (None, None, 0))
        if mean is None:
            mu, sig2 = pooled_mean, pooled_var
        else:
            spread = (var if var is not None else pooled_var) / count
            weight = tau2 / (tau2 + spread) if tau2 + spread > 0 else 0.0
            mu = weight * mean + (1.0 - weight) * pooled_mean
            if var is not None and count > 1:
                sig2 = ((count - 1) * var + VAR_PRIOR_WEIGHT * pooled_var) / (
                    (count - 1) + VAR_PRIOR_WEIGHT)
            else:
                sig2 = pooled_var
        sigma = max(math.sqrt(max(sig2, 0.0)), SIGMA_FLOOR)
        rows.append({"hour": h, "q": _quantile_row(p1, mu, sigma, z_scores)})
    return rows


def fallback_hours(p1, levels, hours):
    """Persistence fallback: median stays at the hour-1 print, spread widens."""
    z_scores = [norm_ppf(t) for t in levels]
    rows = []
    for h in hours:
        sigma = FALLBACK_SIGMA * math.sqrt(h - 1)
        rows.append({"hour": h, "q": _quantile_row(p1, 0.0, sigma, z_scores)})
    return rows


def forecast_case(case):
    known = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if known is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(known["viewers"]))
    levels = [float(t) for t in
              case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]]
    hours = [int(row["hour"]) for row in case.get("forecast") or []]
    days = [release_day_hours(c) for c in case.get("comparables") or []]
    fit = fit_lognormal(days)
    if fit is None:
        rows = fallback_hours(p1, levels, hours)
    else:
        rows = forecast_hours(p1, fit, levels, hours)
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        target = OUTPUT_DIR / (path.stem + ".out")
        target.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                          encoding="utf-8")


if __name__ == "__main__":
    main()
