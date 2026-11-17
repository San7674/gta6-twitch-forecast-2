# Hierarchical lognormal shrinkage quantiles

[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/San7674/gta6-twitch-forecast-2/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/San7674/gta6-twitch-forecast-2/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Forecasts GTA VI's launch-day Twitch viewers hour by hour (hours 2–24, five
quantiles each) from the first hour's print, for the
[make.fun bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01).

## Method

A parametric hierarchical lognormal model over past big launches:

- For every comparable launch, form the log-ratio of each hour's viewers to
  that day's hour-1 viewers, `r[h] = ln(viewers[h] / viewers[1])`.
- Only launches whose hour-1 print reached 20,000 viewers enter the fit —
  small categories follow a different day curve than launch-scale ones.
- For each hour, the log-ratios are modelled as normal (lognormal viewers),
  with a per-hour sample mean and variance.
- Each hour's mean is shrunk toward the pooled mean with precision weighting:
  hours with few comparables shrink more.
- The spread pools each hour's variance with the pooled variance (prior
  weight 10) and is floored at 0.15, so no hour collapses to zero spread.
- Quantiles are `p1 * exp(mu[h] + z_tau * sigma[h])`, anchored on the observed
  hour-1 print.

Python 3.13, standard library only, deterministic — no network, no randomness.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` case in the input directory and writes one `<id>.out` per
case with 23 rows (hours 2–24), five quantiles each, at the levels
0.05, 0.25, 0.5, 0.75, 0.95.

## Leave-one-out check

Scored leave-one-out over the 37 comparable launches in the kit (each launch
held out in turn, fitted on the other 36), pinball loss on the log scale:

| | this entry | house baseline |
|---|---:|---:|
| overall | 0.8530 | 0.9297 |
| big hour-1 (≥100,000 viewers) | 0.2391 | 0.2611 |
| huge hour-1 (≥250,000 viewers) | 0.2158 | 0.2255 |

The entry beats the house baseline in every size regime.
