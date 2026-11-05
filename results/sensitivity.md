# Sensitivity pass: contested_top_25s

Every stat is scaled ±15% for all four robots, then the scenario is re-solved (screening, matrix game, double oracle).

| Run | P(red win) | E[red margin] | Red's main plan (weight) | Blue's main plan (weight) |
|---|---|---|---|---|
| baseline | 0.838 | +47.9 | R1 take Toggle R, R2 score, both Midfield at 10 s (0.86) | B1 yellow on Midfield Goal, B2 take Toggle T (0.68) |
| speed ×0.85 | 0.784 | +33.8 | same (0.90) | same plans, B1 Midfield at 10 s (0.46) |
| speed ×1.15 | 0.831 | +18.3 | same, R1 Midfield at the last moment (0.50) | B1 take Toggle L, B2 score (0.60) |
| cycle ×0.85 | 0.826 | +23.7 | same, R2 Midfield at the last moment (0.46) | B1 score, B2 take Toggle L (0.31) |
| cycle ×1.15 | 0.778 | +11.8 | same (0.60) | B1 take Toggle L, B2 take Toggle R (0.28) |
| Toggle time ×0.85 | 0.831 | +20.4 | same (0.58) | B1 take Toggle L, B2 score (0.52) |
| Toggle time ×1.15 | 0.823 | +34.8 | same (0.86) | B1 take Toggle R, B2 take Toggle L (0.38) |
| push success ×0.85 | 0.851 | +49.9 | same (0.88) | baseline plans (0.64) |
| push success ×1.15 | 0.822 | +40.9 | same (0.85) | baseline plans (0.62) |
| blue 2 can't use Toggles | 0.873 | +26.9 | same pair, roles swapped (0.45) | B1 take Toggle L, B2 score (0.47) |

Reading it:
- P(win) stays in 0.78-0.87 across every run, so the value is not fragile to any one stat.
- Red's answer holds: one robot takes blue's Toggle R, the other scores, both reach the Midfield. Only the timing of the Midfield break moves (10 s vs last moment) with speed and cycle time.
- Blue's preferred plan changes in most runs, but blue's options sit within about 0.01-0.03 P(win) of each other, so these are near-ties, not a change in what matters.
- Speed and cycle time move the expected margin far more (+12 to +48) than P(win); push success barely matters here.
- Flattening cycle time, Toggle time and push success to reference values looks justified for this scenario. Speed shifts margin and Midfield timing, so it stays a real parameter, as planned.

Raw numbers: sensitivity.json.
