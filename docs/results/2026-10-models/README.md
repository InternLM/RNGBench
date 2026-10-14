# October 2026 Matching Pairs model evaluations

All 45 single-player games and 24 duel games completed and passed native offline
replay audits. This release records the final corrected Kimi cohort, the verified
DeepSeek/GLM single-player cohorts, and both duel batches. Superseded Kimi runs,
short diagnostics and video samples are excluded from the scores.

## Single-player

10×10 image/noise, 50 pairs, up to 250 model responses per game. Each of three
tries runs all board seeds `0, 1, 2, 3, 4`. Score is `100 × matched pairs / 50`;
average the five seeds within each try and then the three try means. PF and IA
use all recorded model responses as the denominator. Resp./Score is total
responses divided by total matched pairs.

| Model | Thinking setting | PF%↓ | IA%↓ | Resp./Score↓ | Score%↑ |
|---|---|---:|---:|---:|---:|
| Kimi-K3 | max; complete own thinking history | 0.00 | 0.08 | 6.50 | 76.93 |
| DeepSeek-V4.1-Flash | provider default; level not pinned | 0.13 | 0.00 | 9.38 | 53.33 |
| GLM-5.3-Flash | provider default; level not pinned | 1.44 | 0.51 | 9.52 | 52.53 |

| Model | Try 1 | Try 2 | Try 3 | Mean | Try SD (percentage points) |
|---|---:|---:|---:|---:|---:|
| Kimi-K3 | 82.00 | 80.40 | 68.40 | 76.93 | 7.43 |
| DeepSeek-V4.1-Flash | 48.40 | 54.40 | 57.20 | 53.33 | 4.50 |
| GLM-5.3-Flash | 49.20 | 51.20 | 57.20 | 52.53 | 4.16 |

## Duels

8×10 image/poker, 40 pairs, up to 200 responses shared by both players. Seeds
`0,1` form the initial 12 games; seeds `2,3` form the additional 12. Every matchup
has both starting players for each seed. Each model faces the other two models
in 16 games. A game reaching the declared reply budget is complete even if not
all pairs have been matched. Wins compare final matched-pair counts.

| Matchup | First model wins | Ties | Second model wins | Games |
|---|---:|---:|---:|---:|
| DeepSeek-V4.1-Flash vs. GLM-5.3-Flash | 4 | 0 | 4 | 8 |
| DeepSeek-V4.1-Flash vs. Kimi-K3 | 0 | 1 | 7 | 8 |
| GLM-5.3-Flash vs. Kimi-K3 | 2 | 1 | 5 | 8 |

## Thinking and model configuration

All requests use `temperature=1.0`, `top_p=0.95`, `max_tokens=32768`.
The full game history and Thought/Action prompt are retained, with two action
retries. `cot_enabled=True` controls the prompt format; it is not a provider
thinking switch.

- Kimi-K3: `reasoning_effort=max`; image long edge 448. Its own complete returned
  assistant messages, including reasoning, are passed back in order. Opponent
  private reasoning is not passed to it.
- DeepSeek-V4.1-Flash: provider default, without an explicit thinking/effort
  override; image long edge 512. Previous final assistant content is retained.
  Actual returned reasoning was checked; no evidence of a client-side disabled
  thinking setting or dropped returned reasoning was found.
- GLM-5.3-Flash: provider default, without an explicit thinking/effort override;
  image long edge 512. Previous final assistant content is retained. Some raw
  replies have no separate reasoning field. This does not establish that server
  thinking was disabled. Earlier default/explicit diagnostic requests did not
  establish a hidden effort level.

| Mode | Model | Replies with nonempty reasoning / replies | `length` | Empty final | Parse failures | Invalid actions |
|---|---|---:|---:|---:|---:|---:|
| duel | DeepSeek-V4.1-Flash | 1521 / 1523 | 0 | 1 | 1 | 6 |
| duel | GLM-5.3-Flash | 1368 / 1624 | 49 | 49 | 74 | 9 |
| duel | Kimi-K3 | 1653 / 1653 | 0 | 0 | 6 | 0 |
| single | DeepSeek-V4.1-Flash | 3743 / 3750 | 0 | 2 | 5 | 0 |
| single | GLM-5.3-Flash | 1810 / 3750 | 45 | 45 | 54 | 19 |
| single | Kimi-K3 | 3750 / 3750 | 0 | 0 | 0 | 3 |

Reasoning counts refer to the separate raw thinking/reasoning field, not the
`Thought:` text in the final action. Original JSON/SSE replies were independently
decoded and compared with the accepted API ledger and native turns. Scores,
seeds, starting orders and recorded request options were rechecked. Audit success
establishes that the recorded configuration was executed and scored consistently;
it does not mean every model action was correct or establish a hidden server level.

Valid empty, truncated or malformed model outputs count under native rules.
Unresolved transport errors are not scored as model answers; successful network
retries retain the same request/history boundary. No failed game remains in the
selected cohort. None of these checks changed model outputs or scores.

## Scope and comparability

These are measurements under the configurations above. The complete historical
single-player seed/repetition manifest and every historical model sampling setting
have not been recovered. These three models were not evaluated on 3D Maze or
against historical leaderboard opponents in this cohort. Their new duel results
therefore do not update the historical Elo pool.

The authorized primary deployments were used where available, with JoyRouter
fallbacks after service errors. Model identity uses the deployment mapping supplied
for this experiment. Equality of weights across all routing aliases is not
independently established. Route changes and Kimi's corrected history protocol
prevent attributing the entire difference from superseded runs to a single cause.

## Data and verification

- [45 single-player rows](single.csv)
- [24 duel results](duel.csv)
- [93 player rows, request options and input hashes](results.json)
- [Publication checks](verification.json)

Raw trajectories and provider responses remain in the persistent experiment
archive. This portable release includes per-game score and audit SHA-256 values,
without deployment credentials or machine-specific paths.
