# Project Plan

Read `AGENTS.md` first.

## How to use this
Work **one phase at a time**. Each phase gives a goal, what it must produce, and how to check it. The **how** is up to whoever takes the phase: look at the data, think through options, and pick the approach you can best explain. Before starting, briefly propose your approach to the team. When you finish, leave the outputs listed so the next phase can pick up from there.

Several phases can run at the same time (see the dependency notes).

**Dependencies:** C, F, G can run in parallel. E's final model is published, so F and G should rerun the steps that read it. I needs C-H. J is last.

**Status:** A, B, D, E, H done. Open now: C, F, G. Then I, J. E is frozen: 2025 has been tested, so nothing in sections 3-6 may change in a way that alters the model or its predictions.

## Shared outputs and how to use them
**Getting the data.** `data/` is not in git. From the repo root run `bash scripts/verify.sh` once (a few minutes the first time, it downloads and caches everything). Rerun it after pulling changes to earlier sections.

**Where to work.** Copy `notebooks/starter.ipynb` to `notebooks/<phase>_<topic>.ipynb` (e.g. `E_regression.ipynb`). It loads the data and has the shared evaluation code. Finished cells move into your section of `fantasy_football_rankings.ipynb`. Follow "Avoiding conflicts" in `AGENTS.md`.

**Files in `data/processed/`** (all keyed by `gsis_id, season, week` unless noted):

| File | What it is | Use it for | Never |
|---|---|---|---|
| `features.parquet` | 22,145 player-weeks, 2020-2025: the 48 features, `fantasy_points_ppr` (target), `position`, `team`, `kickoff` | All modeling (E, F, G, H) | Use any column not listed in `feature_families.json` as an input |
| `feature_families.json` | Feature names grouped into 7 families | Choosing inputs, family comparisons | |
| `player_weeks.parquet` | The same rows with that week's full box score and context | Looking up names, explaining results | Use as features: it holds the game's own stats (leakage) |
| `player_games.parquet` | Every game played 2019-2025, all players | Building new rolling features (always `shift(1)` within player first) | Use a row's own stats to predict that row |
| `ecr_weekly.parquet` | Expert ranks: `ecr`, `ecr_pos_rank` (1 = best at the position), `snapshot_date`, `played_before_snapshot` | Comparing against experts (F, H, I) | Use as a feature |

**Rules that apply to every phase:**
- Tune and select only on `season` in 2020-2024. 2025 rows are touched once, in Phase H.
- Use `cv_folds()`, `evaluate()`, and `cross_validate()` from the starter (copied exactly from main notebook section 5.6). They give expanding-window folds (validate 2022, 2023, 2024) and RMSE, MAE, weekly Spearman within position, and top-12 hit rate.
- When comparing to ECR: inner-join on the key, drop `played_before_snapshot == True`, and rank within `season, week, position`. ECR only covers weeks with a snapshot (none in week 18, 2020 starts week 6, 2024 starts week 4).
- Starting points from D: Ridge on all features gets RMSE 6.81, Spearman 0.515, top-12 hit rate 0.490. The recency half-life is 8 games. The luck family adds almost nothing.
- Final model from E: voting ensemble (Lasso + gradient boosting + MLP), CV RMSE 6.77, Spearman 0.525, top-12 hit rate 0.489.

**Files each phase saves** (same keys, so they join to everything above):

| Phase | File | Columns |
|---|---|---|
| E | `preds_regression.parquet` | key, `position`, `pred`, `model`. Out-of-fold predictions for 2022-2024. Written by the last cell of main notebook section 6. Holds the final model (voting ensemble, refit every week). |
| E | `regression_choice.json` | `status` (`baseline` or `final`), `model`, `params`, `features`, `training_window`, `sample_weight_half_life_seasons`, `refit` |
| F | `tier_results.parquet` | key, `position`, `tier_actual`, `tier_from_regression`, `tier_classifier`, `tier_ecr` |
| G | `usage_clusters.parquet` | key, `cluster`, `switched_cluster` |
| H | `preds_test.parquet` | key, `position`, `pred`, `ecr_pos_rank`, `trained_through` (last season and week in the training data, e.g. `2025-W03`). `ecr_pos_rank` is filled only where the expert rank was made before kickoff. |


## Recency principle
The most useful information for next week is the most recent: current role, current team, current health, recent form. Older games still help, mostly to steady noisy recent numbers. So every phase treats recency as something to **measure and tune**, not assume:
- **Explore (C)** measures how fast past games lose predictive value.
- **Features (D)** give recent games more weight and flag changes in a player's situation.
- **Modeling (E)** tunes how much older seasons count in training.
- **Final test (H)** refits the model each week of 2025 on every game before that week (walk-forward), the way a real forecaster would.

Each recency choice is compared against a plain version (equal weights, no refit). If it does not help in CV, it goes in the "What we tried and didn't keep" table.

## Rubric
| Item | Pts | Covered in |
|---|---|---|
| Data cleaning | 10 | Phase B |
| Exploration | 10 | Phase C |
| Feature engineering | 20 | Phase D |
| Modeling | 40 | Phases E, F, G |
| Code and notebook quality | 10 | All phases |
| Results | 10 | Phases H, I |

## Course methods we can use
Linear regression, Ridge/Lasso, polynomial regression, decision trees, KNN, Naive Bayes, SVMs, neural nets (MLP), ensembles (bagging, random forest, boosting, voting, stacking), cross-validation, hyperparameter tuning, PCA, K-Means, hierarchical clustering, DBSCAN, cluster validation, anomaly detection, and class-imbalance handling.

## Data
All public, loaded with `nflreadpy`:
- weekly player stats
- snap counts
- weekly rosters (team and roster status each week)
- schedules with Vegas lines
- expected fantasy points (ffopportunity)
- FantasyPros ECR archive (used only to compare against)
- player ID crosswalk
- injury reports (pre-game status), used for situation features
- not used: depth charts (no 2025 data), play-by-play

Seasons: 2019 for warm-up lags only, 2020-2024 for training and CV, 2025 for the test.

---

## Phase A: Setup (done)
`.venv`, pinned `requirements.txt`, the notebook skeleton, `scripts/verify.sh`, `data/` gitignored, and `notebooks/starter.ipynb` for phase work.

## Phase B: Load and clean (done)
Notebook section 3. One row per pool player per week, 2020-2025 (22,145 rows).
- IDs mapped to `gsis_id`, team codes fixed, ECR snapshots matched to weeks, Thursday games flagged (`played_before_snapshot`).
- Pool: at least 40% offensive snap share over the last 3 games before the week. Inactive players dropped.
- Anomaly pass: 12 data errors dropped, 470 Isolation Forest outliers kept as real games.
- Outputs: `player_weeks`, `player_games`, `ecr_weekly` (see "Shared outputs").

## Phase C: Explore (open)
**Goal:** learn what drives fantasy points before modeling. Training seasons only. Notebook section 4.
**Cover:**
- point distributions by position
- week-to-week stability of usage
- how fast past games lose predictive value (correlation with each of the last 1-8 games and last season). D found the best half-life is 8 games; C shows why.
- early season (weeks 1-4) vs. later
- drift across seasons 2020-2024 (do older seasons look different?)
- Vegas lines vs. production
- how accurate the experts have been (the bar to beat)

**Produce:** plots and tables, each with a "What we see" note.

## Phase D: Features (done)
Notebook section 5. 48 features in 7 families, saved to `features.parquet` and `feature_families.json`.
- Windows per stat: last game, last 3, recency-weighted (half-life 8 games, beat equal weighting in CV), season to date.
- Situation family from the pre-game injury report: changed team, weeks since last game, questionable, starting QB out, vacated target and carry share.
- Shared `cv_folds()`, `evaluate()`, `cross_validate()` in section 5.6.
- Family comparison: Ridge on all features gets RMSE 6.81, Spearman 0.515. Usage, situation, and game script add the most; luck adds almost nothing.
- Leakage test, ECR check, and a McCaffrey spot check (trade and return from IR) all pass.

## Phase E: Points regression (main task, done)
**Goal:** find the best model for weekly points. Notebook section 6.
**Already done:** the last cell of section 6 publishes the Ridge baseline to `preds_regression.parquet` and `regression_choice.json`.
**Cover:**
- simple baselines as the bar: last week's points, last-3 average (numbers already in section 5.8)
- a range of course models, each tuned with `cv_folds()`
- how much the past counts, tuned by CV: training window (all seasons vs. last k) and sample weights that fade with a game's age
- refit cadence for the top 2-3 models: refit weekly through the validation season vs. once before it
- where models overfit (train vs. validation error as complexity grows)
- error by position and by week of season

**Produce:** a model comparison table. Then publish the winner by editing `REGRESSION_CHOICE` and `make_regression_model()` in the last cell (set `status` to `final`).

**Result:** voting ensemble of Lasso, gradient boosting, and an MLP, each tuned in 6.2. All seasons, equal weights, refit every week. CV RMSE 6.77, Spearman 0.525 (Ridge baseline: 6.81, 0.515). Short windows, age weights, stacking, and per-position models did not help. Scratch work is in `notebooks/E_regression.ipynb`.

## Phase F: Tiers (extension, open)
**Goal:** answer the instructor's question: does a direct tier classifier beat tiers cut from our predicted points? Notebook section 7.
**Compare three ways of tiering each position-week:**
- cut from `preds_regression.parquet` ranks (rerun when E goes final)
- a direct classifier on the same features
- cut from ECR ranks (`ecr_weekly.parquet`, rules in "Shared outputs")

Handle class imbalance and use classifier metrics (per-class precision and recall, macro F1, confusion matrix).
**Produce:** `tier_results.parquet`, a comparison table, and a one-paragraph answer.

## Phase G: Usage clustering (extension, open)
**Goal:** group players into usage profiles that reflect their current role. Notebook section 8.
**Cover:**
- cluster on recent usage features (the `usage` family, e.g. `*_last3`, `*_recent`), so players can move between profiles
- compare a few clustering methods with validation metrics
- describe the clusters in plain terms, and how often players switch
- with vs. without cluster labels (and a switched flag) in the regression, using `make_regression_model()` (rerun when E goes final)

**Produce:** `usage_clusters.parquet`, a cluster comparison, cluster descriptions, and the with/without result.

## Phase H: Final test vs. experts (done)
**Goal:** predict 2025 once with every setting frozen. Notebook section 9.
- Rebuild the model from `regression_choice.json` and `make_regression_model()`.
- Use E's refit cadence: the model for 2025 week t trains on 2020-2024 plus 2025 weeks before t (walk-forward, not leakage).
- Compare to ECR on the same players and weeks: overall, by position, week by week, weeks 1-4 vs. the rest, and recently changed situations vs. everyone else.

**Produce:** `preds_test.parquet`, final metrics and plots, and a plain verdict (beat, tie, or lose to the experts).

**Result:** we lose to the experts on rank correlation (weekly Spearman 0.489 vs. 0.512 on 3,025 player-weeks, 95% interval for the gap -0.039 to -0.008) and are slightly ahead on top-12 hit rate (0.526 vs. 0.514). 2025 RMSE 6.83 vs. 6.77 in CV. Checked by `tests/test_final_test_outputs.py`.

## Phase I: Analysis and wrap-up (last)
**Goal:** explain the results. Notebook section 10.
**Cover:** what drives predictions (including recent vs. older games), where we and the experts miss and who was right when we disagreed, conclusions, the "What we tried and didn't keep" table, and a short societal-impact note.
**Also produce:** writeup notes (bullets and numbers) and likely Q&A questions with answers.

## Phase J: Verify (last)
**Goal:** prove it works and meets the rubric.
- `scripts/verify.sh` passes from a fresh kernel. Existing tests already cover the notebook structure, clean data, feature leakage, and the regression outputs.
- Add tests for: the walk-forward test (each week trained only on earlier games), no 2025 game used to choose settings, and reported metrics matching the saved predictions.
- Write `RUBRIC_AUDIT.md` mapping each rubric item to its evidence (DONE, PARTIAL, MISSING).
- Reread the notebook as a strict grader and fix what's unclear.

**Check:** verify passes, and nothing is committed.
