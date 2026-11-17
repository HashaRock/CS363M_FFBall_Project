# Project Plan

Read `AGENTS.md` first.

## How to use this
Work **one phase at a time**. Each phase gives a goal, what it must produce, and how to check it. The **how** is up to whoever takes the phase: look at the data, think through options, and pick the approach you can best explain. Before starting, briefly propose your approach to the team. When you finish, leave the outputs listed so the next phase can pick up from there.

Phases after C can be split between people (see the dependency notes).

**Dependencies:** A -> B -> D (C needs only B and can run alongside D). E, F, G each need D's saved features; F also needs E's predictions file and G needs E's model, so E publishes a baseline first. H needs E; I needs E-H; J is last.

**Status:** A, B, D done. Outputs are in `data/processed/` (`player_weeks`, `player_games`, `ecr_weekly`, `features`, `feature_families.json`). Use `cv_folds()` and `evaluate()` from notebook section 5.6 for every model so results are comparable. Next: C, E, F, G.

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
- optional: injury reports (pre-game practice and game status), depth charts, play-by-play

Seasons: 2019 for warm-up lags only, 2020-2024 for training and CV, 2025 for the test.

---

## Phase A: Setup (done)
**Goal:** a working environment and an empty notebook skeleton.
**Produce:**
- `.venv`
- `requirements.txt`
- `fantasy_football_rankings.ipynb` with the section headings
- `scripts/verify.sh`
- `data/` added to `.gitignore`

**Check:** the notebook runs top to bottom from a fresh kernel.

## Phase B: Load and clean (done)
**Goal:** one clean table with one row per player per week, all sources joined.
**Things to handle:**
- mismatched team codes and player IDs
- matching each ECR snapshot to the right week
- games that kicked off before the experts' snapshot (e.g. Thursday games)
- byes and players who barely played
- picking which players to model each week without using that week's results
- an anomaly-detection pass to separate data errors from real big games

**Produce:**
- `data/processed/player_weeks.parquet`: the modeling table (pool players, 2020-2025), with `kickoff` so later phases can cut data by date
- `data/processed/player_games.parquet`: every game played 2019-2025, the history for all lags
- `data/processed/ecr_weekly.parquet`: ECR, kept separate from the modeling table
- a short table showing rows kept at each step
- join rates

**Check:** no duplicate keys, no missing targets, high join rates.

## Phase C: Explore
**Goal:** learn what drives fantasy points before modeling, including how quickly old information goes stale.
**Should cover:**
- point distributions by position
- how stable usage is week to week
- **how predictive past games are by how long ago they were:** correlation of this week's points with each of the last 1-8 games, and with last season's average. This sets the starting range for the recency weights in D.
- **early vs. late season:** how much harder weeks 1-4 are to predict, when the only history is last season
- **drift across seasons:** do scoring levels or usage patterns shift from 2020 to 2024? This tells E whether older seasons should count less.
- Vegas lines vs. production
- how accurate the experts have been historically (this is the bar to beat)

Add anything else that looks interesting. Use training seasons only.
**Produce:** plots and tables, each followed by a "What we see" note.

## Phase D: Features (done)
**Goal:** pre-game features that capture usage, recent production, luck (actual vs. expected points), game script, opponent strength, and the player's **current situation**.
**Recency in the features:**
- **Several time windows per stat:** last game, last 3, exponentially weighted average (half-life tuned, starting from what C found), and season to date. Let the model weigh them.
- **Carry-over across seasons:** early-season weeks fall back on last season, shrunk toward the position average, and the weight on last season fades as this season's games pile up.
- **Situation changes:** changed team since the last game, a new starting QB on the team, a teammate at the same position out (opportunity freed up), games since a role change, back from injury.
- **Current status (optional sources):** the pre-game injury report status and depth chart rank for that week. Both are public before kickoff.

**Produce:**
- the feature set, grouped into families (include a "recency and situation" family)
- a test showing features only use past weeks
- a comparison showing what each feature family adds, including recency-weighted vs. plain averages

**Check:** no ECR columns in the features; a spot-check of one player's lags, including one who changed teams mid-season.

*After Phase D, phases E, F, and G can run in parallel.*

## Phase E: Points regression (main task)
**Goal:** find the best model for weekly points, starting from simple baselines like "last week's points" and "average of the last 3 games".
**Requirements:**
- compare a range of course models
- tune them with CV that respects time order: expanding-window folds by season (train 2020-2022, validate 2023; train 2020-2023, validate 2024)
- **tune how much the past counts:** treat these as hyperparameters chosen by CV:
  - training window (all seasons vs. the last k seasons)
  - sample weights that decay with a game's age (half-life in seasons), passed as `sample_weight`
- **check the refit cadence:** for the top 2-3 models, rerun the validation season walk-forward (refit each week on all games before it) and compare to fitting once before the season. Pick the cadence for H.
- show where models overfit

Evaluate on point error (RMSE, MAE) and on ranking quality: rank correlation per week and how many of our top 12 actually finished top 12. Also report error by week of season, to see whether recency features close the early-season gap.
**Produce:** a model comparison table, the chosen model with its recency settings, and `data/processed/preds_regression.parquet` (baseline first, final later, same columns).

## Phase F: Tiers (extension)
**Goal:** answer the instructor's question. Compare three ways of assigning tiers:
- cut tiers from the regression's predicted ranks
- a direct tier classifier (same features and recency settings as E)
- tiers cut from the expert rankings

Handle the class imbalance and use proper classifier metrics.
**Produce:** a comparison and a clear one-paragraph answer.

## Phase G: Usage clustering (extension)
**Goal:** group players into usage profiles that reflect their **current** role.
**Requirements:**
- cluster on recent usage (for example the last 4 games) rather than a full season, so a player can move between profiles as their role changes
- compare a few clustering methods with validation metrics
- describe the clusters in plain terms, and how often players switch clusters during a season
- test whether the cluster labels (and a "switched cluster" flag) improve the regression

**Produce:** a cluster comparison, cluster descriptions, and the with/without result.

## Phase H: Final test vs. experts
**Goal:** predict 2025 once, with every setting frozen from CV. Use the refit cadence chosen in E: the model for 2025 week t is trained on 2020-2024 plus 2025 weeks before t. This is not leakage, since those games are over before week t, and the experts have the same information. Compare to ECR on the same players and weeks:
- overall, by position, and week by week
- early season (weeks 1-4) vs. the rest
- players whose situation changed recently (new team, new role) vs. everyone else

**Produce:** final metrics and plots, with a plain verdict (beat, tie, or lose to the experts).

## Phase I: Analysis and wrap-up
**Goal:** explain the results.
**Should cover:**
- what drives predictions, and how much weight lands on recent vs. older games
- where we and the experts miss, and who was right when we disagreed
- conclusions
- the "What we tried and didn't keep" table (including any recency setting that did not help)
- a short note on societal impacts

**Also produce:** writeup notes (bullets and numbers for the team to write from) and likely Q&A questions with answers.

## Phase J: Verify
**Goal:** prove it works and meets the rubric.
**Requirements:**
- `scripts/verify.sh` runs the notebook fresh and runs tests anyone can rerun
- the tests independently recheck:
  - no leakage: every feature for week t uses earlier games only
  - in the walk-forward test, each week's model was trained only on games that kicked off before that week
  - no 2025 game was used to choose models or settings
  - the reported metrics match the saved predictions
- a short `RUBRIC_AUDIT.md` maps each rubric item to its evidence, marked DONE, PARTIAL, or MISSING
- reread the notebook as a strict grader would, then fix what's unclear

**Check:** verify passes, and nothing is committed.
