# Project Plan

Read `AGENTS.md` first.

## How to use this
Work **one phase at a time**. Each phase gives a goal, what it must produce, and how to check it. The **how** is up to whoever takes the phase: look at the data, think through options, and pick the approach you can best explain. Before starting, briefly propose your approach to the team. When you finish, leave the outputs listed so the next phase can pick up from there.

Phases after C can be split between people (see the dependency notes).

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
- schedules with Vegas lines
- expected fantasy points (ffopportunity)
- FantasyPros ECR archive (used only to compare against)
- player ID crosswalk
- optional: play-by-play and injury reports

Seasons: 2019 for warm-up lags only, 2020-2024 for training and CV, 2025 for the test.

---

## Phase A: Setup
**Goal:** a working environment and an empty notebook skeleton.
**Produce:**
- `.venv`
- `requirements.txt`
- `fantasy_football_rankings.ipynb` with the section headings
- `scripts/verify.sh`
- `data/` added to `.gitignore`

**Check:** the notebook runs top to bottom from a fresh kernel.

## Phase B: Load and clean
**Goal:** one clean table with one row per player per week, all sources joined.
**Things to handle:**
- mismatched team codes and player IDs
- matching each ECR snapshot to the right week
- games that kicked off before the experts' snapshot (e.g. Thursday games)
- byes and players who barely played
- picking which players to model each week without using that week's results
- an anomaly-detection pass to separate data errors from real big games

**Produce:**
- the clean table, saved to `data/processed/`
- a short table showing rows kept at each step
- join rates

**Check:** no duplicate keys, no missing targets, high join rates.

## Phase C: Explore
**Goal:** learn what drives fantasy points before modeling.
**Should cover:**
- point distributions by position
- how stable usage is week to week
- Vegas lines vs. production
- how accurate the experts have been historically (this is the bar to beat)

Add anything else that looks interesting. Use training seasons only.
**Produce:** plots and tables, each followed by a "What we see" note.

## Phase D: Features
**Goal:** pre-game features that capture usage, recent production, luck (actual vs. expected points), game script, and opponent strength.
**Produce:**
- the feature set, grouped into families
- a test showing features only use past weeks
- a comparison showing what each feature family adds

**Check:** no ECR columns in the features; a spot-check of one player's lags.

*After Phase D, phases E, F, and G can run in parallel.*

## Phase E: Points regression (main task)
**Goal:** find the best model for weekly points, starting from simple baselines like "last week's points".
**Requirements:**
- compare a range of course models
- tune them with CV that respects time order (no future seasons in training)
- show where models overfit

Evaluate on point error (RMSE, MAE) and on ranking quality: rank correlation per week and how many of our top 12 actually finished top 12.
**Produce:** a model comparison table and the chosen model.

## Phase F: Tiers (extension)
**Goal:** answer the instructor's question. Compare three ways of assigning tiers:
- cut tiers from the regression's predicted ranks
- a direct tier classifier
- tiers cut from the expert rankings

Handle the class imbalance and use proper classifier metrics.
**Produce:** a comparison and a clear one-paragraph answer.

## Phase G: Usage clustering (extension)
**Goal:** group players into usage profiles.
**Requirements:**
- compare a few clustering methods with validation metrics
- describe the clusters in plain terms
- test whether the cluster labels improve the regression

**Produce:** a cluster comparison, cluster descriptions, and the with/without result.

## Phase H: Final test vs. experts
**Goal:** train on 2020-2024, then predict 2025 once. Compare to ECR on the same players and weeks: overall, by position, and week by week.
**Produce:** final metrics and plots, with a plain verdict (beat, tie, or lose to the experts).

## Phase I: Analysis and wrap-up
**Goal:** explain the results.
**Should cover:**
- what drives predictions
- where we and the experts miss, and who was right when we disagreed
- conclusions
- the "What we tried and didn't keep" table
- a short note on societal impacts

**Also produce:** writeup notes (bullets and numbers for the team to write from) and likely Q&A questions with answers.

## Phase J: Verify
**Goal:** prove it works and meets the rubric.
**Requirements:**
- `scripts/verify.sh` runs the notebook fresh and runs tests anyone can rerun
- the tests independently recheck no leakage, that 2025 was never used in training, and that the reported metrics match the saved predictions
- a short `RUBRIC_AUDIT.md` maps each rubric item to its evidence, marked DONE, PARTIAL, or MISSING
- reread the notebook as a strict grader would, then fix what's unclear

**Check:** verify passes, and nothing is committed.
