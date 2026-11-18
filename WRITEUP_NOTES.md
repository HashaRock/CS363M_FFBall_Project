# Writeup notes and Q&A prep

Bullets and numbers for the writeup, plus questions we expect in the Dec 1 Q&A. Every number comes from the notebook.

## Writeup notes

**Problem**
- Main task: predict each QB/RB/WR/TE's PPR points for next week from pre-game info only, rank by those predictions, and compare to the FantasyPros expert consensus (ECR).
- Extensions: tiers (classifier vs. tiers cut from our predictions) and usage-profile clustering.
- Answer to the instructor: tiers come from points, so we made points the main task and tested whether a separate tier model adds anything. It does not (section 7).

**Data (section 3)**
- Public nflverse data: stats, snaps, rosters, schedules, betting lines, expected points, injury reports, ECR.
- 22,145 player-weeks, 2020-2025. 2019 is used only for lags. 2025 is the test set, used once.
- Player pool chosen from earlier games only: 40%+ snap share over the last 3 games.
- Anomaly pass separates data errors from real big games.

**Features (section 5)**
- 48 features in 7 families: recent production, usage, luck, game script, opponent, situation, position.
- Recency-weighted averages (half-life 8 games) beat equal-weight last-8 averages (Spearman 0.501 vs. 0.490).
- Leakage check: 300 rows recomputed by hand, all match.

**Model (section 6)**
- 12 course models tuned with expanding-window CV (validate on 2022, 2023, 2024). Picked on weekly Spearman within position.
- Final: voting ensemble of Lasso, gradient boosting, and an MLP. CV RMSE 6.77, Spearman 0.525, top-12 hit rate 0.489.
- Plain Ridge gets 0.515, so the model adds only about 0.01. The features do the work.
- All seasons with equal weights beat shorter windows and age weights. Refit every week: tiny gain.

**Final test vs. experts (section 9)**
- 2025: RMSE 6.83, Spearman 0.505. Close to CV. Last-3 baseline: 7.72 and 0.407.
- Same 3,025 player-weeks: our Spearman 0.489, ECR 0.512. Gap 95% interval -0.039 to -0.008, so we lose by the rule we set in advance.
- Top-12 hit rate: 0.526 for us, 0.514 for ECR.
- Experts lead most at QB (their 0.433 vs. our 0.375) and in weeks 2-4. TE is level.

**Analysis (section 10)**
- Recent production is the main driver (shuffling it drops Spearman by 0.27). Usage is next (0.09).
- Recency-weighted averages matter far more than last-game, last-3, or last-season stats.
- Our ranks and ECR's correlate at 0.93. When we disagree by 6+ places the experts are closer 52% of the time; at 12+ places we are closer 53%.
- Experts win on news we cannot see (suspensions, players returning). We win when they overreact to injury tags.

**Extensions**
- Tiers: regression cut macro F1 0.495 vs. classifier 0.454, ahead in 47 of 54 weeks.
- Clusters: 4 usage roles (passers, high-volume receivers, rushing backs, low-volume receivers). No predictive gain.

## Likely Q&A questions

**Why is points regression the main task and tiers the extension?**
Tiers are a coarse cut of the points ranking. We tested a direct tier classifier anyway and it lost (0.454 vs. 0.495 macro F1). It judges each player alone, so it fills tiers with the wrong number of players.

**How do you know there is no leakage?**
Every feature for week t uses games before t only (shift by one game). A test recomputes 300 rows by hand. ECR is never a feature. 2025 was used once, after every setting was frozen.

**Isn't refitting on 2025 games during the test leakage?**
No. The model for week t is trained only on games already played before week t. A real forecaster would do the same.

**Why pick models on Spearman and not RMSE?**
The goal is ranking players, and Spearman within each position-week measures that directly. We still report RMSE next to it.

**Why an ensemble if it only adds 0.01?**
It was the best on our pre-set metric and better than each member in two of three CV seasons. It also smooths out the MLP's sensitivity to its random seed.

**Why did you lose to the experts?**
They read news our data does not have: suspensions, practice reports, role changes. Their edge is largest at QB and early in the season, when our features have little 2025 data.

**Is the loss real or noise?**
The 95% bootstrap interval for the weekly gap is -0.039 to -0.008, all below zero. So it is small but real.

**How did you handle recency?**
Three places, each tested against a plain version in CV: recency-weighted features (kept), shorter training windows and age weights (dropped), weekly refit (kept, small gain).

**Why does luck stay in the model if it adds nothing?**
Lasso sets it to zero on its own, and dropping it changed Spearman by only 0.001. Keeping it avoids an extra hand-made choice.

**Why expanding-window CV and not random k-fold?**
Random folds would let the model train on future games. Expanding windows always train on the past and validate on the next season.

**What would you do with more time?**
Add news-based signals (practice participation, depth charts) and treat players with little history separately, since that is where the experts win.

**Societal impact?**
Fantasy rankings feed betting. Our model is not better than free expert rankings, so it should not be used as a betting edge. It also ranks rookies and players back from injury low by design.
