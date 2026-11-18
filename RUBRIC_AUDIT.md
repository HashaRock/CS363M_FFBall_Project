# Rubric audit

Each rubric item from `Project_Expectations.pdf`, with where the evidence is and a status. Checked on Nov 18, 2026, after a fresh run of `scripts/verify.sh`.

**DONE** = clearly covered. **PARTIAL** = covered, but a grader could push on it. **MISSING** = not done yet.

## Summary

| Item | Pts | Status | One-line reason |
|---|---|---|---|
| Data cleaning | 10 | DONE | 7 sources joined on one ID, rule and model-based anomaly checks, every dropped row counted |
| Data exploration | 10 | DONE | 5 subsections and 5 plots, all on training seasons only |
| Feature engineering | 20 | DONE | 48 features in 7 families, recency tuned by CV, leakage tested by hand |
| Modeling | 40 | DONE | 12 regressors tuned, 2 ensembles, overfitting curves, 3 classifiers, 3 clustering methods |
| Code and notebook quality | 10 | PARTIAL | Clear narration and tests; sections 4 and 8 use a terser code style; writeup not written yet |
| Outcome and results | 10 | DONE | One frozen test on 2025, a verdict rule set in advance, error and disagreement analysis |
| Process writeup (PDF, max 5 pages) | n/a | MISSING | The team writes this. Notes are ready in `WRITEUP_NOTES.md` |
| Work split in the writeup | n/a | MISSING | Goes at the end of the writeup |

## Data preparation (40)

**Data cleaning (10): DONE**
- Appropriate data: 7 public nflverse sources (stats, snaps, rosters, schedules with Vegas lines, expected points, injuries, ECR), cached in `data/raw/` (section 3 intro).
- Team codes and IDs standardized; foreign IDs that map to more than one player are dropped (3.1).
- Anomaly detection with three methods: rule checks, a cross-source check against play-by-play (99.85% agree within 1 point), and Isolation Forest. 12 real data errors dropped, 470 real extreme games kept (3.4).
- Player pool built from past games only, with every excluded row explained by roster status (3.5).
- Rows kept at each step and join rates shown (3.7). Checked by `tests/test_clean_data.py`.

**Data exploration (10): DONE**
- Points by position, memory of past games, early vs. late season, Vegas lines vs. production, and how good the experts are on our pool (4.1-4.5). 5 plots.
- All on 2020-2024, so the test season is never explored.
- Exploration informs later choices: the expert benchmark in 4.5 (Spearman 0.509) sets what "close to the experts" means in section 9.

**Feature engineering (20): DONE**
- 48 features in 7 families: production, usage, luck, game script, opponent, situation, position (5.1-5.5).
- Recency tuned rather than assumed: half-lives of 3 to 16 games against an equal-weight version (5.7).
- Each family's value measured by adding it on top of the others, with a plot (5.8).
- Leakage test recomputes 300 rows by hand; spot check on a traded and injured player (5.9). Checked by `tests/test_features.py`.

## Modeling (40): DONE
- **Correct techniques:** expanding-window CV by season (no random folds on time data), one shared `evaluate()` for every model, selection metric fixed before results (5.6, section 6 intro).
- **Comparison:** 12 course regressors, from linear regression to MLP (6.2), plus voting and stacking (6.3).
- **Tuning:** a grid per model (6.2), training window, age weights, and refit cadence (6.4), and complexity curves showing where trees, KNN, and boosting overfit (6.5).
- **Error analysis:** by position, week, and part of season, with bias (6.6).
- **Extensions:** 3 tier classifiers with class-imbalance handling and a bootstrap comparison against the regression cut (7); K-Means, Ward, and DBSCAN compared on silhouette and Davies-Bouldin, then tested as model inputs (8).
- **Established libraries:** every model is from scikit-learn. Checked by `tests/test_regression_outputs.py`, `tests/test_tiers.py`, `tests/test_usage_clusters.py`.

## Code and notebook quality (10): PARTIAL
- **Done well:** one config cell; small named functions; every step ends with a "What we see" note that uses real numbers; only the chosen approach is shown, with dropped ideas in one table (10.4). `verify.sh` runs the notebook fresh and then 44 tests.
- **Tests added in Phase J:**
  - `tests/test_test_season_untouched.py`: no 2025 game reaches any choice. Checks that `cv_folds()` cannot hand out a 2025 row, that sections 4-8 only fit on training tables, that saved choice files hold no 2025 rows, and that sections 9-10 tune nothing. We planted three kinds of leak in a copy of the notebook and the test caught all three.
  - `tests/test_reported_metrics.py`: the headline numbers in the text match the saved predictions (CV scores, 2025 scores, the head-to-head with the experts, the weekly gap, the disagreement shares). Changing one number in the text makes it fail.
  - The walk-forward test (each 2025 week trained only on earlier games) already existed in `tests/test_final_test_outputs.py`.
- **Where a grader could push:**
  - Sections 4 and 8 use a terser style than the rest (prefixed names like `c_rows` and `g_rows`, single quotes, packed lines), and their notes are generated inside code cells. They read fine but look different. Not reformatted, to keep the diff small the day before the deadline.
  - The writeup is part of this item, and it is not written yet.

## Outcome and results (10): DONE
- **Complete:** the main task and both extensions answered, including the instructor's question about tiers (7.5).
- **Fair test:** 2025 used once with every setting frozen, walk-forward refits, and a verdict rule set before running (section 9 intro).
- **Interpreted correctly:** we report the loss to the experts plainly (Spearman 0.489 vs. 0.512, 95% interval for the gap -0.039 to -0.008), next to the small top-12 edge (0.526 vs. 0.514).
- **Analysis:** what drives predictions, including recent vs. older games (10.1); who was right when we disagreed with the experts (10.2); conclusions (10.3); societal impact (10.5).

## Fixes made during the strict reread
- 5.7 listed 3 of the 5 half-lives tried; now lists all 5.
- 5.8 and 10.1 had results only as tables; each now has a plot.
- 3.1 and 3.3 printed results with no note; each now has a "What we see".
- 8.3's generated note called a +0.0003 Spearman change a gain; it now calls it noise, which matches the text below it.
- 9.5 gave one pair of numbers in the experts-first order; now it is ours first, like the rest.

## Left for the team
1. Write the process writeup (max 5 pages, PDF) with the work split at the end. `WRITEUP_NOTES.md` has the bullets and numbers.
2. Optional: restyle the code in sections 4 and 8 to match the rest.
3. Commit and push.
