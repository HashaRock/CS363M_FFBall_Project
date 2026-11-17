# AGENTS.md

CS 363M final project (UT Austin, Fall 2026). Team: Hassaan Ali, Shyamal Gupta, Amin Momin.
Notebook + writeup due **Nov 19, 2026**. Q&A **Dec 1**. The plan is in `PROJECT_PLAN.md`.

## The project
Predict each QB/RB/WR/TE's fantasy points (PPR) for the coming week using only pre-game info. Rank players by those predictions and compare our rankings to the FantasyPros expert consensus (ECR).

- **Main task:** points regression.
- **Extensions:** tiers (does a tier classifier beat tiers cut from our predictions?) and usage-profile clustering.
- We are graded on methodology, not accuracy, and the team has to explain every choice in the Q&A.

Instructor feedback we're answering: *"Which is your main problem, and which are extensions? The tiers come from points, so they will be closely tied to your regression results."*

## Rules
- **No git writes.** Don't add, commit, branch, or push. The team does that.
- **ECR is never a model feature.** It is used only to compare against.
- **No leakage.** Features for week t use weeks before t only. 2025 is the test set and is used once, at the end: every model choice and setting is frozen before 2025 is touched. In that final test, the model for week t may be refit on all games before week t (walk-forward); that is allowed because those games are already over.
- **Recency is tuned, not assumed.** Recent games and current situation should count more (see "Recency principle" in `PROJECT_PLAN.md`), but each recency choice must beat a plain version in CV to stay.
- **Course methods only**, all from scikit-learn (list in `PROJECT_PLAN.md`). No xgboost, lightgbm, or methods outside the syllabus.
- **One notebook** is the deliverable. The notebook shows what we did; the writeup (written by the team) explains why.
- **Readable code:** small, clearly named functions; one config cell; each step ends with a short "What we see" note that uses real numbers.
- **Only the chosen approach goes in the notebook.** Dropped ideas get one line each in a "What we tried and didn't keep" table.
- **Writing style:** no emojis, no em dashes, and keep everything short and plain.
- **Before saying a phase is done,** run its checks and report the results honestly.

## Working on a phase
- Read "Shared outputs and how to use them" in `PROJECT_PLAN.md` before writing code. It lists every saved file, which columns are safe as features, and the file each phase must save.
- Start from a scratch notebook copied from `notebooks/starter.ipynb` (see "Avoiding conflicts").
- Use the shared `cv_folds()`, `evaluate()`, and `cross_validate()` for every model so results are comparable.
- `player_weeks.parquet` holds each game's own box score. Never use it as features; use `features.parquet` and `feature_families.json`.

## Avoiding conflicts
Anyone can change any file. Build in a temp spot first, then update the shared version (the single source of truth) in one focused step.
- **Temp spots:** iterate in `notebooks/<phase>_<topic>.ipynb` and save work-in-progress files to `data/scratch/<phase>/`. Nothing reads from there except you.
- **Promote when it works:** move final outputs to `data/processed/` under the names in `PROJECT_PLAN.md`, and move finished cells into your section of the main notebook.
- **Before editing a shared file** (main notebook, starter, docs, config): pull first, change only what you need, then run `bash scripts/verify.sh`.
- **Shared functions** (`cv_folds`, `evaluate`, `cross_validate`): change them in main notebook section 5.6 and the starter together. A test checks they match.
- **New raw data:** cache it in `data/raw/` under a new file name instead of replacing an existing one.
- **Keep diffs small:** don't reformat or reorder code you aren't changing.

## Commands
```bash
source .venv/bin/activate
bash scripts/verify.sh     # runs the notebook fresh, then the tests
```
