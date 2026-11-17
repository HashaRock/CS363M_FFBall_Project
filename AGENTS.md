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

## Commands
```bash
source .venv/bin/activate
bash scripts/verify.sh     # runs the notebook fresh, then the tests
```
