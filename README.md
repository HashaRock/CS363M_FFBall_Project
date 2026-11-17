# Fantasy Football Position/ADP/ECR Analysis: CS363M Final Project

## Hassaan Ali, Shyamal Gupta, Amin Momin

## Overview

We predict each QB, RB, WR, and TE's weekly PPR fantasy points using only pre-game information (recent usage and production, Vegas lines, opponent strength). We then rank players by those predictions and compare our rankings to the FantasyPros expert consensus (ECR), using 2020-2025 NFL data.

## Getting started

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/verify.sh     # runs the notebook fresh (downloads data the first time), then the tests
```

- The deliverable is `fantasy_football_rankings.ipynb`.
- For a new phase, copy `notebooks/starter.ipynb` to `notebooks/<phase>_<topic>.ipynb`.
- `PROJECT_PLAN.md` has the phases, the saved data files, and how to use them. `AGENTS.md` has the project rules.
