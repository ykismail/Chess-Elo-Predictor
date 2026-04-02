# Chess Game Outcome Prediction

### CMPS344 Applied Data Science — Phase 2

Predicting chess game outcomes (White wins / Draw / Black wins) using game metadata,
opening theory, and Stockfish engine evaluations across two merged real-world datasets.

---

## Team

> Youssuf Kamel 1220298
> Hana Akabawy
> Nour Eldeen Hassan
> Abdelrahman Ashraf

---

---

## Data Sources

| #   | Source                                                                                                                                | Format          | Rows               | Key Features                            |
| --- | ------------------------------------------------------------------------------------------------------------------------------------- | --------------- | ------------------ | --------------------------------------- |
| 1   | [Kaggle Chess Games (PGN)](https://www.kaggle.com/competitions/finding-elo/data) **need to join the competition to download dataset** | `.pgn`          | ~25,000 (with Elo) | Elo ratings, moves, result              |
| 2   | [Lichess Chess Games](https://www.kaggle.com/datasets/mysarahmadbhat/online-chess-games?select=chess_games.csv)                       | `.csv`          | ~20000             | Casual Chess games                      |
| 3   | [Lichess Game Database](https://github.com/lichess-org/chess-openings)                                                                | `.csv` / `.tsv` | 20,058 + ECO DB    | ECO codes, opening names, time controls |

---

## Setup

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/docs/#installation)

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd project

# Install dependencies
poetry install

# Activate the environment
poetry shell
```

### Dependencies

```toml
[tool.poetry.dependencies]
python     = "^3.10"
pandas     = "^2.0"
numpy      = "^1.25"
python-chess = "^1.10"
requests   = "^2.31"
jupyter    = "^1.0"
scikit-learn = "^1.3"
mlflow     = "^2.9"
pytest     = "^7.4"
```

---

## Running the Pipeline

```bash
poetry run jupyter notebook notebooks/chess_data_pipeline.ipynb
```

Run all cells top to bottom. The notebook will:

1. Download the ECO opening database from GitHub (cached after first run)
2. Parse `data.pgn` and `data_uci.pgn`
3. Match games to ECO opening codes
4. Extract Stockfish evaluation features
5. Merge all sources and engineer features
6. Integrate the Lichess CSV dataset
7. Print three validation reports
8. Save `games.csv` and `merged_games.csv`

## Output Datasets

### `games.csv` — Kaggle PGN games with Stockfish (~23k rows)

Use this for models that leverage Stockfish evaluation features.

| Column                | Type  | Description                               |
| --------------------- | ----- | ----------------------------------------- |
| `event_id`            | int   | Game identifier                           |
| `white_elo`           | int   | White player Elo rating                   |
| `black_elo`           | int   | Black player Elo rating                   |
| `result`              | str   | `1-0`, `0-1`, `1/2-1/2`                   |
| `num_moves`           | int   | Number of full moves                      |
| `white_castled`       | bool  | Whether White castled                     |
| `black_castled`       | bool  | Whether Black castled                     |
| `white_castle_side`   | str   | `kingside`, `queenside`, or `none`        |
| `black_castle_side`   | str   | `kingside`, `queenside`, or `none`        |
| `num_captures`        | int   | Total captures in the game                |
| `termination`         | str   | `checkmate`, `resignation`, or `draw`     |
| `eco_code`            | str   | ECO opening code (e.g. `B90`)             |
| `opening_name`        | str   | Full opening name                         |
| `eco_family`          | str   | Opening family letter (A–E)               |
| `white_acl`           | float | White average centipawn loss              |
| `black_acl`           | float | Black average centipawn loss              |
| `white_blunders`      | int   | White moves losing ≥100 centipawns        |
| `black_blunders`      | int   | Black moves losing ≥100 centipawns        |
| `white_mistakes`      | int   | White moves losing 50–99 centipawns       |
| `black_mistakes`      | int   | Black moves losing 50–99 centipawns       |
| `final_eval`          | int   | Centipawn evaluation at game end          |
| `max_white_advantage` | int   | Peak advantage White held                 |
| `max_black_advantage` | int   | Peak advantage Black held (most negative) |
| `game_sharpness`      | float | Std dev of all centipawn scores           |
| `elo_gap`             | int   | `white_elo - black_elo`                   |
| `avg_elo`             | float | Mean of both Elos                         |
| `acl_gap`             | float | `white_acl - black_acl`                   |
| `winner_multiclass`   | int   | **Target**: 0=Black, 1=Draw, 2=White      |
| `winner_binary`       | float | **Target**: 1=White, 0=Black, NaN=Draw    |

### `merged_games.csv` — All games combined (~43k rows)

Use this for models that do not rely on Stockfish features.
Same columns as above plus:

| Column          | Type | Description                              |
| --------------- | ---- | ---------------------------------------- |
| `source`        | str  | `kaggle_pgn` or `lichess_csv`            |
| `has_stockfish` | bool | Whether Stockfish features are available |

Stockfish columns are `NaN` for `lichess_csv` rows.

---

## Target Variable

**`winner_multiclass`**

- `0` = Black wins
- `1` = Draw
- `2` = White wins

| Dataset                 | Black wins | Draw   | White wins |
| ----------------------- | ---------- | ------ | ---------- |
| games.csv (Kaggle only) | ~6,900     | ~7,800 | ~8,900     |
| merged_games.csv (all)  | ~16,000    | ~8,700 | ~18,900    |

---

## Experiment Design

Two parallel modelling tracks are planned:

| Track | Dataset                   | Features                       | Goal                                 |
| ----- | ------------------------- | ------------------------------ | ------------------------------------ |
| A     | `merged_games.csv` (~43k) | Metadata + opening only        | Baseline & generalisation            |
| B     | `games.csv` (~23k)        | Metadata + opening + Stockfish | Does engine eval improve prediction? |

Both tracks will be logged in MLflow for direct comparison.

---

## Academic Integrity

All code is original. Open-source libraries and public datasets are used with citation.
See individual source URLs in the Data Sources section above.
