"""
scripts/run_pipeline.py
=======================
Runs the full data pipeline end-to-end:
    1. Download ECO database
    2. Parse PGN + UCI files
    3. Extract Stockfish features
    4. Merge all sources
    5. Engineer features
    6. Integrate Lichess dataset
    7. Save games.csv and merged_games.csv

Usage
-----
    python scripts/run_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tomllib
from src.data.load_data import (
    download_eco_database, parse_pgn, build_eco_lookup, match_eco,
    parse_uci, extract_stockfish_features, merge_datasets,
    integrate_datasets, save_dataset,
)
from src.features.build_features import engineer_features, validation_report


def main():
    with open("configs/config.toml", "rb") as f:
        cfg = tomllib.load(f)

    p = cfg["data"]

    print("Step 1 — ECO database")
    df_eco    = download_eco_database(p["eco_file"])
    eco_lookup = build_eco_lookup(df_eco)

    print("\nStep 2 — Parse PGN")
    df_pgn = parse_pgn(p["pgn_file"])

    print("\nStep 3 — ECO matching")
    eco_results          = df_pgn["moves_san"].apply(lambda m: match_eco(m, eco_lookup))
    df_pgn["eco_code"]   = eco_results.apply(lambda x: x[0])
    df_pgn["opening_name"] = eco_results.apply(lambda x: x[1])
    df_pgn["eco_family"] = eco_results.apply(lambda x: x[2])

    print("\nStep 4 — Parse UCI")
    df_uci = parse_uci(p["uci_file"])

    print("\nStep 5 — Stockfish features")
    df_sf = extract_stockfish_features(p["stockfish_file"])

    print("\nStep 6 — Merge")
    df = merge_datasets(df_pgn, df_uci, df_sf)

    print("\nStep 7 — Feature engineering")
    df = engineer_features(df)

    print("\nStep 8 — Integrate Lichess")
    df_combined = integrate_datasets(df, p["lichess_file"], p["lichess_sf_file"])

    print("\nStep 9 — Save")
    save_dataset(df,          p["games_output"])
    save_dataset(df_combined, p["merged_output"])
    validation_report(df_combined)
    

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
