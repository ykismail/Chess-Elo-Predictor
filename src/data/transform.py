"""
Phase 3 — Transformation & Encoding
=====================================
Responsibilities:
  1. ECO opening matching on parsed PGN data   (was commented block #1 in data_loading.py)
  2. Stockfish feature extraction from raw CSV  (was commented block #2 in data_loading.py)
  3. Inner-join merge: PGN + UCI + Stockfish    → Kaggle unified DataFrame
  4. Schema harmonisation / column encoding     (result, termination, castle columns)

Note: `extract_stockfish_features()` replaces the raw `pd.read_csv` + broken rename
that was in the original data_loading.py  (`rename({"event_id":"Event"})` was a bug —
the column in stockfish.csv is called "Event", not "event_id").

Output
------
  data/intermediate/kaggle_merged.csv  — fully merged Kaggle dataset,
                                         ready for Phase 4 feature engineering.
"""

import os
import sys

# ── Path bootstrap ────────────────────────────────────────────────────────────
script_dir  = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

import pandas as pd
from load_data import (
    build_eco_lookup,
    extract_stockfish_features,
    match_eco,
    merge_datasets,
)

# ── Directory paths ───────────────────────────────────────────────────────────
raw_dir          = os.path.join(project_dir, "data", "raw")
intermediate_dir = os.path.join(project_dir, "data", "intermediate")
external_dir     = os.path.join(project_dir, "data", "external")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load Phase-1 outputs
# ─────────────────────────────────────────────────────────────────────────────

def load_parsed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the intermediate CSVs written by Phase 1 (data_loading.py).

    Returns
    -------
    parsed_data_uci : DataFrame
    parsed_data_pgn : DataFrame
    """
    parsed_data_uci = pd.read_csv(os.path.join(intermediate_dir, "parsed_data_uci.csv"))
    parsed_data_pgn = pd.read_csv(os.path.join(intermediate_dir, "parsed_data_pgn.csv"))
    print(f"Loaded parsed_data_uci : {parsed_data_uci.shape}")
    print(f"Loaded parsed_data_pgn : {parsed_data_pgn.shape}")
    return parsed_data_uci, parsed_data_pgn


# ─────────────────────────────────────────────────────────────────────────────
# 2. ECO opening matching   (was commented block #1 in data_loading.py)
# ─────────────────────────────────────────────────────────────────────────────

def apply_eco_matching(parsed_data_pgn: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich parsed_data_pgn with ECO opening codes, names, and family letters
    using longest-prefix matching against the Lichess ECO database.

    Parameters
    ----------
    parsed_data_pgn : DataFrame — output of parse_pgn() from Phase 1

    Returns
    -------
    DataFrame with three new columns:
        eco_code     : str  (e.g. 'B20')
        opening_name : str  (e.g. 'Sicilian Defense')
        eco_family   : str  (e.g. 'B')
    """
    eco_csv = os.path.join(external_dir, "eco_openings.csv")
    print(f"Loading ECO database from {eco_csv}...")
    df_eco   = pd.read_csv(eco_csv)
    eco_lookup = build_eco_lookup(df_eco)
    print(f"  {len(eco_lookup):,} opening entries loaded into lookup.")

    print("Matching ECO codes to PGN games (longest-prefix)...")
    eco_results = parsed_data_pgn["moves_san"].apply(
        lambda m: match_eco(m, eco_lookup)
    )
    parsed_data_pgn = parsed_data_pgn.copy()
    parsed_data_pgn["eco_code"]     = eco_results.apply(lambda x: x[0])
    parsed_data_pgn["opening_name"] = eco_results.apply(lambda x: x[1])
    parsed_data_pgn["eco_family"]   = eco_results.apply(lambda x: x[2])

    matched = (parsed_data_pgn["eco_code"] != "Unknown").sum()
    print(f"  ECO matched : {matched:,} / {len(parsed_data_pgn):,} games "
          f"({matched / len(parsed_data_pgn) * 100:.1f}%)")
    return parsed_data_pgn


# ─────────────────────────────────────────────────────────────────────────────
# 3. Stockfish feature extraction   (was commented block #2 in data_loading.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_stockfish_features() -> pd.DataFrame:
    """
    Extract per-game Stockfish evaluation features from the raw stockfish.csv.

    Uses `extract_stockfish_features()` from load_data.py which returns a
    DataFrame with "event_id" already set — fixing the broken rename
    (`{"event_id": "Event"}`) that existed in the original data_loading.py.

    Returns
    -------
    DataFrame with columns:
        event_id, total_half_moves, white_acl, black_acl,
        white_blunders, black_blunders, white_mistakes, black_mistakes,
        final_eval, max_white_advantage, max_black_advantage, game_sharpness
    """
    stockfish_path = os.path.join(raw_dir, "stockfish.csv")
    print(f"Extracting Stockfish features from {stockfish_path}...")
    df_sf = extract_stockfish_features(stockfish_path)
    # extract_stockfish_features() already returns the column as "event_id"
    # (reads row["Event"] and stores it as "event_id") — no rename needed.
    print(f"  Stockfish features extracted : {df_sf.shape}")
    return df_sf


# ─────────────────────────────────────────────────────────────────────────────
# 4. Merge: PGN + UCI + Stockfish
# ─────────────────────────────────────────────────────────────────────────────

def merge_kaggle_data(
    parsed_data_pgn: pd.DataFrame,
    parsed_data_uci: pd.DataFrame,
    df_sf: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inner-join merge of the three Kaggle sources on event_id.

    Parameters
    ----------
    parsed_data_pgn : ECO-enriched PGN DataFrame
    parsed_data_uci : UCI move sequences DataFrame
    df_sf           : Stockfish features DataFrame

    Returns
    -------
    df_kaggle : merged DataFrame ready for feature engineering
    """
    print("Merging PGN + UCI + Stockfish (inner joins on event_id)...")
    df_kaggle = merge_datasets(parsed_data_pgn, parsed_data_uci, df_sf)
    print(f"  Merged shape : {df_kaggle.shape}")
    return df_kaggle


# ─────────────────────────────────────────────────────────────────────────────
# 5. Save
# ─────────────────────────────────────────────────────────────────────────────

def save_kaggle_merged(df_kaggle: pd.DataFrame) -> None:
    """
    Persist the merged Kaggle dataset for Phase 4 feature engineering.

    Output
    ------
    data/intermediate/kaggle_merged.csv
    """
    out_path = os.path.join(intermediate_dir, "kaggle_merged.csv")
    df_kaggle.to_csv(out_path, index=False)
    print(f"\nSaved kaggle_merged.csv → {out_path}  {df_kaggle.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Load Phase-1 outputs
    parsed_data_uci, parsed_data_pgn = load_parsed_data()

    # 2. ECO matching (previously commented block #1 in data_loading.py)
    parsed_data_pgn = apply_eco_matching(parsed_data_pgn)

    # 3. Stockfish feature extraction (previously commented block #2)
    df_sf = load_stockfish_features()

    # 4. Three-way merge
    df_kaggle = merge_kaggle_data(parsed_data_pgn, parsed_data_uci, df_sf)

    # 5. Persist
    save_kaggle_merged(df_kaggle)

    print(f"\n✓ Phase 3 Complete — proceed to build_features.py")
    print(f"  kaggle_merged shape : {df_kaggle.shape}")