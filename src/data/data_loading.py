"""
Phase 1 — Data Loading
======================
Responsibilities:
  1. Download all raw data sources (Kaggle competition, Lichess ECO TSVs).
  2. Parse raw PGN / UCI files into structured DataFrames.
  3. Persist the parsed files to data/intermediate/ for downstream phases.

Outputs
-------
  data/intermediate/parsed_data_uci.csv   — UCI move sequences per game
  data/intermediate/parsed_data_pgn.csv   — game metadata + SAN move sequences
  data/external/eco_openings.csv          — Lichess ECO opening database
"""


import io
import re
import zipfile
import requests
import pandas as pd
from load_data import parse_uci, parse_pgn
import os
import tomllib

# ─────────────────────────────────────────────────────────────────────────────
# 1. Data Acquisition
# ─────────────────────────────────────────────────────────────────────────────


def acquire_data(raw_directory_path: str, eco_opening_path: str) -> None:
    """
    Download all raw data sources:
      - Kaggle competition files (finding-elo):
          data.pgn, data_uci.pgn, stockfish.csv
      - Kaggle dataset (online-chess-games):
          chess_games.csv  (Lichess export)
      - Lichess ECO opening database (a–e TSVs) →
          data/external/eco_openings.csv

    All zip archives are extracted recursively and removed after extraction.
    """
    os.environ["KAGGLE_API_TOKEN"] = "KGAT_145e8e3e445758440676eb29a8713200"
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    # ── Kaggle competition files ──────────────────────────────────────────────
    print("Downloading Kaggle competition files (finding-elo)...")
    api.competition_download_files("finding-elo", path=raw_directory_path)

    # ── Kaggle dataset: chess_games.csv ──────────────────────────────────────
    print("Downloading chess_games.csv from Kaggle dataset...")
    api.dataset_download_file(
        "mysarahmadbhat/online-chess-games",
        "chess_games.csv",
        path=raw_directory_path,
    )

    # ── Recursive unzip ───────────────────────────────────────────────────────
    def _unzip_all(directory: str) -> None:
        import traceback

        extracted = True
        while extracted:
            extracted = False
            for fname in os.listdir(directory):
                if not fname.endswith(".zip"):
                    continue
                fpath = os.path.join(directory, fname)
                print(f"  Extracting {fname}...")
                try:
                    with zipfile.ZipFile(fpath, "r") as z:
                        print(f"    Contents: {z.namelist()}")
                        z.extractall(directory)
                    print(f"    OK")
                except Exception as exc:
                    print(f"    FAILED: {exc}")
                    traceback.print_exc()
                try:
                    os.remove(fpath)
                except Exception:
                    pass
                extracted = True

    print("\nUnzipping downloaded files (including nested zips)...")
    _unzip_all(raw_directory_path)

    # ── Lichess ECO database ──────────────────────────────────────────────────
    print("\nDownloading Lichess ECO opening database from GitHub...")
    frames = []
    for letter in "abcde":
        url = f"https://raw.githubusercontent.com/lichess-org/chess-openings/master/{letter}.tsv"
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        df_letter = pd.read_csv(io.StringIO(resp.text), sep="\t")
        df_letter["eco_family"] = letter.upper()
        frames.append(df_letter)

    def _normalise_moves(pgn_str: str) -> str:
        if not isinstance(pgn_str, str):
            return ""
        cleaned = re.sub(r"\d+\.\s*", "", pgn_str)
        return " ".join(cleaned.split()).strip()

    df_eco = pd.concat(frames, ignore_index=True)
    df_eco["moves_normalised"] = df_eco["pgn"].apply(_normalise_moves)
    df_eco.to_csv(eco_opening_path, index=False)
    print(f"  Saved ECO database → {eco_opening_path}  ({len(df_eco):,} openings)")

    print("\nPhase 1a — Data Acquisition Complete.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Raw Parsing
# ─────────────────────────────────────────────────────────────────────────────


def parse_raw_files(raw_directory_path: str, intermediate_directory_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse the raw PGN and UCI files produced by acquire_data() into
    structured DataFrames and persist them to data/intermediate/.

    Returns
    -------
    parsed_data_uci : DataFrame  → intermediate/parsed_data_uci.csv
    parsed_data_pgn : DataFrame  → intermediate/parsed_data_pgn.csv
    """
    print("Parsing UCI file (data_uci.pgn)...")
    parsed_data_uci = parse_uci(os.path.join(raw_directory_path, "data_uci.pgn"))
    uci_path = os.path.join(intermediate_directory_path, "parsed_data_uci.csv")
    parsed_data_uci.to_csv(uci_path, index=False)
    print(f"  Saved → {uci_path}  {parsed_data_uci.shape}")

    print("\nParsing PGN file (data.pgn)...")
    parsed_data_pgn = parse_pgn(os.path.join(raw_directory_path, "data.pgn"))
    pgn_path = os.path.join(intermediate_directory_path, "parsed_data_pgn.csv")
    parsed_data_pgn.to_csv(pgn_path, index=False)
    print(f"  Saved → {pgn_path}  {parsed_data_pgn.shape}")

    print("\nPhase 1b — Raw Parsing Complete.")
    return parsed_data_uci, parsed_data_pgn


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with open("configs/config.toml", "rb") as f:
        config = tomllib.load(f)
    
    path_config = config["paths"]
    raw_directory = path_config["raw_data_dir"]
    intermediate_directory = path_config["intermediate_dir"]
    output_config = config["output"]
    eco_opening_path = output_config["output_eco_csv"]

    acquire_data(raw_directory, eco_opening_path)
    parse_raw_files(raw_directory, intermediate_directory)
    print("\n✓ Phase 1 Complete — proceed to validate_data.py")
