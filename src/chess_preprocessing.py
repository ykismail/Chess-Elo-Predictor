"""
chess_preprocessing.py
=======================
CMPS344 Applied Data Science — Phase 2
Data acquisition, parsing, and preprocessing functions for chess game datasets.

Sources
-------
- data.pgn         : 50,000 chess games in standard PGN notation (Kaggle)
- data_uci.pgn     : Same games with moves in UCI coordinate notation (Kaggle)
- stockfish.csv    : Per-game Stockfish centipawn evaluations (Kaggle)
- chess_games.csv  : Lichess casual/club games with ECO codes (Lichess/Kaggle)
- eco_openings.csv : Lichess ECO opening database (downloaded from GitHub)

Usage
-----
    from chess_preprocessing import (
        download_eco_database,
        parse_pgn,
        build_eco_lookup,
        match_eco,
        parse_uci,
        extract_stockfish_features,
        merge_datasets,
        engineer_features,
        load_lichess,
        integrate_datasets,
        validation_report,
        save_dataset,
    )
"""

import os
import re
import io
import requests
import numpy as np
import pandas as pd
import chess
import chess.pgn


# ══════════════════════════════════════════════════════════════════════════════
# 1. ECO OPENING DATABASE
# ══════════════════════════════════════════════════════════════════════════════

def download_eco_database(save_path: str = "eco_openings.csv") -> pd.DataFrame:
    """
    Download the Lichess ECO opening database from GitHub and save locally.
    Covers all ECO codes A00-E99 across 5 TSV files (a.tsv through e.tsv).
    Loads from disk if already downloaded.

    Parameters
    ----------
    save_path : str — local path to save/load the CSV

    Returns
    -------
    pd.DataFrame with columns: eco, name, pgn, uci, epd, eco_family, moves_normalised

    Source
    ------
    https://github.com/lichess-org/chess-openings (public domain)
    """
    if os.path.exists(save_path):
        print(f"Found cached '{save_path}', loading from disk...")
        df_eco = pd.read_csv(save_path)
        df_eco["moves_normalised"] = df_eco["moves_normalised"].fillna("")
        print(f"Loaded {len(df_eco):,} ECO entries.")
        return df_eco

    url_template = "https://raw.githubusercontent.com/lichess-org/chess-openings/master/{}.tsv"
    frames = []

    for letter in list("abcde"):
        url = url_template.format(letter)
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            raw = resp.text
            print(f"  {letter}.tsv — HTTP {resp.status_code}, {len(raw)} chars")
            print(f"  First line : {repr(raw.splitlines()[0])}")
            print(f"  Second line: {repr(raw.splitlines()[1]) if len(raw.splitlines()) > 1 else 'N/A'}")

            df_letter = pd.read_csv(io.StringIO(raw), sep="\t")
            print(f"  Columns detected: {list(df_letter.columns)}")
            print(f"  Rows: {len(df_letter)}")
            df_letter["eco_family"] = letter.upper()
            frames.append(df_letter)
            print(f"  ✓ Appended successfully\n")
        except Exception as e:
            print(f"  ✗ {letter}.tsv FAILED: {type(e).__name__}: {e}\n")

    if not frames:
        raise RuntimeError(
            "All ECO downloads failed — check the debug output above.\n"
            "Common fixes:\n"
            "  - HTTP 200 but 0 rows: TSV format changed, check column names above\n"
            "  - Connection error: check internet / firewall\n"
            "  - HTTP 403/404: URL changed, check github.com/lichess-org/chess-openings"
        )

    df_eco = pd.concat(frames, ignore_index=True)

    # Standardise column names
    col_map = {}
    for col in df_eco.columns:
        c = col.lower().strip()
        if c == "eco":                      col_map[col] = "eco"
        elif c == "name":                   col_map[col] = "name"
        elif c in ("pgn", "moves", "san"):  col_map[col] = "pgn"
        elif c == "uci":                    col_map[col] = "uci"
        elif c == "epd":                    col_map[col] = "epd"
    df_eco = df_eco.rename(columns=col_map)

    if "pgn" not in df_eco.columns:
        raise RuntimeError(
            f"Could not find a moves/pgn column. Found: {list(df_eco.columns)}\n"
            "Update col_map in download_eco_database() to match."
        )

    df_eco["moves_normalised"] = df_eco["pgn"].apply(_normalise_moves)
    df_eco.to_csv(save_path, index=False)

    print(f"Total ECO entries : {len(df_eco):,}")
    print(f"ECO family counts :\n{df_eco['eco_family'].value_counts().sort_index().to_string()}")
    print(f"Saved to '{save_path}'")
    return df_eco


def _normalise_moves(pgn_str: str) -> str:
    """Strip move numbers (e.g. '1.', '12.') and normalise whitespace."""
    if not isinstance(pgn_str, str):
        return ""
    cleaned = re.sub(r"\d+\.\s*", "", pgn_str)
    return " ".join(cleaned.split()).strip()


def build_eco_lookup(df_eco: pd.DataFrame) -> dict:
    """
    Build a prefix lookup dictionary from the ECO database.

    Parameters
    ----------
    df_eco : pd.DataFrame — output of download_eco_database()

    Returns
    -------
    dict mapping normalised move prefix → (eco_code, opening_name, eco_family)
    """
    lookup = {}
    for _, row in df_eco.iterrows():
        key = str(row["moves_normalised"]).strip()
        if key:
            lookup[key] = (row["eco"], row["name"], row["eco_family"])
    return lookup


def match_eco(moves_san: str, lookup: dict, max_depth: int = 10) -> tuple:
    """
    Match a game's moves against the ECO lookup using longest-prefix matching.

    Parameters
    ----------
    moves_san : str   — full SAN move sequence (from parse_pgn)
    lookup    : dict  — from build_eco_lookup()
    max_depth : int   — maximum number of half-moves to try (default 10)

    Returns
    -------
    (eco_code, opening_name, eco_family) or ('Unknown', 'Unknown', 'Unknown')
    """
    if not isinstance(moves_san, str):
        return ("Unknown", "Unknown", "Unknown")

    clean = re.sub(r"[+#!?]", "", moves_san).strip()
    tokens = clean.split()

    for depth in range(min(max_depth, len(tokens)), 0, -1):
        prefix = " ".join(tokens[:depth])
        if prefix in lookup:
            return lookup[prefix]

    return ("Unknown", "Unknown", "Unknown")


# ══════════════════════════════════════════════════════════════════════════════
# 2. PGN PARSING (python-chess)
# ══════════════════════════════════════════════════════════════════════════════

def parse_pgn(filepath: str) -> pd.DataFrame:
    """
    Parse a PGN file using python-chess and return a DataFrame with one row per game.
    Games with missing or invalid Elo values are skipped.

    Parameters
    ----------
    filepath : str — path to .pgn file

    Returns
    -------
    pd.DataFrame with columns:
        event_id, white_elo, black_elo, result, num_moves,
        white_castled, black_castled, white_castle_side, black_castle_side,
        num_captures, termination, moves_san
    """
    records = []

    with open(filepath, "r", encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            headers = game.headers
            try:
                white_elo = int(headers.get("WhiteElo", ""))
                black_elo = int(headers.get("BlackElo", ""))
            except ValueError:
                continue  # skip games with missing/invalid Elo

            result   = headers.get("Result", "*")
            event_id = int(headers.get("Event", 0))

            board = game.board()
            moves_san         = []
            num_captures      = 0
            white_castled     = False
            black_castled     = False
            white_castle_side = "none"
            black_castle_side = "none"

            for move in game.mainline_moves():
                san = board.san(move)
                moves_san.append(san)

                if board.is_capture(move):
                    num_captures += 1

                if board.is_castling(move):
                    side = "kingside" if board.is_kingside_castling(move) else "queenside"
                    if board.turn == chess.WHITE:
                        white_castled     = True
                        white_castle_side = side
                    else:
                        black_castled     = True
                        black_castle_side = side

                board.push(move)

            num_moves = len(moves_san) // 2

            if board.is_checkmate():
                termination = "checkmate"
            elif result == "1/2-1/2":
                termination = "draw"
            elif result in ("1-0", "0-1"):
                termination = "resignation"
            else:
                termination = "unknown"

            records.append({
                "event_id":          event_id,
                "white_elo":         white_elo,
                "black_elo":         black_elo,
                "result":            result,
                "num_moves":         num_moves,
                "white_castled":     white_castled,
                "black_castled":     black_castled,
                "white_castle_side": white_castle_side,
                "black_castle_side": black_castle_side,
                "num_captures":      num_captures,
                "termination":       termination,
                "moves_san":         " ".join(moves_san),
            })

    df = pd.DataFrame(records)
    print(f"Parsed {len(df):,} games from {filepath}")
    print(f"\nResult distribution:\n{df['result'].value_counts().to_string()}")
    print(f"\nTermination distribution:\n{df['termination'].value_counts().to_string()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. UCI PARSING (regex)
# ══════════════════════════════════════════════════════════════════════════════

def parse_uci(filepath: str) -> pd.DataFrame:
    """
    Parse a UCI-format PGN file and return a DataFrame with one row per game.

    Parameters
    ----------
    filepath : str — path to UCI .pgn file

    Returns
    -------
    pd.DataFrame with columns: event_id, moves_uci
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    game_blocks = re.split(r"\n(?=\[Event )", content.strip())
    records = []

    for block in game_blocks:
        event_match = re.search(r'\[Event "([^"]+)"\]', block)
        if not event_match:
            continue

        move_lines = [
            line for line in block.split("\n")
            if line.strip() and not line.startswith("[")
        ]
        moves_raw   = " ".join(move_lines)
        moves_clean = re.sub(r"\s*(1-0|0-1|1/2-1/2|\*)\s*$", "", moves_raw).strip()

        records.append({
            "event_id":  int(event_match.group(1)),
            "moves_uci": moves_clean,
        })

    df = pd.DataFrame(records)
    print(f"Parsed {len(df):,} games from {filepath}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. STOCKFISH FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_stockfish_features(filepath: str) -> pd.DataFrame:
    """
    Parse stockfish.csv and compute per-game evaluation features.

    Parameters
    ----------
    filepath : str — path to stockfish.csv

    Returns
    -------
    pd.DataFrame with columns:
        event_id, total_half_moves, white_acl, black_acl,
        white_blunders, black_blunders, white_mistakes, black_mistakes,
        final_eval, max_white_advantage, max_black_advantage, game_sharpness
    """
    df_sf = pd.read_csv(filepath)
    records = []

    for _, row in df_sf.iterrows():
        try:
            scores = list(map(int, str(row["MoveScores"]).split()))
        except (ValueError, AttributeError):
            continue

        if len(scores) < 2:
            continue

        white_scores = scores[0::2]
        black_scores = scores[1::2]

        records.append({
            "event_id":            row["Event"],
            "total_half_moves":    len(scores),
            "white_acl":           _avg_centipawn_loss(white_scores, is_white=True),
            "black_acl":           _avg_centipawn_loss(black_scores, is_white=False),
            "white_blunders":      _count_errors(white_scores, True,  threshold=100),
            "black_blunders":      _count_errors(black_scores, False, threshold=100),
            "white_mistakes":      _count_errors(white_scores, True,  threshold=50) -
                                   _count_errors(white_scores, True,  threshold=100),
            "black_mistakes":      _count_errors(black_scores, False, threshold=50) -
                                   _count_errors(black_scores, False, threshold=100),
            "final_eval":          scores[-1],
            "max_white_advantage": max(scores),
            "max_black_advantage": min(scores),
            "game_sharpness":      round(float(np.std(scores)), 2),
        })

    df = pd.DataFrame(records)
    print(f"Extracted Stockfish features for {len(df):,} games")
    print(df[["white_acl", "black_acl", "white_blunders", "black_blunders"]].describe().round(2).to_string())
    return df


def _avg_centipawn_loss(scores_list: list, is_white: bool) -> float:
    """Compute average centipawn loss per move for one player."""
    losses = []
    for i in range(1, len(scores_list)):
        prev, curr = scores_list[i - 1], scores_list[i]
        loss = (prev - curr) if is_white else (curr - prev)
        if loss > 0:
            losses.append(loss)
    return round(np.mean(losses), 2) if losses else 0.0


def _count_errors(scores_list: list, is_white: bool, threshold: int) -> int:
    """Count moves where centipawn loss exceeds a threshold."""
    count = 0
    for i in range(1, len(scores_list)):
        prev, curr = scores_list[i - 1], scores_list[i]
        loss = (prev - curr) if is_white else (curr - prev)
        if loss >= threshold:
            count += 1
    return count


# ══════════════════════════════════════════════════════════════════════════════
# 5. MERGING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def merge_datasets(
    df_pgn: pd.DataFrame,
    df_uci: pd.DataFrame,
    df_sf: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge PGN, UCI, and Stockfish DataFrames on event_id using inner joins.

    Parameters
    ----------
    df_pgn : pd.DataFrame — from parse_pgn()
    df_uci : pd.DataFrame — from parse_uci()
    df_sf  : pd.DataFrame — from extract_stockfish_features()

    Returns
    -------
    Merged pd.DataFrame with all features from all three sources
    """
    df_uci_slim = df_uci[["event_id", "moves_uci"]]
    df = (
        df_pgn
        .merge(df_uci_slim, on="event_id", how="inner")
        .merge(df_sf,       on="event_id", how="inner")
    )
    print(f"Merged DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features and encode target variables.

    New columns added:
        elo_gap          : white_elo - black_elo
        avg_elo          : mean of both Elos
        elo_bucket_white : skill tier of White player
        elo_bucket_black : skill tier of Black player
        acl_gap          : white_acl - black_acl
        winner_binary    : 1=White wins, 0=Black wins, NaN=Draw
        winner_multiclass: 0=Black, 1=Draw, 2=White

    Parameters
    ----------
    df : pd.DataFrame — from merge_datasets()

    Returns
    -------
    pd.DataFrame with additional engineered columns
    """
    df = df.copy()

    df["elo_gap"] = df["white_elo"] - df["black_elo"]
    df["avg_elo"] = ((df["white_elo"] + df["black_elo"]) / 2).round(0)

    elo_bins   = [0, 1000, 1500, 2000, 2500, 9999]
    elo_labels = ["Beginner", "Intermediate", "Advanced", "Expert", "Master"]

    df["elo_bucket_white"] = pd.cut(df["white_elo"], bins=elo_bins, labels=elo_labels, right=False)
    df["elo_bucket_black"] = pd.cut(df["black_elo"], bins=elo_bins, labels=elo_labels, right=False)

    df["acl_gap"] = (df["white_acl"] - df["black_acl"]).round(2)

    df["winner_binary"]     = df["result"].map({"1-0": 1, "0-1": 0})
    df["winner_multiclass"] = df["result"].map({"0-1": 0, "1/2-1/2": 1, "1-0": 2})

    print("Engineered features added.")
    print(f"\nElo bucket distribution (White):\n{df['elo_bucket_white'].value_counts().to_string()}")
    print(f"\nWinner (multiclass) distribution:\n{df['winner_multiclass'].value_counts().to_string()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 6. LICHESS CSV INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════

def load_lichess(filepath: str) -> pd.DataFrame:
    """
    Load chess_games.csv (Lichess) and harmonise columns to match the PGN pipeline schema.

    Parameters
    ----------
    filepath : str — path to chess_games.csv

    Returns
    -------
    pd.DataFrame with unified column names, ready to concatenate with PGN pipeline output
    """
    df_l = pd.read_csv(filepath)
    print(f"Loaded {len(df_l):,} games from {filepath}")

    out = pd.DataFrame()

    out["event_id"]         = df_l["game_id"]
    out["source"]           = "lichess_csv"
    out["white_elo"]        = df_l["white_rating"]
    out["black_elo"]        = df_l["black_rating"]
    out["num_moves"]        = df_l["turns"] // 2
    out["total_half_moves"] = df_l["turns"]

    elo_bins   = [0, 1000, 1500, 2000, 2500, 9999]
    elo_labels = ["Beginner", "Intermediate", "Advanced", "Expert", "Master"]
    out["elo_bucket_white"] = pd.cut(df_l["white_rating"], bins=elo_bins, labels=elo_labels, right=False)
    out["elo_bucket_black"] = pd.cut(df_l["black_rating"], bins=elo_bins, labels=elo_labels, right=False)

    termination_map = {
        "Resign":      "resignation",
        "Mate":        "checkmate",
        "Out of Time": "timeout",
        "Draw":        "draw",
    }
    out["termination"]  = df_l["victory_status"].map(termination_map).fillna("unknown")
    out["eco_code"]     = df_l["opening_code"]
    out["opening_name"] = df_l["opening_fullname"]
    out["eco_family"]   = df_l["opening_code"].str[0].str.upper()
    out["moves_san"]    = df_l["moves"]
    out["moves_uci"]    = np.nan
    out["elo_gap"]      = df_l["white_rating"] - df_l["black_rating"]
    out["avg_elo"]      = ((df_l["white_rating"] + df_l["black_rating"]) / 2).round(0)

    out["winner_multiclass"] = df_l["winner"].map({"Black": 0, "Draw": 1, "White": 2})
    out["winner_binary"]     = df_l["winner"].map({"White": 1, "Black": 0})

    stockfish_cols = [
        "white_acl", "black_acl", "white_blunders", "black_blunders",
        "white_mistakes", "black_mistakes", "final_eval",
        "max_white_advantage", "max_black_advantage", "game_sharpness", "acl_gap"
    ]
    for col in stockfish_cols:
        out[col] = np.nan

    out["white_castled"]     = np.nan
    out["black_castled"]     = np.nan
    out["white_castle_side"] = np.nan
    out["black_castle_side"] = np.nan
    out["num_captures"]      = np.nan
    out["has_stockfish"]     = False

    print(f"Harmonised columns: {list(out.columns)}")
    return out


def integrate_datasets(df_pgn_pipeline: pd.DataFrame, lichess_path: str) -> pd.DataFrame:
    """
    Concatenate the PGN pipeline output with the Lichess CSV dataset.

    Parameters
    ----------
    df_pgn_pipeline : pd.DataFrame — output of engineer_features()
    lichess_path    : str          — path to chess_games.csv

    Returns
    -------
    Unified pd.DataFrame with all games from both sources
    """
    df_pgn_tagged = df_pgn_pipeline.copy()
    df_pgn_tagged["source"]        = "kaggle_pgn"
    df_pgn_tagged["has_stockfish"] = True

    df_lichess   = load_lichess(lichess_path)
    df_combined  = pd.concat([df_pgn_tagged, df_lichess], ignore_index=True, sort=False)
    df_combined["event_id"] = range(1, len(df_combined) + 1)

    print(f"\n── Integration Summary ──────────────────────────────────")
    print(f"PGN pipeline rows  : {len(df_pgn_tagged):,}")
    print(f"Lichess CSV rows   : {len(df_lichess):,}")
    print(f"Combined total     : {len(df_combined):,}")
    print(f"\nSource breakdown:\n{df_combined['source'].value_counts().to_string()}")
    print(f"\nhas_stockfish breakdown:\n{df_combined['has_stockfish'].value_counts().to_string()}")
    print(f"\nwinner_multiclass distribution:")
    label_map = {0: "Black wins", 1: "Draw", 2: "White wins"}
    print(df_combined["winner_multiclass"].value_counts().rename(label_map).to_string())
    return df_combined


# ══════════════════════════════════════════════════════════════════════════════
# 7. VALIDATION & SAVING
# ══════════════════════════════════════════════════════════════════════════════

def validation_report(df: pd.DataFrame) -> None:
    """
    Print a comprehensive data validation report.

    Covers: shape, data types, missing values, duplicates,
    numeric summary, and target variable class distributions.

    Parameters
    ----------
    df : pd.DataFrame — any stage of the pipeline
    """
    print("=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)

    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    print("\n── Data Types ──────────────────────────────────────────────")
    print(df.dtypes.to_string())

    print("\n── Missing Values ──────────────────────────────────────────")
    missing     = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df  = pd.DataFrame({"count": missing, "pct": missing_pct})
    missing_df  = missing_df[missing_df["count"] > 0]
    print("No missing values found." if missing_df.empty else missing_df.to_string())

    print(f"\n── Duplicates ──────────────────────────────────────────────")
    dup_count = df.duplicated(subset=["event_id"]).sum() if "event_id" in df.columns else "N/A"
    print(f"Duplicate event_id rows: {dup_count}")

    print("\n── Numeric Summary ─────────────────────────────────────────")
    numeric_cols = [c for c in ["white_elo", "black_elo", "num_moves", "white_acl",
                    "black_acl", "white_blunders", "black_blunders",
                    "elo_gap", "avg_elo", "game_sharpness"] if c in df.columns]
    print(df[numeric_cols].describe().round(2).to_string())

    if "winner_multiclass" in df.columns:
        print("\n── Target: winner_multiclass ───────────────────────────────")
        label_map = {0: "Black wins", 1: "Draw", 2: "White wins"}
        print(df["winner_multiclass"].value_counts().rename(label_map).to_string())

    if "elo_bucket_white" in df.columns:
        print("\n── Target: elo_bucket_white ────────────────────────────────")
        print(df["elo_bucket_white"].value_counts().to_string())

    print("\n" + "=" * 60)


def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    """
    Save a DataFrame to CSV, dropping raw move text columns.

    Parameters
    ----------
    df          : pd.DataFrame — final engineered DataFrame
    output_path : str          — destination CSV path
    """
    cols_to_drop = [c for c in ["moves_pgn", "moves_uci", "moves_san"] if c in df.columns]
    df_save = df.drop(columns=cols_to_drop)
    df_save.to_csv(output_path, index=False)
    print(f"Saved {len(df_save):,} rows × {df_save.shape[1]} columns to '{output_path}'")
    print(f"\nFinal columns:\n{list(df_save.columns)}")
