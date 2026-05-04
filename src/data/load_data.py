import requests
import io
import pandas as pd
import re
import chess.pgn
import numpy as np


def download_eco_database(save_path: str = "eco_openings.csv") -> pd.DataFrame:
    """
    Download the Lichess ECO opening database from GitHub and save locally.
    Covers all ECO codes A00-E99 across 5 TSV files (a.tsv through e.tsv).

    Source: https://github.com/lichess-org/chess-openings (public domain)
    Citation: lichess-org/chess-openings, github.com/lichess-org/chess-openings
    """
    url_template = (
        "https://raw.githubusercontent.com/lichess-org/chess-openings/master/{}.tsv"
    )
    frames = []

    for letter in list("abcde"):
        url = url_template.format(letter)
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()

            # ── DEBUG: print raw content so we can see what we're getting ────
            raw = resp.text
            print(f"  {letter}.tsv — HTTP {resp.status_code}, {len(raw)} chars")
            print(f"  First line: {repr(raw.splitlines()[0])}")
            print(
                f"  Second line: {repr(raw.splitlines()[1]) if len(raw.splitlines()) > 1 else 'N/A'}"
            )

            # ── Auto-detect columns from header row ──────────────────────────
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
            "All ECO downloads failed — check the debug output above for clues.\n"
            "Common fixes:\n"
            "  - If HTTP 200 but 0 rows: the TSV format changed, check the column names printed above\n"
            "  - If connection error: check your internet / firewall\n"
            "  - If HTTP 403/404: GitHub URL changed, check github.com/lichess-org/chess-openings"
        )

    df_eco = pd.concat(frames, ignore_index=True)

    # ── Standardise column names regardless of what GitHub returns ───────────
    # Expected columns: eco, name, pgn (or moves), uci, epd
    col_map = {}
    for col in df_eco.columns:
        col_lower = col.lower().strip()
        if col_lower == "eco":
            col_map[col] = "eco"
        elif col_lower == "name":
            col_map[col] = "name"
        elif col_lower in ("pgn", "moves", "san"):
            col_map[col] = "pgn"
        elif col_lower == "uci":
            col_map[col] = "uci"
        elif col_lower == "epd":
            col_map[col] = "epd"
    df_eco = df_eco.rename(columns=col_map)

    if "pgn" not in df_eco.columns:
        raise RuntimeError(
            f"Could not find a moves/pgn column. Columns found: {list(df_eco.columns)}\n"
            "Update the col_map dict above to match."
        )

    # ── Normalise move sequences for prefix matching ──────────────────────────
    def normalise_moves(pgn_str: str) -> str:
        if not isinstance(pgn_str, str):
            return ""
        cleaned = re.sub(r"\d+\.\s*", "", pgn_str)
        return " ".join(cleaned.split()).strip()

    df_eco["moves_normalised"] = df_eco["pgn"].apply(normalise_moves)

    # Ensure the directory exists before saving
    import os

    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    df_eco.to_csv(save_path, index=False)

    print(f"Total ECO entries : {len(df_eco):,}")
    print(
        f"ECO family counts :\n{df_eco['eco_family'].value_counts().sort_index().to_string()}"
    )
    print(f"Saved to '{save_path}'")
    return df_eco


def parse_pgn(filepath: str) -> pd.DataFrame:
    """
    Parse a PGN file using python-chess and return a DataFrame with one row per game.

    Extracted columns:
        event_id         : int   — game identifier
        white_elo        : int   — White player Elo
        black_elo        : int   — Black player Elo
        result           : str   — '1-0', '0-1', or '1/2-1/2'
        num_moves        : int   — number of full moves played
        white_castled    : bool  — whether White castled
        black_castled    : bool  — whether Black castled
        white_castle_side: str   — 'kingside', 'queenside', or 'none'
        black_castle_side: str   — 'kingside', 'queenside', or 'none'
        num_captures     : int   — total number of captures
        termination      : str   — how the game ended
        moves_san        : str   — full move sequence in SAN (for ECO matching)
    """
    records = []

    with open(filepath, "r", encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            headers = game.headers

            # ── Skip games missing essential fields ──────────────────────────
            try:
                white_elo = int(headers.get("WhiteElo", ""))
                black_elo = int(headers.get("BlackElo", ""))
            except ValueError:
                continue

            result = headers.get("Result", "*")
            event_id = int(headers.get("Event", 0))

            # ── Walk through moves to extract board-level features ────────────
            board = game.board()
            moves_san = []
            num_captures = 0
            white_castled = False
            black_castled = False
            white_castle_side = "none"
            black_castle_side = "none"

            for move in game.mainline_moves():
                san = board.san(move)
                moves_san.append(san)

                # Detect captures
                if board.is_capture(move):
                    num_captures += 1

                # Detect castling
                if board.is_castling(move):
                    is_kingside = board.is_kingside_castling(move)
                    side = "kingside" if is_kingside else "queenside"
                    if board.turn == chess.WHITE:
                        white_castled = True
                        white_castle_side = side
                    else:
                        black_castled = True
                        black_castle_side = side

                board.push(move)

            num_moves = len(moves_san) // 2  # full moves
            moves_text = " ".join(moves_san)

            # ── Infer termination ─────────────────────────────────────────────
            if board.is_checkmate():
                termination = "checkmate"
            elif result in ("1/2-1/2",):
                termination = "draw"
            elif result in ("1-0", "0-1"):
                termination = "resignation"
            else:
                termination = "unknown"

            records.append(
                {
                    "event_id": event_id,
                    "white_elo": white_elo,
                    "black_elo": black_elo,
                    "result": result,
                    "num_moves": num_moves,
                    "white_castled": white_castled,
                    "black_castled": black_castled,
                    "white_castle_side": white_castle_side,
                    "black_castle_side": black_castle_side,
                    "num_captures": num_captures,
                    "termination": termination,
                    "moves_san": moves_text,
                }
            )

    df = pd.DataFrame(records)
    print(f"Parsed {len(df):,} games from {filepath}")
    print(f"\nResult distribution:\n{df['result'].value_counts().to_string()}")
    print(
        f"\nTermination distribution:\n{df['termination'].value_counts().to_string()}"
    )
    return df


def build_eco_lookup(df_eco: pd.DataFrame) -> dict:
    """
    Build a prefix lookup dictionary from the ECO database.
    Maps normalised move prefix → (eco_code, opening_name, eco_family).

    Parameters
    ----------
    df_eco : DataFrame from download_eco_database()

    Returns
    -------
    dict keyed by normalised move string
    """
    lookup = {}
    for _, row in df_eco.iterrows():
        key = row["moves_normalised"].strip()
        if key:
            lookup[key] = (row["eco"], row["name"], row["eco_family"])
    return lookup


def match_eco(moves_san: str, lookup: dict, max_depth: int = 10) -> tuple:
    """
    Match a game's moves against the ECO lookup using longest-prefix matching.

    Tries progressively shorter prefixes until a match is found.

    Parameters
    ----------
    moves_san : str   — full SAN move sequence from parse_pgn()
    lookup    : dict  — from build_eco_lookup()
    max_depth : int   — maximum number of half-moves to try

    Returns
    -------
    (eco_code, opening_name, eco_family) or ('Unknown', 'Unknown', 'Unknown')
    """
    if not isinstance(moves_san, str):
        return ("Unknown", "Unknown", "Unknown")

    # Remove annotations like '!', '?', '+', '#'
    clean = re.sub(r"[+#!?]", "", moves_san).strip()
    tokens = clean.split()

    # Try longest prefix first, shrink until a match is found
    for depth in range(min(max_depth, len(tokens)), 0, -1):
        prefix = " ".join(tokens[:depth])
        if prefix in lookup:
            return lookup[prefix]

    return ("Unknown", "Unknown", "Unknown")


def parse_uci(filepath: str) -> pd.DataFrame:
    """
    Parse a UCI-format PGN file and return a DataFrame with one row per game.

    Extracted columns:
        event_id  : int — game identifier (links back to data.pgn)
        moves_uci : str — full move sequence in UCI coordinate notation
                          e.g. 'e2e4 e7e5 g1f3 b8c6'
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
            line
            for line in block.split("\n")
            if line.strip() and not line.startswith("[")
        ]
        moves_raw = " ".join(move_lines)
        moves_clean = re.sub(r"\s*(1-0|0-1|1/2-1/2|\*)\s*$", "", moves_raw).strip()

        records.append(
            {
                "event_id": int(event_match.group(1)),
                "moves_uci": moves_clean,
            }
        )

    df = pd.DataFrame(records)
    print(f"Parsed {len(df):,} games from {filepath}")
    return df


def extract_stockfish_features(filepath: str) -> pd.DataFrame:
    """
    Parse stockfish.csv and compute per-game evaluation features.

    Parameters
    ----------
    filepath : str — path to stockfish.csv

    Returns
    -------
    pd.DataFrame with one row per game and derived evaluation features
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

        # ── Split scores by player ───────────────────────────────────────────
        # After White's move: even indices (0, 2, 4, ...)
        # After Black's move: odd indices  (1, 3, 5, ...)
        white_scores = scores[0::2]  # evaluation after White moves
        black_scores = scores[1::2]  # evaluation after Black moves

        # ── Helper: average centipawn loss ───────────────────────────────────
        def avg_centipawn_loss(scores_list, is_white: bool) -> float:
            losses = []
            for i in range(1, len(scores_list)):
                prev, curr = scores_list[i - 1], scores_list[i]
                # White wants score to go up; Black wants it to go down
                loss = (prev - curr) if is_white else (curr - prev)
                if loss > 0:
                    losses.append(loss)
            return round(np.mean(losses), 2) if losses else 0.0

        # ── Helper: count errors above a centipawn threshold ─────────────────
        def count_errors(scores_list, is_white: bool, threshold: int) -> int:
            count = 0
            for i in range(1, len(scores_list)):
                prev, curr = scores_list[i - 1], scores_list[i]
                loss = (prev - curr) if is_white else (curr - prev)
                if loss >= threshold:
                    count += 1
            return count

        records.append(
            {
                "event_id": row["Event"],
                "total_half_moves": len(scores),
                "white_acl": avg_centipawn_loss(white_scores, is_white=True),
                "black_acl": avg_centipawn_loss(black_scores, is_white=False),
                "white_blunders": count_errors(white_scores, True, threshold=100),
                "black_blunders": count_errors(black_scores, False, threshold=100),
                "white_mistakes": count_errors(white_scores, True, threshold=50)
                - count_errors(white_scores, True, threshold=100),
                "black_mistakes": count_errors(black_scores, False, threshold=50)
                - count_errors(black_scores, False, threshold=100),
                "final_eval": scores[-1],
                "max_white_advantage": max(scores),
                "max_black_advantage": min(scores),
                "game_sharpness": round(float(np.std(scores)), 2),
            }
        )

    df = pd.DataFrame(records)
    print(f"Extracted Stockfish features for {len(df):,} games")
    print(
        df[["white_acl", "black_acl", "white_blunders", "black_blunders"]]
        .describe()
        .round(2)
        .to_string()
    )
    return df


def merge_datasets(
    df_pgn: pd.DataFrame, df_uci: pd.DataFrame, df_sf: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge the three parsed DataFrames on event_id using inner joins.

    Sources:
        df_pgn : game metadata + ECO opening info  (from data.pgn + lichess ECO)
        df_uci : UCI move sequences                 (from data_uci.pgn)
        df_sf  : Stockfish evaluation features      (from stockfish.csv)

    Returns
    -------
    Merged DataFrame with all features from all three sources
    """
    # Keep only moves_uci from UCI file (metadata already in df_pgn)
    df_uci_slim = df_uci[["event_id", "moves_uci"]]

    df = df_pgn.merge(df_uci_slim, on="event_id", how="inner").merge(
        df_sf, on="event_id", how="inner"
    )

    print(f"Merged DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df


def load_lichess(filepath: str, stockfish_path: str) -> pd.DataFrame:
    """
    Load chess_games.csv (Lichess) and harmonise columns to match the PGN pipeline schema.
    Now also loads pre-computed Stockfish features from lichess_stockfish.csv so that
    all ~43k games in the merged dataset have full engine evaluation features.

    Parameters
    ----------
    filepath       : str — path to chess_games.csv
    stockfish_path : str — path to lichess_stockfish.csv

    Returns
    -------
    DataFrame with unified column names and full Stockfish features,
    ready to concatenate with the PGN pipeline output.
    """
    df_l = pd.read_csv(filepath)
    print(f"Loaded {len(df_l):,} games from '{filepath}'")

    # ── Load and extract Stockfish features ──────────────────────────────────
    print(f"Loading Stockfish features from '{stockfish_path}'...")
    df_sf = extract_stockfish_features(stockfish_path)
    # Rename Event → game_id so we can merge on game_id
    df_sf = df_sf.rename(columns={"event_id": "game_id"})
    print(f"Extracted Stockfish features for {len(df_sf):,} games")

    # ── Merge Lichess games with their Stockfish features ────────────────────
    df_l = df_l.merge(df_sf, on="game_id", how="left")
    missing_sf = df_l["white_acl"].isna().sum()
    if missing_sf > 0:
        print(f"Warning: {missing_sf} games have no Stockfish data (will be NaN)")

    out = pd.DataFrame()

    # ── IDs & source ─────────────────────────────────────────────────────────
    out["event_id"] = df_l["game_id"]
    out["source"] = "lichess_csv"

    # ── Elo ───────────────────────────────────────────────────────────────────
    out["white_elo"] = df_l["white_rating"]
    out["black_elo"] = df_l["black_rating"]

    # ── Game length ───────────────────────────────────────────────────────────
    out["num_moves"] = df_l["turns"] // 2
    out["total_half_moves"] = df_l["turns"]

    # ── Termination ───────────────────────────────────────────────────────────
    termination_map = {
        "Resign": "resignation",
        "Mate": "checkmate",
        "Out of Time": "timeout",
        "Draw": "draw",
    }
    out["termination"] = df_l["victory_status"].map(termination_map).fillna("unknown")

    # ── Opening info ──────────────────────────────────────────────────────────
    out["eco_code"] = df_l["opening_code"]
    out["opening_name"] = df_l["opening_fullname"]
    out["eco_family"] = df_l["opening_code"].str[0].str.upper()

    # ── Move sequences ────────────────────────────────────────────────────────
    out["moves_san"] = df_l["moves"]
    out["moves_uci"] = np.nan

    # ── Elo classification target ────────────────────────────────────────────
    elo_bins = [0, 1000, 1500, 2000, 2500, 9999]
    elo_labels = ["Beginner", "Intermediate", "Advanced", "Expert", "Master"]
    out["elo_bucket_white"] = pd.cut(
        df_l["white_rating"], bins=elo_bins, labels=elo_labels, right=False
    )
    out["elo_bucket_black"] = pd.cut(
        df_l["black_rating"], bins=elo_bins, labels=elo_labels, right=False
    )

    # ── Ordinal encoding of target ────────────────────────────────────────────
    # Beginner=0, Intermediate=1, Advanced=2, Expert=3, Master=4
    ordinal_map = {
        "Beginner": 0,
        "Intermediate": 1,
        "Advanced": 2,
        "Expert": 3,
        "Master": 4,
    }
    out["elo_bucket_white_categorical"] = out["elo_bucket_white"].map(ordinal_map)
    out["elo_bucket_black_categorical"] = out["elo_bucket_black"].map(ordinal_map)

    # ── Winner target ─────────────────────────────────────────────────────────
    out["winner_multiclass"] = df_l["winner"].map({"Black": 0, "Draw": 1, "White": 2})
    out["winner_binary"] = df_l["winner"].map({"White": 1, "Black": 0})

    # ── Stockfish features (now populated from lichess_stockfish.csv) ─────────
    stockfish_cols = [
        "white_acl",
        "black_acl",
        "white_blunders",
        "black_blunders",
        "white_mistakes",
        "black_mistakes",
        "final_eval",
        "max_white_advantage",
        "max_black_advantage",
        "game_sharpness",
    ]
    for col in stockfish_cols:
        out[col] = df_l[col] if col in df_l.columns else np.nan

    out["acl_gap"] = (out["white_acl"] - out["black_acl"]).round(2)

    # ── Castling — not available in Lichess CSV ───────────────────────────────
    out["white_castled"] = np.nan
    out["black_castled"] = np.nan
    out["white_castle_side"] = np.nan
    out["black_castle_side"] = np.nan
    out["num_captures"] = np.nan

    # ── Source flag ───────────────────────────────────────────────────────────
    out["has_stockfish"] = out["white_acl"].notna()

    print(f"\nLichess harmonised shape: {out.shape}")
    print(f"has_stockfish = True : {out['has_stockfish'].sum():,}")
    print(f"has_stockfish = False: {(~out['has_stockfish']).sum():,}")
    return out


def integrate_datasets(
    df_pgn_pipeline: pd.DataFrame, lichess_path: str, lichess_sf_path: str
) -> pd.DataFrame:
    """
    Concatenate the PGN pipeline output with the Lichess CSV dataset.
    Both sources now have full Stockfish evaluation features.

    Parameters
    ----------
    df_pgn_pipeline : DataFrame — output of engineer_features()
    lichess_path    : str       — path to chess_games.csv
    lichess_sf_path : str       — path to lichess_stockfish.csv

    Returns
    -------
    Unified DataFrame with all ~43k games and full feature coverage
    """
    df_pgn_tagged = df_pgn_pipeline.copy()
    df_pgn_tagged["source"] = "kaggle_pgn"
    df_pgn_tagged["has_stockfish"] = True

    df_lichess = load_lichess(lichess_path, lichess_sf_path)
    df_combined = pd.concat([df_pgn_tagged, df_lichess], ignore_index=True, sort=False)
    df_combined["event_id"] = range(1, len(df_combined) + 1)

    print(f"\n── Integration Summary ──────────────────────────────────")
    print(f"Kaggle PGN rows    : {len(df_pgn_tagged):,}")
    print(f"Lichess CSV rows   : {len(df_lichess):,}")
    print(f"Combined total     : {len(df_combined):,}")
    print(f"\nhas_stockfish breakdown:")
    print(df_combined["has_stockfish"].value_counts().to_string())
    print(f"\nelo_bucket_white distribution:")
    print(df_combined["elo_bucket_white"].value_counts().sort_index().to_string())
    print(f"\nMissing white_acl  : {df_combined['white_acl'].isna().sum():,}")
    print(f"Missing black_acl  : {df_combined['black_acl'].isna().sum():,}")
    return df_combined


def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    """
    Save the final DataFrame to CSV, dropping raw move columns
    that are not needed for modeling (but keeping them available
    in the PGN files if needed later).

    Parameters
    ----------
    df          : final engineered DataFrame
    output_path : path to save the CSV file
    """
    # Drop raw move text — not needed in the model CSV
    cols_to_drop = ["moves_pgn", "moves_uci"]
    df_save = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    df_save.to_csv(output_path, index=False)
    print(
        f"Saved {len(df_save):,} rows × {df_save.shape[1]} columns to '{output_path}'"
    )
    print(f"\nFinal columns:\n{list(df_save.columns)}")
