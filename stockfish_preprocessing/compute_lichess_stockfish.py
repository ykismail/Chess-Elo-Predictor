"""
compute_lichess_stockfish.py
============================
CMPS344 Applied Data Science — Phase 2

Computes per-game Stockfish centipawn evaluations for chess_games.csv
(the Lichess dataset) and saves results in the same format as the
existing stockfish.csv from the Kaggle PGN dataset.

Output
------
lichess_stockfish.csv  — two columns: Event (game_id), MoveScores
    MoveScores is a space-separated string of centipawn evaluations,
    one per half-move, matching the format of the Kaggle stockfish.csv.

Usage
-----
    # 1. Install dependencies (once)
    pip install python-chess

    # 2. Download Stockfish binary from https://stockfishchess.org/download/
    #    Place the binary in the same folder as this script, or update
    #    STOCKFISH_PATH below to the full path on your system.
    #
    #    Windows : stockfish-windows-x86-64.exe
    #    macOS   : stockfish-macos
    #    Linux   : stockfish-ubuntu-x86-64

    # 3. Run the script
    python compute_lichess_stockfish.py

    # 4. Optional flags
    python compute_lichess_stockfish.py --depth 10 --workers 4
    python compute_lichess_stockfish.py --resume          # resume interrupted run
    python compute_lichess_stockfish.py --input path/to/chess_games.csv
    python compute_lichess_stockfish.py --output path/to/lichess_stockfish.csv

Checkpointing
-------------
Results are saved every CHECKPOINT_EVERY games so the run can be
resumed after interruption with --resume.

Time estimate (single core)
---------------------------
depth  8  : ~40  minutes for 20,058 games
depth 10  : ~60  minutes for 20,058 games
depth 12  : ~100 minutes for 20,058 games

Use --workers N to parallelise across N CPU cores (recommended).
"""

import argparse
import csv
import io
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import chess
import chess.engine
import chess.pgn
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
STOCKFISH_PATH   = "stockfish_preprocessing/stockfish"        # update if binary has a different name/path
DEFAULT_DEPTH    = 10                 # analysis depth per position
CHECKPOINT_EVERY = 500                # save progress every N games
CAP_CENTIPAWNS   = 2000               # cap evaluations at ±2000 to avoid mate scores
# ──────────────────────────────────────────────────────────────────────────────


def _cap(value: int, cap: int = CAP_CENTIPAWNS) -> int:
    """Clamp a centipawn value to [-cap, +cap]."""
    return max(-cap, min(cap, value))


def evaluate_game(
    game_id: int,
    moves_san: str,
    stockfish_path: str,
    depth: int,
) -> Optional[dict]:
    """
    Evaluate every position in a single game using Stockfish.

    Parameters
    ----------
    game_id      : int — identifier for the game (from chess_games.csv game_id)
    moves_san    : str — space-separated SAN moves (e.g. 'd4 d5 c4 c6 ...')
    stockfish_path : str — path to Stockfish binary
    depth        : int — analysis depth per position

    Returns
    -------
    dict with keys 'Event' and 'MoveScores', or None on failure
    """
    try:
        # Build a PGN string python-chess can parse
        # Lichess moves column is already SAN without move numbers
        tokens    = moves_san.strip().split()
        pgn_moves = " ".join(
            f"{i // 2 + 1}. {m}" if i % 2 == 0 else m
            for i, m in enumerate(tokens)
        )
        pgn_str = f"[Event \"{game_id}\"]\n\n{pgn_moves}\n"

        game = chess.pgn.read_game(io.StringIO(pgn_str))
        if game is None:
            return None

        board  = game.board()
        scores = []

        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            for move in game.mainline_moves():
                board.push(move)
                info  = engine.analyse(board, chess.engine.Limit(depth=depth))
                score = info["score"].white()

                if score.is_mate():
                    # Convert mate-in-N to a large centipawn value with sign
                    cp = CAP_CENTIPAWNS if score.mate() > 0 else -CAP_CENTIPAWNS
                else:
                    cp = _cap(score.score())

                scores.append(cp)

        if not scores:
            return None

        return {
            "Event":      game_id,
            "MoveScores": " ".join(map(str, scores)),
        }

    except Exception as e:
        # Silently skip games that fail (corrupt move string, engine crash, etc.)
        return None


def _worker(args):
    """Wrapper for multiprocessing — unpacks arguments."""
    return evaluate_game(*args)


def load_already_processed(output_path: str) -> set:
    """Return set of game_ids already written to the output CSV."""
    if not os.path.exists(output_path):
        return set()
    done = set()
    with open(output_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                done.add(int(row["Event"]))
            except (KeyError, ValueError):
                pass
    return done


def run(
    input_path:  str  = "../data/chess_games.csv",
    output_path: str  = "../data/lichess_stockfish.csv",
    stockfish_path: str = STOCKFISH_PATH,
    depth:       int  = DEFAULT_DEPTH,
    workers:     int  = 1,
    resume:      bool = False,
):
    """
    Main entry point — processes all games in chess_games.csv.

    Parameters
    ----------
    input_path     : path to chess_games.csv
    output_path    : path to save lichess_stockfish.csv
    stockfish_path : path to Stockfish binary
    depth          : Stockfish analysis depth per position
    workers        : number of parallel processes
    resume         : if True, skip games already in output_path
    """
    # ── Validate Stockfish binary ─────────────────────────────────────────────
    try:
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            engine_name = engine.id.get("name", "Unknown")
            print(f"Stockfish found: {engine_name}")
    except Exception as e:
        print(f"\nERROR: Could not start Stockfish at '{stockfish_path}'")
        print(f"  {e}")
        print("\nFix: Download Stockfish from https://stockfishchess.org/download/")
        print("     Place the binary in this folder, or update STOCKFISH_PATH in the script.")
        sys.exit(1)

    # ── Load input data ────────────────────────────────────────────────────────
    df = pd.read_csv(input_path)
    print(f"\nLoaded {len(df):,} games from '{input_path}'")

    # ── Resume support ────────────────────────────────────────────────────────
    already_done = set()
    if resume and os.path.exists(output_path):
        already_done = load_already_processed(output_path)
        print(f"Resuming — {len(already_done):,} games already processed, skipping.")

    todo = df[~df["game_id"].isin(already_done)].copy()
    print(f"Games to process: {len(todo):,}")

    if todo.empty:
        print("Nothing to do — all games already processed.")
        return

    # ── Estimate time ─────────────────────────────────────────────────────────
    avg_halfmoves  = df["turns"].mean()
    ms_per_pos     = {8: 2, 10: 5, 12: 10}.get(depth, 5)
    est_minutes    = (avg_halfmoves * len(todo) * ms_per_pos) / 1000 / 60
    print(f"Estimated time  : ~{est_minutes:.0f} minutes at depth {depth}"
          f" ({workers} worker{'s' if workers > 1 else ''})")
    print(f"Output          : '{output_path}'")
    print(f"Checkpoint every: {CHECKPOINT_EVERY} games\n")

    # ── Open output file ──────────────────────────────────────────────────────
    write_header = not os.path.exists(output_path) or not resume
    out_file = open(output_path, "a" if resume else "w", newline="")
    writer   = csv.DictWriter(out_file, fieldnames=["Event", "MoveScores"])
    if write_header:
        writer.writeheader()

    # ── Process games ─────────────────────────────────────────────────────────
    tasks = [
        (int(row["game_id"]), row["moves"], stockfish_path, depth)
        for _, row in todo.iterrows()
    ]

    processed = 0
    failed    = 0
    t_start   = time.time()

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_worker, t): t[0] for t in tasks}
            buffer  = []

            for future in as_completed(futures):
                result = future.result()
                if result:
                    buffer.append(result)
                else:
                    failed += 1

                processed += 1

                # Checkpoint
                if len(buffer) >= CHECKPOINT_EVERY:
                    writer.writerows(buffer)
                    out_file.flush()
                    buffer = []

                # Progress
                elapsed  = time.time() - t_start
                rate     = processed / elapsed if elapsed > 0 else 1
                eta_secs = (len(tasks) - processed) / rate
                print(
                    f"\r  {processed:>6,}/{len(tasks):,} games"
                    f"  |  failed: {failed}"
                    f"  |  {rate:.1f} games/s"
                    f"  |  ETA: {eta_secs/60:.1f} min   ",
                    end="", flush=True
                )

            if buffer:
                writer.writerows(buffer)
                out_file.flush()
    else:
        # Single-process mode
        buffer = []
        for task in tasks:
            result = _worker(task)
            if result:
                buffer.append(result)
            else:
                failed += 1

            processed += 1

            if len(buffer) >= CHECKPOINT_EVERY:
                writer.writerows(buffer)
                out_file.flush()
                buffer = []

            elapsed  = time.time() - t_start
            rate     = processed / elapsed if elapsed > 0 else 1
            eta_secs = (len(tasks) - processed) / rate
            print(
                f"\r  {processed:>6,}/{len(tasks):,} games"
                f"  |  failed: {failed}"
                f"  |  {rate:.1f} games/s"
                f"  |  ETA: {eta_secs/60:.1f} min   ",
                end="", flush=True
            )

        if buffer:
            writer.writerows(buffer)
            out_file.flush()

    out_file.close()

    elapsed_total = time.time() - t_start
    success_count = processed - failed
    print(f"\n\nDone.")
    print(f"  Processed : {processed:,} games in {elapsed_total/60:.1f} minutes")
    print(f"  Succeeded : {success_count:,}")
    print(f"  Failed    : {failed:,}")
    print(f"  Saved to  : '{output_path}'")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Stockfish evaluations for Lichess chess_games.csv"
    )
    parser.add_argument(
        "--input",   default="chess_games.csv",
        help="Path to chess_games.csv (default: chess_games.csv)"
    )
    parser.add_argument(
        "--output",  default="lichess_stockfish.csv",
        help="Output CSV path (default: lichess_stockfish.csv)"
    )
    parser.add_argument(
        "--stockfish", default=STOCKFISH_PATH,
        help=f"Path to Stockfish binary (default: {STOCKFISH_PATH})"
    )
    parser.add_argument(
        "--depth",   type=int, default=DEFAULT_DEPTH,
        help=f"Analysis depth per position (default: {DEFAULT_DEPTH})"
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel processes (default: 1)"
    )
    parser.add_argument(
        "--resume",  action="store_true",
        help="Resume an interrupted run (skip already-processed games)"
    )

    args = parser.parse_args()

    run(
        input_path=args.input,
        output_path=args.output,
        stockfish_path=args.stockfish,
        depth=args.depth,
        workers=args.workers,
        resume=args.resume,
    )
