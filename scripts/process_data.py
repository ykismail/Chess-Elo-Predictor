import pandas as pd
from src.data.load_data import download_eco_database, build_eco_lookup
from src.data.preprocess import parse_pgn, load_lichess_data
from src.features.build_features import extract_stockfish_features


def main():
    # 1. Parse the main PGN
    df_pgn = parse_pgn("data/raw/data.pgn")

    # 2. Load Local Stockfish & Extract Features
    df_sf_local = pd.read_csv("data/raw/stockfish.csv")
    df_feat_local = extract_stockfish_features(df_sf_local)

    # 3. Load Lichess Stockfish (The part we missed!)
    df_lichess = load_lichess_data("data/raw/lichess_stockfish.csv")

    # 4. The Triple Merge
    # Merge PGN with Local Stockfish
    df_intermediate = df_pgn.merge(df_feat_local, on="event_id", how="inner")

    # Merge with Lichess data (Source 3)
    df_final = df_intermediate.merge(df_lichess, on="event_id", how="inner")

    # 5. Save final canonical dataset
    df_final.to_csv("data/processed/merged_games.csv", index=False)
    print(f"Triple Merge Complete! Final count: {len(df_final)} rows.")


if __name__ == "__main__":
    main()
