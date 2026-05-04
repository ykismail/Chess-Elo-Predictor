import os
import sys

# Get the directory of the current script (src/data)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up to the project root (from src/data to project root)
project_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))

# Add project root to Python path to allow absolute imports like 'from src...'
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

import io
import re
import zipfile
import requests
import pandas as pd
import chess.pgn
from load_data import build_eco_lookup, extract_stockfish_features, match_eco, parse_uci, parse_pgn,load_lichess
from src.features.build_features import engineer_features

raw_dir = os.path.join(project_dir, "data", "raw")
os.makedirs(raw_dir, exist_ok=True)
intermediate_dir = os.path.join(project_dir, "data", "intermediate")
os.makedirs(intermediate_dir, exist_ok=True)
external_dir = os.path.join(project_dir, "data", "external")
os.makedirs(external_dir, exist_ok=True)

def acquire_data():
    os.environ["KAGGLE_API_TOKEN"] = "KGAT_145e8e3e445758440676eb29a8713200"
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print("Downloading Kaggle Datasets...")
    api.competition_download_files("finding-elo", path=raw_dir)

    # Download specific file from dataset
    api.dataset_download_file(
        "mysarahmadbhat/online-chess-games",
        "chess_games.csv",
        path=raw_dir
    )

    def unzip_all_in_dir(directory):
        import traceback
        extracted = True
        while extracted:
            extracted = False
            for file in os.listdir(directory):
                if file.endswith(".zip"):
                    file_path = os.path.join(directory, file)
                    print(f"Extracting {file}...")
                    try:
                        with zipfile.ZipFile(file_path, "r") as zip_ref:
                            print(f"Contents of {file}: {zip_ref.namelist()}")
                            zip_ref.extractall(directory)
                        print(f"Successfully extracted {file}")
                    except Exception as e:
                        print(f"Failed to extract {file}: {e}")
                        traceback.print_exc()
                    
                    try:
                        os.remove(file_path)
                    except:
                        pass
                    extracted = True

    print("\nUnzipping Kaggle files (including nested zips)...")
    unzip_all_in_dir(raw_dir)


    # 3. Download Lichess ECO Database from GitHub URL
    print("\nDownloading ECO Database from GitHub...")
    frames = []
    for letter in list("abcde"):
        url = f"https://raw.githubusercontent.com/lichess-org/chess-openings/master/{letter}.tsv"
        resp = requests.get(url)
        df_letter = pd.read_csv(io.StringIO(resp.text), sep="\t")
        df_letter["eco_family"] = letter.upper()
        frames.append(df_letter)

    def normalise_moves(pgn_str: str) -> str:
        if not isinstance(pgn_str, str):
            return ""
        cleaned = re.sub(r"\d+\.\s*", "", pgn_str)
        return " ".join(cleaned.split()).strip()
    
    df_eco = pd.concat(frames, ignore_index=True) 
    df_eco["moves_normalised"] = df_eco["pgn"].apply(normalise_moves)
    df_eco.to_csv(os.path.join(external_dir, "eco_openings.csv"), index=False)
    print("Data Acquisition Complete!")



if __name__ == "__main__":
    acquire_data()
    print("Parsing UCI file...")
    parsed_data_uci = parse_uci(raw_dir + "/data_uci.pgn")
    parsed_data_uci.to_csv(os.path.join(intermediate_dir, "parsed_data_uci.csv"), index=False)

    print("Parsing PGN file...")
    parsed_data_pgn = parse_pgn(raw_dir + "/data.pgn")

    #________________1)must be in enginearing feature phase_________________________________________________ 
    # df_eco = pd.read_csv(external_dir + "/eco_openings.csv")
    # eco_lookup = build_eco_lookup(df_eco)
    # eco_results = parsed_data_pgn["moves_san"].apply(lambda m: match_eco(m, eco_lookup))
    # parsed_data_pgn["eco_code"] = eco_results.apply(lambda x: x[0])
    # parsed_data_pgn["opening_name"] = eco_results.apply(lambda x: x[1])
    # parsed_data_pgn["eco_family"] = eco_results.apply(lambda x: x[2])
    parsed_data_pgn.to_csv(os.path.join(intermediate_dir, "parsed_data_pgn.csv"), index=False)
    

    chess_games_path = os.path.join(raw_dir, "chess_games.csv")
    stockfish_path = os.path.join(raw_dir, "stockfish.csv")
    
    #_______________________2)must be in enginearing feature phase__________________________________
    # should get stockfish.csv and rename Event to event_id then exctract  features later after merging 
    #df_sf = extract_stockfish_features(stockfish_path)
    df_sf = pd.read_csv(stockfish_path)
    df_sf = df_sf.rename(columns={"event_id":"Event"})
    # Merge Kaggle Datasets (Inner Join) pgn + uci + stockfish.csv
    df_kaggle = parsed_data_pgn.merge(parsed_data_uci[['event_id', 'moves_uci']], on='event_id', how='inner')
    df_kaggle = df_kaggle.merge(df_sf, on='event_id', how='inner')

    #_______________________3)must be in enginearing feature phase__________________________________
    # should be after merging 
    df_engineering_feature = engineer_features(df_kaggle)
    df_engineering_feature["source"]        = "kaggle_pgn"
    df_engineering_feature["has_stockfish"] = True
    lichess_path = os.path.join(raw_dir, "lichess_stockfish.csv")
    #_______________________4)must be in Transformation and enginearing feature phase__________________________________
    # Merge Lichess specific files 
    # # this contains  engineering features + encoding(transformations)
    df_lichess = load_lichess(chess_games_path, lichess_path)
    df_combined = pd.concat([df_engineering_feature, df_lichess], ignore_index=True, sort=False)
    df_combined["event_id"] = range(1, len(df_combined) + 1)


    # Save the Kaggle-only subset (games.csv)
    df_engineering_feature.to_csv(os.path.join(intermediate_dir, "games.csv"), index=False)
    print(f"Saved Kaggle-only raw dataset to games.csv: {df_engineering_feature.shape}")

    # Save the fully merged dataset (merged_games.csv)
    df_combined.to_csv(os.path.join(intermediate_dir, "merged_games.csv"), index=False)
    print(f"Saved fully merged raw dataset to merged_games.csv: {df_combined.shape}")

    print(f"\nStage 1 Complete. Raw Dataset Shape: {df_combined.shape}")
