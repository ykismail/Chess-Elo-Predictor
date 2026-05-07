import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)


# Set visual styling
sns.set_theme(style="whitegrid", palette="muted")

# Target Variable Distribution (elo_bucket_white_categorical)
def plot_target_variable_distribution(df: pd.DataFrame, output_dir: str):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="elo_bucket_white_categorical", color="#4C72B0")
    plt.title("Distribution of Target Variable: Elo Buckets (White Player)", fontsize=14)
    plt.xlabel("Elo Bucket (White Player)")
    plt.ylabel("Number of Games")
    # plt.show()
    plt.savefig(os.path.join(output_dir, "target_variable_distribution.png"), bbox_inches="tight")
    plt.close()
# Feature-to-Target — Skill Gap vs. Outcome
def plot_win_rate_by_elo_gap(df: pd.DataFrame, output_dir: str):
    # Bin the Elo gap for visualization
    df_plot = df.copy()
    df_plot['elo_gap_bin'] = pd.cut(
        df_plot['white_elo'] - df_plot['black_elo'], 
        bins=[-4000, -200, -50, 50, 200, 4000], 
        labels=['Black >> White', 'Black > White', 'Even', 'White > Black', 'White >> Black']
    )

    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_plot, x='elo_gap_bin', hue='winner_multiclass', multiple='fill', shrink=.8, palette=['#55A868', '#C44E52', '#4C72B0'])
    plt.title('Win Rate by Relative Skill Level (Elo Gap)', fontsize=14)
    plt.ylabel('Proportion of Outcomes')
    plt.xlabel('Elo Advantage')
    plt.legend(labels=['White Wins', 'Draw', 'Black Wins'], title='Outcome', loc='upper left')
    
    plt.savefig(os.path.join(output_dir, "win_rate_elo_gap.png"), bbox_inches="tight")
    plt.close()

# Feature-to-Target — Engine Accuracy Gap (ACL)
def plot_acl_gap(df_sf: pd.DataFrame, output_dir: str):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_sf, x='winner_multiclass', y='acl_gap', hue='winner_multiclass', palette=['#55A868', '#C44E52', '#4C72B0'], legend=False)
    plt.title('Average Centipawn Loss Gap by Winner', fontsize=14)
    plt.xticks(ticks=[0, 1, 2], labels=['Black Wins', 'Draw', 'White Wins'])
    plt.ylabel('ACL Gap (White ACL - Black ACL)')
    plt.ylim(-150, 150) 
    
    plt.savefig(os.path.join(output_dir, "acl_gap.png"), bbox_inches="tight")
    plt.close()

# Game Dynamics — Sharpness vs. Draws
def plot_game_sharpness(df_sf: pd.DataFrame, output_dir: str):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df_sf, x='game_sharpness', hue='winner_multiclass', fill=True, common_norm=False, alpha=0.4, palette=['#55A868', '#C44E52', '#4C72B0'])
    plt.title('Game Volatility (Sharpness) Distribution by Outcome', fontsize=14)
    plt.xlabel('Game Sharpness (Standard Deviation of Centipawns)')
    plt.xlim(0, 1200)
    plt.legend(labels=['White Wins', 'Draw', 'Black Wins'], title='Outcome')
    
    plt.savefig(os.path.join(output_dir, "game_sharpness.png"), bbox_inches="tight")
    plt.close()

# Openings — Does the First Move Matter?
def plot_white_win_rate_eco(df: pd.DataFrame, output_dir: str):
    # Get top 10 opening families by volume
    top_openings = df['eco_family'].value_counts().index[:10]

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='eco_family', y='winner_binary', order=top_openings, color='#4C72B0')
    plt.title('White Win Rate by ECO Opening Family', fontsize=14)
    plt.xlabel('ECO Family (A-E)')
    plt.ylabel('White Win Rate (Excluding Draws)')
    plt.axhline(y=df['winner_binary'].mean(), color='red', linestyle='--', label='Global Average')
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, "white_win_rate_eco.png"), bbox_inches="tight")
    plt.close()

# Customer Segmentation: ECO Win Rate by Elo Bucket
def plot_eco_win_rate_by_elo(df: pd.DataFrame, output_dir: str):
    # Filter to the top 5 ECO families to keep the chart readable (A, B, C, D, E)
    top_openings = df['eco_family'].value_counts().index[:5]
    df_filtered = df[df['eco_family'].isin(top_openings)].copy()

    # Ensure Elo buckets are treated as categorical strings for the legend
    df_filtered['elo_bucket'] = df_filtered['elo_bucket_white_categorical'].astype(str)

    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=df_filtered, 
        x='eco_family', 
        y='winner_binary', 
        hue='elo_bucket',
        order=top_openings,
        palette='viridis',
        errorbar=None # Clean look without error bars for business presentations
    )
    
    plt.title('White Win Rate by ECO Family & Elo Bucket (Customer Segmentation)', fontsize=14)
    plt.xlabel('ECO Family (A-E)')
    plt.ylabel('White Win Rate')
    
    # Move legend outside the plot
    plt.legend(title='Elo Bucket (White)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axhline(y=df['winner_binary'].mean(), color='red', linestyle='--', alpha=0.5, label='Global Average')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eco_win_rate_by_elo.png"), bbox_inches="tight")
    plt.close()


def main():

    figures_dir = os.path.join(project_dir, "reports/figures")
    os.makedirs(figures_dir, exist_ok=True)

    intermediate_dir = os.path.join(project_dir, "data", "intermediate")
    merged_games_path = os.path.join(intermediate_dir, "merged_games.csv")
    
    df = pd.read_csv(merged_games_path, low_memory=False)

    # Separate a dataframe for Stockfish-only analysis 
    df_sf = df[df['has_stockfish'] == True].copy()
    
    print(f"Total Games: {df.shape[0]:,}")
    print(f"Games with Stockfish Analysis: {df_sf.shape[0]:,}")
    
    print(f"Saving plots to {figures_dir} ...")
    plot_target_variable_distribution(df, figures_dir)
    plot_win_rate_by_elo_gap(df, figures_dir)
    plot_acl_gap(df_sf, figures_dir)
    plot_game_sharpness(df_sf, figures_dir)
    plot_white_win_rate_eco(df, figures_dir)
    plot_eco_win_rate_by_elo(df, figures_dir)
    print("Done.")

if __name__ == "__main__":
    main()
