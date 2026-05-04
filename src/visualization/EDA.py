import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual styling
sns.set_theme(style="whitegrid", palette="muted")

# Load the dataset
df = pd.read_csv("C:\\Users\\abdelrahman\\Desktop\\college\\data_science\\Chess-Elo-Predictor\\data\\processed\\merged_games.csv")

# Separate a dataframe for Stockfish-only analysis (ignoring Lichess missing data)
df_sf = df[df['has_stockfish'] == True].copy()

print(f"Total Games: {df.shape[0]:,}")
print(f"Games with Stockfish Analysis: {df_sf.shape[0]:,}")


plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, x='winner_multiclass', order=[2, 1, 0], palette=['#4C72B0', '#C44E52', '#55A868'])
plt.title('Distribution of Match Outcomes (Target)', fontsize=14)
plt.xticks(ticks=[0, 1, 2], labels=['White Wins (2)', 'Draw (1)', 'Black Wins (0)'])
plt.ylabel('Number of Games')

# Add percentage labels
total = len(df)
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%'
    x = p.get_x() + p.get_width() / 2 - 0.1
    y = p.get_height() + 200
    ax.annotate(percentage, (x, y), size=12)
plt.show()

# Bin the Elo gap for visualization
df['elo_gap_bin'] = pd.cut(
df['white_elo'] - df['black_elo']
, 
    bins=[-4000, -200, -50, 50, 200, 4000], 
    labels=['Black >> White', 'Black > White', 'Even', 'White > Black', 'White >> Black']
)

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='elo_gap_bin', hue='winner_multiclass', multiple='fill', shrink=.8, palette=['#55A868', '#C44E52', '#4C72B0'])
plt.title('Win Rate by Relative Skill Level (Elo Gap)', fontsize=14)
plt.ylabel('Proportion of Outcomes')
plt.xlabel('Elo Advantage')
plt.legend(labels=['White Wins', 'Draw', 'Black Wins'], title='Outcome', loc='upper left')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(data=df_sf, x='winner_multiclass', y='acl_gap', palette=['#55A868', '#C44E52', '#4C72B0'])
plt.title('Average Centipawn Loss Gap by Winner', fontsize=14)
plt.xticks(ticks=[0, 1, 2], labels=['Black Wins', 'Draw', 'White Wins'])
plt.ylabel('ACL Gap (White ACL - Black ACL)')
plt.ylim(-150, 150) # Zooming in to exclude extreme outliers
plt.show()


plt.figure(figsize=(10, 6))
sns.kdeplot(data=df_sf, x='game_sharpness', hue='winner_multiclass', fill=True, common_norm=False, alpha=0.4, palette=['#55A868', '#C44E52', '#4C72B0'])
plt.title('Game Volatility (Sharpness) Distribution by Outcome', fontsize=14)
plt.xlabel('Game Sharpness (Standard Deviation of Centipawns)')
plt.xlim(0, 1200)
plt.legend(labels=['White Wins', 'Draw', 'Black Wins'], title='Outcome')
plt.show()

# Get top 10 opening families by volume
top_openings = df['eco_family'].value_counts().index[:10]

plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='eco_family', y='winner_binary', order=top_openings, color='#4C72B0')
plt.title('White Win Rate by ECO Opening Family', fontsize=14)
plt.xlabel('ECO Family (A-E)')
plt.ylabel('White Win Rate (Excluding Draws)')
plt.axhline(y=df['winner_binary'].mean(), color='red', linestyle='--', label='Global Average')
plt.legend()
plt.show()