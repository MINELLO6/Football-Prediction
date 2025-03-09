import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Path to your data
data_path = '../../data/processed/merged_E0_common_sorted.csv'

# Load the data
df = pd.read_csv(data_path)

# Create a figure with multiple subplots
fig = plt.figure(figsize=(15, 10))
plt.suptitle('English Premier League Data Overview (1993-Present)', fontsize=16)

# Plot 1: Matches per season
ax1 = plt.subplot(2, 3, 1)
season_counts = df['SeasonSource'].value_counts().sort_index()
ax1.bar(season_counts.index.astype(str), season_counts.values)
ax1.set_title('Matches per Season')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
ax1.set_xlabel('Season')
ax1.set_ylabel('Number of Matches')

# Plot 2: Goals distribution
ax2 = plt.subplot(2, 3, 2)
df['TotalGoals'] = df['FTHG'] + df['FTAG']
sns.histplot(df['TotalGoals'], kde=True, ax=ax2)
ax2.set_title('Distribution of Goals per Match')
ax2.set_xlabel('Total Goals')
ax2.set_ylabel('Frequency')

# Plot 3: Home vs Away win percentage
ax3 = plt.subplot(2, 3, 3)
result_counts = df['FTR'].value_counts()
ax3.pie(result_counts, labels=['Away Win', 'Draw', 'Home Win'], 
        autopct='%1.1f%%', startangle=90,
        colors=['#FF9999', '#66B2FF', '#99FF99'])
ax3.set_title('Match Results Distribution')

# Plot 4: Average odds comparison
ax4 = plt.subplot(2, 3, 4)
odds_means = [df['B365H'].mean(), df['B365D'].mean(), df['B365A'].mean()]
odds_labels = ['Home Win', 'Draw', 'Away Win']
ax4.bar(odds_labels, odds_means, color=['#99FF99', '#66B2FF', '#FF9999'])
ax4.set_title('Average Bet365 Odds')
ax4.set_ylabel('Average Odds')

# Plot 5: Stats availability over time (using HTHG as proxy for stats availability)
ax5 = plt.subplot(2, 3, 5)
stats_by_season = df.groupby('SeasonSource')['HST'].notna().mean() * 100
ax5.plot(stats_by_season.index.astype(str), stats_by_season.values, marker='o')
ax5.set_title('Match Statistics Availability')
ax5.set_xticklabels(ax5.get_xticklabels(), rotation=90)
ax5.set_xlabel('Season')
ax5.set_ylabel('Percentage of Matches with Stats')

# Plot 6: Top teams performance
ax6 = plt.subplot(2, 3, 6)
top_teams = df['HomeTeam'].value_counts().nlargest(5).index
team_win_rates = {}
for team in top_teams:
    # Calculate win rate when team is home
    home_matches = df[df['HomeTeam'] == team]
    home_wins = home_matches[home_matches['FTR'] == 'H'].shape[0]
    home_win_rate = home_wins / home_matches.shape[0] if home_matches.shape[0] > 0 else 0
    
    # Calculate win rate when team is away
    away_matches = df[df['AwayTeam'] == team]
    away_wins = away_matches[away_matches['FTR'] == 'A'].shape[0]
    away_win_rate = away_wins / away_matches.shape[0] if away_matches.shape[0] > 0 else 0
    
    team_win_rates[team] = (home_win_rate + away_win_rate) / 2

team_names = list(team_win_rates.keys())
win_rates = list(team_win_rates.values())
ax6.bar(team_names, win_rates, color='gold')
ax6.set_title('Win Rates for Top Teams')
ax6.set_ylabel('Win Rate')
ax6.set_ylim(0, 1)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the figure for PowerPoint
plt.savefig('data_overview.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()