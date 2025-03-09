import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../../penaltyblog")
import penaltyblog as pb

# Read data
data_path = "../../data/processed/merged_E0_common_sorted.csv"
df = pd.read_csv(data_path)

# Preprocess date column
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")

# Calculate total days
total_days = (df["Date"].max() - df["Date"].min()).days
print(f"Total dataset span: {total_days} days")

# Define day increments (200 days each)
increment = 200
start_day = 200  # Start from day 200
days_to_analyze = list(range(start_day, total_days + 1, increment))
# Add the final day if it's not already included
if days_to_analyze[-1] != total_days:
    days_to_analyze.append(total_days)

home_advantage_results = []
team_params_dict = {
    "Sheffield United": {"days": [], "attack": [], "defense": []},
    "Norwich": {"days": [], "attack": [], "defense": []},
    "Everton": {"days": [], "attack": [], "defense": []}
}

# Teams to track
teams_to_track = ["Sheffield United", "Norwich", "Everton"]

# Analyze home advantage for different day ranges
start_date = df["Date"].min()
for days in days_to_analyze:
    print(f"\nAnalyzing first {days} days of data...")

    # Calculate end date
    end_date = start_date + pd.Timedelta(days=days)

    # Extract data for this range
    date_range_data = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
    match_count = len(date_range_data)

    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Number of matches: {match_count}")

    if match_count < 50:
        print(f"Warning: Sample size too small ({match_count}), skipping this range")
        continue

    # Fit Dixon-Coles model
    model = pb.models.DixonColesGoalModel(
        date_range_data["Date"],
        date_range_data["FTHG"],
        date_range_data["FTAG"],
        date_range_data["HomeTeam"],
        date_range_data["AwayTeam"],
        xi=0.0002  # Use the same xi value as in the original code
    )
    model.fit()

    # Extract home advantage parameter (rho, usually the last parameter)
    home_advantage = model._params[-1]
    # Calculate exponential of home advantage
    exp_home_advantage = np.exp(home_advantage)
    print(f"Home advantage (rho): {home_advantage}")
    print(f"Exponential of home advantage: {exp_home_advantage}")

    # Save result
    home_advantage_results.append({
        "days": days,
        "home_advantage": home_advantage,
        "exp_home_advantage": exp_home_advantage,
        "match_count": match_count
    })

    # Extract attack and defense parameters for tracked teams
    params = model.get_params()
    for team in teams_to_track:
        attack_key = f"attack_{team}"
        defense_key = f"defence_{team}"
        if attack_key in params and defense_key in params:
            attack = params[attack_key]
            defense = params[defense_key]
            # Save exponential values
            team_params_dict[team]["days"].append(days)
            team_params_dict[team]["attack"].append(np.exp(attack))
            team_params_dict[team]["defense"].append(np.exp(defense))
            print(f"{team}: attack={np.exp(attack):.4f}, defense={np.exp(defense):.4f}")

# Plot 1: Home advantage vs days (exponential values)
plt.figure(figsize=(10, 6))
days = [result["days"] for result in home_advantage_results]
exp_advantages = [result["exp_home_advantage"] for result in home_advantage_results]

# Plot with black line, no markers
plt.plot(days, exp_advantages, linestyle='-', color='black')
plt.xlabel("Analysis Period (days)")
plt.ylabel("Home Advantage Parameter (exp(rho))")
plt.title("Home Advantage (Exponential) by Time Period")

# Set y-axis to start from 0
plt.ylim(0, max(exp_advantages) * 1.1)
plt.grid(False)

plt.savefig("home_advantage_exp_by_days.png", dpi=300)
print("\nChart saved as home_advantage_exp_by_days.png")

# Plot 2: Team attack parameters
plt.figure(figsize=(10, 6))

line_styles = {
    "Sheffield United": "-",
    "Norwich": "--",
    "Everton": "-."
}

for team in teams_to_track:
    if team_params_dict[team]["days"]:  # Check if we have data for this team
        plt.plot(
            team_params_dict[team]["days"],
            team_params_dict[team]["attack"],
            linestyle=line_styles[team],
            color='black',
            label=f"{team}"
        )

plt.xlabel("Analysis Period (days)")
plt.ylabel("Attack Parameter Value (exponential)")
plt.title("Attack Parameters for Selected Teams")
plt.legend()
plt.grid(False)
plt.ylim(0, max([max(team_params_dict[team]["attack"]) for team in teams_to_track if team_params_dict[team]["attack"]]) * 1.1)

plt.savefig("team_attack_parameters_by_days.png", dpi=300)
print("Team attack parameters chart saved as team_attack_parameters_by_days.png")

# Plot 3: Team defense parameters
plt.figure(figsize=(10, 6))

for team in teams_to_track:
    if team_params_dict[team]["days"]:  # Check if we have data for this team
        plt.plot(
            team_params_dict[team]["days"],
            team_params_dict[team]["defense"],
            linestyle=line_styles[team],
            color='black',
            label=f"{team}"
        )

plt.xlabel("Analysis Period (days)")
plt.ylabel("Defense Parameter Value (exponential)")
plt.title("Defense Parameters for Selected Teams")
plt.legend()
plt.grid(False)
plt.ylim(0, max([max(team_params_dict[team]["defense"]) for team in teams_to_track if team_params_dict[team]["defense"]]) * 1.1)

plt.savefig("team_defense_parameters_by_days.png", dpi=300)
print("Team defense parameters chart saved as team_defense_parameters_by_days.png")

# Save results to CSV
results_df = pd.DataFrame(home_advantage_results)
results_df.to_csv("home_advantage_by_days.csv", index=False)
print("Results saved as home_advantage_by_days.csv")

# Save team parameters to CSV
all_team_data = []
for team in teams_to_track:
    for i in range(len(team_params_dict[team]["days"])):
        all_team_data.append({
            "team": team,
            "days": team_params_dict[team]["days"][i],
            "attack": team_params_dict[team]["attack"][i],
            "defense": team_params_dict[team]["defense"][i]
        })
team_params_df = pd.DataFrame(all_team_data)
team_params_df.to_csv("team_parameters_by_days.csv", index=False)
print("Team parameters saved as team_parameters_by_days.csv")