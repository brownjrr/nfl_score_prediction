import pandas as pd
import json

# create season from date 
def create_season(row):
    year,month, day = map(int, row['event_date'].split('-'))
    if month < 3:  # Adjust year for games in the first 3 months
        year -= 1
    return year

# Define weights for each team statistic in game_stats
weights = {
    'first_downs': 0.035, 'rush_att': 0.01, 'rush_yds': 0.06, 'rush_tds': 0.04, 'pass_cmp': 0.025, 'pass_att': 0.015,'pass_cmp_pct': 0.07,
    'pass_yds': 0.06, 'pass_tds': 0.05, 'pass_int': -0.05, 'times_sacked': -0.04, 'yds_sacked_for': -0.025, 'net_pass_yards': 0.05, 
    'total_yards': 0.06, 'fumbles': -0.03, 'fumbles_lost': -0.045, 'turnovers': -0.06, 'penalties': -0.02,'penalty_yds': -0.02, 
    'third_down_conv': 0.03, 'third_down_att': 0.01, 'third_down_conv_pct': 0.04, 'fourth_down_conv': 0.02, 
    'fourth_down_att': 0.01, 'fourth_down_conv_pct': 0.03, 'time_of_possession': 0.05  
}

def calculate_team_ratings(df, abbrev, weights, window_size=3, decay_factor=0.85):
    """
    Calculates a rating for each team based on game statistics and applies a rolling average,
    giving more weight to the current season's ratings when transitioning from the previous season.
    
    Args:
    df (pd.DataFrame): DataFrame containing game stats.
    abbrev (dict): Dictionary of team abbreviations for each teams.
    weights (dict): Dictionary of weights for each statistic.
    window_size (int): The window size for rolling averages. Default is 3.
    decay_factor (float): The weight given to the current season's rating when combining with
                          the previous season's rating. Default is 0.85.
    
    Returns:
    pd.DataFrame: DataFrame with rolled ratings by team and season.
    """

    # Create team_id and season using abbreviation dict and season function
    df['team_id'] = df['team'].map(abbrev)
    df['season'] = df.apply(create_season, axis=1)
    
    # Ensure 'event_date' is in datetime format if needed
    df['event_date'] = pd.to_datetime(df['event_date'])
    
    # Sort data by team and event_date to ensure correct chronological order
    df = df.sort_values(by=['team', 'event_date'])
    
    # Define a function for calculating the rating from game stats
    def calculate_rating(row, weights):
        return sum(row[stat] * weights.get(stat, 0) for stat in weights)
    
    # Apply the rating function to each row
    df['rating'] = df.apply(lambda row: calculate_rating(row, weights), axis=1)
    
    # Initialize the rolling_rating column
    df['rolling_rating'] = 0.0
    
    # Process each team separately
    for team, team_data in df.groupby('team'):
        # Sort by season and event_date for each team
        team_data = team_data.sort_values(by=['season', 'event_date'])
        
        # Iterate through each season for the current team
        for season, season_data in team_data.groupby('season'):
            if season == df['season'].min():
                # For the first season, use a regular rolling average directly
                team_data.loc[season_data.index, 'rolling_rating'] = (
                    season_data['rating'].rolling(window=window_size, closed='left').mean()
                )
            else:
                # Handle the transition from the previous season
                # Get the average rating of the previous season as the starting point
                previous_season_data = team_data[team_data['season'] < season]
                previous_season_avg_rating = previous_season_data['rating'][-window_size:].mean() if not previous_season_data.empty else 0
                
                # Iterate through the first few games of the new season
                for i in range(len(season_data)):
                    if i == 0:
                        # Use the rolling average of the last season for the first game
                        combined_rating = previous_season_avg_rating
                    elif i < window_size:
                        # For the next few games, use a weighted combination of the previous season's rating and current season's ratings
                        weight = decay_factor + (i * (1 - decay_factor) / window_size)
                        current_game_rating = season_data['rating'].iloc[:i].mean()
                        combined_rating = weight * current_game_rating + (1 - weight) * previous_season_avg_rating
                    else:
                        # After window size games, use regular rolling average of the current season
                        combined_rating = season_data['rating'].iloc[:i].rolling(window=window_size, min_periods=1).mean().iloc[-1]
                    
                    # Assign the combined rating to the current game
                    team_data.loc[season_data.index[i], 'rolling_rating'] = combined_rating
        
        # Update the DataFrame with the team's adjusted ratings
        df.loc[team_data.index, 'rolling_rating'] = team_data['rolling_rating']
    
    # Fill NaN values that may result from the shifting process
    df['rolling_rating'] = df['rolling_rating'].fillna(0)
    
    # Keep only the necessary columns for analysis or further modeling
    final_columns = ['team_id', 'season', 'event_date', 'rating', 'rolling_rating']
    df = df[final_columns]
    df.sort_values(by='event_date',inplace=True)
    # Reset the index
    df.reset_index(drop=True, inplace=True)
    df.rename({'rolling_rating': 'team_rating'}, axis=1, inplace=True)
    
    return df



# files used for team rating calculation

# games_stats = pd.read_csv(f"{filepath}/game_stats_all.csv")
# with open(f"{filepath}/abbrev_map.json", 'r') as file:
#     abb_dict = json.load(file)

# # Apply the function to the provided team stats
# window_size = 3  # Define the rolling window size as needed
# decay_factor = 0.7  # Adjust this to give more weight to the previous season for the first game
# team_ratings = calculate_team_ratings(games_stats, abb_dict, weights, window_size=window_size, decay_factor=decay_factor)