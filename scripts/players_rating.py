
import pandas as pd
import json
import os


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

print(f"script_dir: {script_dir}")

# Weighting factors for game outcomes and previous rating influence
outcome_factor = {'W': 1.05, 'L': 0.98, 'T': 0}
#outcome_factor = {'won': 0.25, 'loss': -0.25, 'tie': 0}
previous_rating_weight = 0.7  # Weight for the previous rating

# injury weight map for custom player rating
injury_weight_map = {
    'Available': 0.3,
    'Probable': 0.2,
    'Questionable': -0.1,
    'Doubtful': -0.2,
    'Out': -0.3,
    'Physically Unable To Perform': -0.3,
    'Injured Reserve': -0.3
}

# Function to create event_date for injuries dataset
def create_event_date(row):
    month, day = map(int, row['date'].split('/'))
    year = row['season']
    if month < 3:  # Adjust year for games in the first 3 months
        year += 1
    return pd.Timestamp(year=year, month=month, day=day)

# Convert 'years played' into numeric form
def convert_years_played(value):
    if value == 'Rook':
        return 0
    return int(value)

# Add game outcome to games_info based on team scores
def determine_outcome(row):
    if row['team_a_score'] > row['team_b_score']:
        return {'team_a': 'W', 'team_b': 'L'}
    elif row['team_a_score'] < row['team_b_score']:
        return {'team_a': 'L', 'team_b': 'W'}
    else:
        return {'team_a': 'T', 'team_b': 'T'}
    
def determine_home_advantage(row, team):
    if row['location'] in team:
        return 1
    else:
        return 0
    
# create season from date 
def create_season(row):
    year,month, day = map(int, row['date'].split('-'))
    if month < 3:  # Adjust year for games in the first 3 months
        year -= 1
    return year

def data_cleaning_formatting(snap_counts,rosters, injuries, starters_df,games_info,team_dict,team_dict_short):

    starters_df.columns = starters_df.columns.str.lower()
    rosters.loc[:, 'years_played'] = rosters['years_played'].apply(convert_years_played)
    rosters['season'] = rosters['roster_year']

    # Apply the function to create event_date column
    injuries['event_date'] = injuries.apply(create_event_date, axis=1)

    # Clean the percentage columns in snap_counts by removing '%' and converting to numeric
    snap_counts['off_pct'] = pd.to_numeric(snap_counts['off_pct'].str.replace('%', ''), errors='coerce')/100
    snap_counts['def_pct'] = pd.to_numeric(snap_counts['def_pct'].str.replace('%', ''), errors='coerce')/100
    snap_counts['st_pct'] = pd.to_numeric(snap_counts['st_pct'].str.replace('%', ''), errors='coerce')/100

    # add _abb columns for team uniformity
    rosters['team_abb'] = rosters['team'].map(team_dict.get)
    injuries['team_abb'] = injuries['team'].map(team_dict.get)
    starters_df['team_abb'] = starters_df['team'].map(team_dict_short.get)
    snap_counts['team_abb'] = snap_counts['team'].map(team_dict_short.get)
    games_info['team_a_abb'] = games_info['team_a'].map(team_dict.get)
    games_info['team_b_abb'] = games_info['team_b'].map(team_dict.get)

    rosters.drop_duplicates(subset=['player_id','team_abb','season','birth_date'], inplace=True)

    # Apply game outcomes
    games_info['outcome_team_a'] = games_info.apply(lambda row: determine_outcome(row)['team_a'], axis=1)
    games_info['outcome_team_b'] = games_info.apply(lambda row: determine_outcome(row)['team_b'], axis=1)
    games_info['home_team_a'] = games_info.apply(lambda row: determine_home_advantage(row, row['team_a']), axis=1)
    games_info['home_team_b'] = games_info.apply(lambda row: determine_home_advantage(row, row['team_b']), axis=1)

    # Merge the snap_counts with games_info based on team_a and event_date
    merged_df = pd.merge(snap_counts, games_info, left_on=['date', 'team_abb'], right_on=['event_date', 'team_a_abb'], how='left')

    # Merge again for team_b to cover both teams
    merged_b = pd.merge(snap_counts, games_info, left_on=['date', 'team_abb'], right_on=['event_date', 'team_b_abb'], how='left')

    # Combine the two merged DataFrames based on the outcome and home game
    merged_df['outcome'] = merged_df['outcome_team_a'].combine_first(merged_b['outcome_team_b'])
    merged_df['home'] = merged_df['home_team_a'].combine_first(merged_b['home_team_b'])

    # Clean up the resulting DataFrame if needed
    merged_df = merged_df[list(snap_counts.columns) + ['outcome','home']]

    snap_counts = merged_df.copy()

    # Apply the function to create event_date column
    snap_counts.loc[:, 'season'] = snap_counts.apply(create_season, axis=1)
    snap_counts['position'] = snap_counts['position'].str.split('/').str[0]

    return snap_counts,rosters, injuries, starters_df,games_info
    

# Function to merge player stats for a given event date
def merge_data_for_event_date(event_date, offense_stats, defense_stats, kicking_stats, returns_stats, snap_counts,roster_data,injuries,starters):
    # Filter stats up to the event date
    offense_up_to_date = offense_stats[offense_stats['date'] == event_date].copy()
    defense_up_to_date = defense_stats[defense_stats['date'] == event_date].copy()
    kicking_up_to_date = kicking_stats[kicking_stats['date'] == event_date].copy()
    returns_up_to_date = returns_stats[returns_stats['date'] == event_date].copy()
    # Merge snap counts and injuries
    snap_counts_up_to_date = snap_counts[snap_counts['date'] == event_date].copy()
    injuries_up_to_date = injuries[injuries['event_date'] == event_date].copy()
    starters_up_to_date = starters[starters['date'] == event_date].copy()

    offense_up_to_date = offense_up_to_date.drop(['player_name','team'], axis=1)
    defense_up_to_date = defense_up_to_date.drop(['player','team'], axis=1)
    kicking_up_to_date = kicking_up_to_date.drop(['player_name','team'], axis=1)
    returns_up_to_date = returns_up_to_date.drop(['player_name','team'], axis=1)
    snap_counts_up_to_date = snap_counts_up_to_date.copy()

    # Merge all stats into one dataset with suffixes to avoid column conflicts
    merged_stats = pd.merge(offense_up_to_date, defense_up_to_date, on=['player_id', 'date'], how='outer', suffixes=('_off', '_def'))
    merged_stats = pd.merge(merged_stats, kicking_up_to_date, on=['player_id', 'date'], how='outer', suffixes=('', '_kick'))
    merged_stats = pd.merge(merged_stats, returns_up_to_date, on=['player_id', 'date'], how='outer', suffixes=('', '_ret'))
    merged_stats = pd.merge(merged_stats, snap_counts_up_to_date,on=['player_id', 'date'], how='outer', suffixes=('', '_snap'))
    merged_stats = pd.merge(merged_stats, injuries_up_to_date[['player_id', 'game_status']], on=['player_id'], how='left')
    merged_stats["game_status"] = merged_stats["game_status"].fillna('Available')

    # Set Pandas option to handle the FutureWarning
    try:
        pd.set_option('future.no_silent_downcasting', True)
    except Exception as e:
        pass

    # Join with the roster to get player positions
    merged_data = pd.merge(merged_stats, roster_data[['player_id','team_abb','years_played','season']], on=['player_id','team_abb','season'], how='left')
    merged_data['is_starter'] = merged_data['player_id'].apply(lambda x: 1 if x in starters_up_to_date['player_id'].values else 0)

    # beside active and available, most columns are empty:
    merged_data['active'] = merged_data['player_id'].apply(lambda x: 0 if x in injuries_up_to_date.loc[injuries_up_to_date['active_inactive'] == 'Out', 'player_id'].values else 1)
    merged_data['game_status_available'] = merged_data['player_id'].apply(lambda x: 1 if x in injuries_up_to_date.loc[injuries_up_to_date['game_status'] == 'Available', 'player_id'].values else 0)
    merged_data['game_status_out'] = merged_data['player_id'].apply(lambda x: 1 if x in injuries_up_to_date.loc[injuries_up_to_date['game_status'] == 'Out', 'player_id'].values else 0)
    merged_data['game_status_probable'] = merged_data['player_id'].apply(lambda x: 1 if x in injuries_up_to_date.loc[injuries_up_to_date['game_status'] == 'Probable', 'player_id'].values else 0)
    merged_data['game_status_doubtful'] = merged_data['player_id'].apply(lambda x: 1 if x in injuries_up_to_date.loc[injuries_up_to_date['game_status'] == 'Doubtful', 'player_id'].values else 0)
    merged_data['game_status_injured_reserve'] = merged_data['player_id'].apply(lambda x: 1 if x in injuries_up_to_date.loc[injuries_up_to_date['game_status'].isin(['Injured Reserve', 'Injured Rese']), 'player_id'].values else 0)
    merged_data['game_status_questionable'] = merged_data['player_id'].apply(lambda x: 1 if x in injuries_up_to_date.loc[injuries_up_to_date['game_status'] == 'Questionable', 'player_id'].values else 0)
    merged_data['game_status_unable_to_perform'] = merged_data['player_id'].apply(lambda x: 1 if x in injuries_up_to_date.loc[injuries_up_to_date['game_status'] == 'Physically Unable To Perform', 'player_id'].values else 0)
    merged_data['game_status_suspended'] = merged_data['player_id'].apply(lambda x: 1 if x in injuries_up_to_date.loc[injuries_up_to_date['game_status'] == 'Suspended', 'player_id'].values else 0)
    merged_data['game_status_day_to_day'] = merged_data['player_id'].apply(lambda x: 1 if x in injuries_up_to_date.loc[injuries_up_to_date['game_status'] == 'Day-To-Day', 'player_id'].values else 0)
    merged_data['game_status_reserve'] = merged_data['player_id'].apply(lambda x: 1 if x in injuries_up_to_date.loc[injuries_up_to_date['game_status'] == 'Reserve/Futur', 'player_id'].values else 0)
    merged_data['game_status_unknown'] = merged_data['player_id'].apply(lambda x: 1 if x in injuries_up_to_date.loc[injuries_up_to_date['game_status'] == 'Unknown', 'player_id'].values else 0)
    
    merged_data['injury_weight'] = merged_data['game_status'].map(injury_weight_map)
    merged_data['game_outcome_adjustment'] = merged_data['outcome'].map(outcome_factor)
    merged_data = merged_data.fillna(0).infer_objects(copy=False)

    return merged_data

# Function to calculate weighted rating for each player based on custom weights
def calculate_player_rating(row, weights_dict):
    rating = 0
    for feature, weight in weights_dict.items():
        if feature in row:
            rating += float(row[feature]) * weight
    # Add the injury weight
    if 'injury_weight' in row:
        rating += row['injury_weight']
    if 'game_outcome_adjustment' in row:
        rating *= row['game_outcome_adjustment']
    return rating

# Function to calculate weighted rating for each player based on calcualted overall wieghts 
def calculate_player_rating_overall(row, weights_overall):
    stat_weights_overall_dict = dict(zip(weights_overall['statistic'], weights_overall['weight']))   
    rating = 0
    for feature, weight in stat_weights_overall_dict.items():
        if feature in row:
            rating += float(row[feature]) * weight
    return rating


# Function to calculate weighted rating for each player based on position
def calculate_player_rating_with_pos(row,stat_weights):
    df = stat_weights[stat_weights['position'] == row['position']]
    stat_weight_dict = dict(zip(df['statistic'], df['weight']))    
    rating = 0
    for feature, weight in stat_weight_dict.items():
        if feature in row:
            rating += float(row[feature]) * weight
    return rating


# Function to calculate player rating incorporating previous ratings and outcomes
def calculate_player_rating_with_previous(row, previous_ratings, stat_weights, stat_weights_overall, weights_dict,pos_weight=True, overall=False):
    if pos_weight:
        current_rating = calculate_player_rating_with_pos(row,stat_weights)
    elif overall:
        current_rating = calculate_player_rating_overall(row, stat_weights_overall)   
    else:
        current_rating = calculate_player_rating(row,weights_dict)
        
    
    # Incorporate previous rating if available
    previous_rating = previous_ratings.get(row['player_id'], current_rating)
    
    # Calculate final rating as a weighted average of the previous rating and the current rating
    final_rating = previous_rating_weight * previous_rating + (1 - previous_rating_weight) * current_rating
    return final_rating


# Function to calculate player rating
def players_rating(player_offense, player_defense, player_kicking, player_returns, snap_counts,rosters, injuries, starters_df,games_info,team_dict,team_dict_short,pos_alias_dict,stat_weights,stat_weights_overall,weight_dict,pos_weight=False,overall=False):
    #Clean data
    snap_counts,rosters, injuries, starters_df,games_info = data_cleaning_formatting(snap_counts,rosters, injuries, starters_df,games_info,team_dict,team_dict_short)
    
    # Replace the values in the 'position' column based on the mapping
    snap_counts['position'] = snap_counts['position'].map(pos_alias_dict)

    # Apply the ratings to all players
    unique_game_dates = games_info.sort_values(by='event_date')['event_date'].unique() # for custom weights
    if pos_weight or overall:
        games_info = games_info[games_info['season']>= 2015]
        unique_game_dates = games_info.sort_values(by='event_date')['event_date'].unique() # for calculated weights

    # Initialize dictionary to store previous ratings for each player
    previous_ratings = {}

    # Apply ratings to all players, incorporating win/loss and previous ratings
    player_ratings_df = pd.DataFrame()

    for i, event_date in enumerate(unique_game_dates):
        if i % 100 == 0:
            print(f"Step Number {i} of {len(unique_game_dates)}")
        
        merged_data_for_date = merge_data_for_event_date(event_date, player_offense, player_defense, player_kicking, player_returns, snap_counts,rosters, injuries, starters_df)
        merged_data_for_date['rating'] = merged_data_for_date.apply(lambda row: calculate_player_rating_with_previous(row, previous_ratings, stat_weights ,stat_weights_overall,weight_dict,pos_weight=pos_weight,overall=overall), axis=1)
        previous_ratings.update(merged_data_for_date.set_index('player_id')['rating'].to_dict())
        player_ratings_df = pd.concat([player_ratings_df, merged_data_for_date[['season','date','team_abb','player_id','player','position','rating']]])

    # Reset index and display the result
    player_ratings_df.reset_index(drop=True, inplace=True)

    return player_ratings_df


# files used for player rating calculation
filepath = f"{script_dir}../data"

injuries = pd.read_csv(f"{filepath}/injuries_all.csv")
rosters = pd.read_csv(f"{filepath}/Rosters.csv")
games_info = pd.read_csv(f"{filepath}/games_info_all.csv")
starters_df = pd.read_csv(f"{filepath}/game_starters_all.csv")
snap_counts = pd.read_csv(f"{filepath}/snap_counts_all.csv")
player_offense = pd.read_csv(f"{filepath}/player_offense_all.csv")
player_defense = pd.read_csv(f"{filepath}/player_defense_all.csv")
player_returns = pd.read_csv(f"{filepath}/player_returns_all.csv")
player_kicking = pd.read_csv(f"{filepath}/player_kicking_all.csv")
stat_weights = pd.read_csv(f"{filepath}/stat_weights_by_position.csv")
stat_weights_overall = pd.read_csv(f"{filepath}/stat_weights_overall.csv")

# Reading the JSON file as a dictionary
with open(f"{filepath}/team_dict.json", 'r') as file:
    team_dict = json.load(file)

with open(f"{filepath}/team_dict_short.json", 'r') as file:
    team_dict_short = json.load(file)

with open(f"{filepath}/position_alias.json", 'r') as file:
    pos_alias_dict = json.load(file)

with open(f"{filepath}/custom_weights.json", 'r') as file:
    custom_weights_dict = json.load(file)



pr = players_rating(player_offense, player_defense, player_kicking, player_returns, snap_counts,rosters, injuries, starters_df,games_info,team_dict,team_dict_short,pos_alias_dict,stat_weights,stat_weights_overall,custom_weights_dict,pos_weight=True,overall=True)

pr.to_csv(f"{filepath}/player_ratings.csv", index=False)