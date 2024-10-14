import pandas as pd
import os
import math


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

print(f"script_dir: {script_dir}")

# custom weights for coaching statistics
stat_weights = {
    'avg_score_team': 1, 'avg_score_opp': -1, 'avg_off_first_downs': 0.6,
    'avg_off_yds_total': 1, 'avg_off_yds_pass': 1, 'avg_off_yds_rush': 1,
    'avg_off_timeout': 0.01, 'avg_def_first_downs': -0.5, 'avg_def_yds_total': -1,
    'avg_def_yds_pass': -1, 'avg_def_yds_rush': -1, 'avg_def_timeout': 0.01, 'avg_punts': -0.1,
    'avg_field_goals': 0.6, 'avg_run_plays': 0.5, 'avg_pass_plays': 0.5,
    'avg_off_penalties': -0.1, 'avg_forced_punts': 0.5, 'avg_fumbles': 0.7,
    'avg_interceptions': 0.7, 'avg_def_penalties': -0.1, 'avg_point_diff': 1,
}

# variable that defines how much importance the previous
# coach rating has in their new rating
PREV_RATING_WEIGHT = 0.5

def get_coach_ratings():
    df = pd.read_csv(script_dir+"../data/game_level_coach_data_extended.csv")

    week_dict = {
        'Wild Card': 100, 
        'Division': 101,
        'Conf. Champ.': 102,
        'SuperBowl': 103
    }

    df['week'] = pd.to_numeric(df['week'].apply(lambda x: week_dict[x] if x in week_dict else x))

    print(df.columns)

    # find individual match dates
    dates = sorted(df['event_date'].unique())
    
    # grabbing every unique combo of season/week
    season_week_combos = df[['season', 'week']].drop_duplicates().sort_values(by=['season', 'week']).values

    # creating a dictionary with initial weights for all coaches
    previous_ratings = {i:100 for i in df['coach_id'].unique()}

    def get_coach_rating(row):
        rating = 0
        for feature, weight in stat_weights.items():
            if feature in row and not math.isnan(row[feature]):
                rating += float(row[feature])*weight

        # Incorporate previous rating if available
        prev_rating = previous_ratings[row['coach_id']]
        
        # Calculate final rating as a weighted average of the previous rating and the current rating
        final_rating = PREV_RATING_WEIGHT * prev_rating + (1 - PREV_RATING_WEIGHT) * rating

        return final_rating

    dfs = []
    for season, week in season_week_combos:
        temp_df = df[(df['season']==season) & (df['week']==week)]
        temp_df['rating'] = temp_df.apply(get_coach_rating, axis=1)
        
        dfs.append(temp_df)
    
    new_df = pd.concat(dfs)

    new_df = new_df[[
        'coach_name', 'coach_id', 'team_id', 'season', 
        'boxscore_id', 'event_date', 'week', 'home_team_id', 
        'opp_team_id', 'rating',
    ]]
    
    new_df.to_csv(script_dir+"../data/coach_ratings.csv", index=False)

def analyze_coach_ratings():
    df = pd.read_csv(script_dir+"../data/coach_ratings.csv")

    print(df.columns)

    for season in sorted(df['season'].unique()):
        print(f"Season: {season}")

        temp_df = df[df['season']==season].groupby(['coach_id', 'coach_name'])['rating'].mean().sort_values(ascending=False)
        
        print(f"Top 10 Coaches:\n{temp_df.head(10)}")
        print(f"Bottom 10 Coaches:\n{temp_df.tail(10)}")
        print("==============================================================")


if __name__ == '__main__':
    get_coach_ratings()
    # analyze_coach_ratings()
