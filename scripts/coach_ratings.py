import pandas as pd
import re
from play_by_play_augment import get_position_type_dicts

def get_games_dataframe():
    df = pd.read_csv("../data/teams.csv")
    df['boxscore_id'] = df['boxscore_stub'].str.split("/", expand=False).str[-1].str.replace(".htm", "")
    return df

def get_play_by_play_table():
    _, inv_pos_dict = get_position_type_dicts()
    df = pd.read_csv("../data/play_by_play_extended_v2.csv")
    
    df['intercept_on_play'] = df['play_text'].str.contains(("intercept"))
    df['fumble_on_play'] = df['play_text'].str.contains(("fumble"))

    pattern = r'penalty on (.*?):'
    def get_offensive_defensive_penalty(x):
        x = x.lower()
        if "penalty on" in x:
            penalties = re.findall(pattern, x)

            penalty_on = set()

            for i in penalties:
                i=i.upper()

                if i in inv_pos_dict:
                    penalty_on.add(inv_pos_dict[i])
            
            if len(penalty_on)==0 or len(penalty_on)>1:
                return None
            else:
                return list(penalty_on)[0]
        else:
            return None
    
    df['penalty_on'] = df['play_text'].apply(get_offensive_defensive_penalty)

    return df

def get_coaches_table():
    df = pd.read_csv("../data/coaches.csv")
    return df

def get_game_level_coach_data():
    df = pd.read_csv("../data/game_level_coach_data.csv")
    return df

def get_rolling_avg(df, col):
    if col.startswith("num_"):
        avg_col = col.replace("num", "avg")
    else:
        avg_col = f"avg_{col}"

    rolling_means = []
    for i, ind in enumerate(df.index):
        temp_df = df.iloc[:i]
        try:
            rolling_means.append(temp_df[col].mean())
        except Exception as e:
            print(f"Column: {col}")
            print(f"Values: {temp_df[col].values}")
            assert False, f"{e}"

    df[avg_col] = rolling_means

    return df

def create_additional_coach_stats():
    agg_coach_df = get_coaches_table()
    coach_game_df = get_game_level_coach_data()
    games_df = get_games_dataframe()
    merged_df = coach_game_df.merge(
        agg_coach_df, 
        left_on=['coach_name', 'season'], 
        right_on=['Coach', 'Season'], 
        how="inner"
    )
    merged_df = merged_df.merge(
        games_df, 
        on=['boxscore_id', 'team_id'], 
        how="inner"
    ).sort_values(by=['boxscore_id'])

    play_df = get_play_by_play_table()

    def get_coach_stats(row):
        team_id, boxscore_id = row['team_id'], row['boxscore_id']
        temp_df = play_df[play_df['boxscore_id']==boxscore_id]

        num_punts = temp_df[(temp_df['offensive_team']==team_id) & (temp_df['punt']==True)].shape[0]
        num_field_goals = temp_df[(temp_df['offensive_team']==team_id) & (temp_df['field_goal']==True)].shape[0]
        num_running_plays = temp_df[(temp_df['offensive_team']==team_id) & (temp_df['running_play']==True)].shape[0]
        num_passing_plays = temp_df[(temp_df['offensive_team']==team_id) & (temp_df['passing_play']==True)].shape[0]
        num_off_penalties = temp_df[(temp_df['offensive_team']==team_id) & (temp_df['penalty_on']=='offense')].shape[0]

        num_forced_punts = temp_df[(temp_df['defensive_team']==team_id) & (temp_df['punt']==True)].shape[0]
        num_fumbles = temp_df[(temp_df['defensive_team']==team_id) & (temp_df['fumble_on_play']==True)].shape[0]
        num_interceptions = temp_df[(temp_df['defensive_team']==team_id) & (temp_df['intercept_on_play']==True)].shape[0]
        num_def_penalties = temp_df[(temp_df['defensive_team']==team_id) & (temp_df['penalty_on']=='defense')].shape[0]

        return num_punts, num_field_goals, num_running_plays, \
            num_passing_plays, num_off_penalties, num_forced_punts, \
                num_fumbles, num_interceptions, num_def_penalties
    
    merged_df['extra_stats'] = merged_df[['coach_id', 'team_id', 'boxscore_id']].apply(get_coach_stats, axis=1)
    merged_df[[
        'num_punts', 'num_field_goals', 'num_run_plays', 
        'num_pass_plays', 'num_off_penalties', 'num_forced_punts', 
        'num_fumbles', 'num_interceptions', 'num_def_penalties'
    ]] = pd.DataFrame(merged_df['extra_stats'].tolist(), index=merged_df.index)


    data_cols = [
        'score_team',
        'score_opp', 'off_first_downs', 'off_yds_total', 'off_yds_pass',
        'off_yds_rush', 'off_timeout', 'def_first_downs', 'def_yds_total',
        'def_yds_pass', 'def_yds_rush', 'def_timeout',
        'num_punts', 'num_field_goals', 'num_run_plays',
        'num_pass_plays', 'num_off_penalties', 'num_forced_punts',
        'num_fumbles', 'num_interceptions', 'num_def_penalties',
    ]

    def get_agg_stats(group):
        group = group.sort_values(by='event_date')
        
        for stat in data_cols:
            group[stat] = pd.to_numeric(group[stat])
            group = get_rolling_avg(group, stat)
        
        return group

    merged_df = merged_df.groupby(['coach_id']).apply(get_agg_stats)

    merged_df['avg_point_diff'] = merged_df['avg_score_team'] - merged_df['avg_score_opp']

    print(f"Final Shape: {merged_df.shape}")

    merged_df.to_csv("../data/game_level_coach_data_extended.csv", index=False)

def examine_extended_coach_table():
    df = pd.read_csv("../data/game_level_coach_data_extended.csv")
    
    print(df.columns)
    print(df)
    

    
if __name__ == "__main__":
    # get_play_by_play_table()
    # get_coaches_table()
    # get_game_level_coach_data()
    # get_games_dataframe()
    # create_additional_coach_stats()

    examine_extended_coach_table()
