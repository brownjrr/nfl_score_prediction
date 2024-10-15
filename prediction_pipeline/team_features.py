#%%
import pandas as pd
import json

#%%
# Define our feature engineering by grouping
# 1. Coaches table

def prep_coaches_df(file_loc:str) -> pd.DataFrame:
    """
    Reads in the coaches dataframe and ensures the columns
    are in the correct order
    """
    df = pd.read_csv(file_loc)
    df.columns = map(str.lower, df.columns)
    df['coach_index'] = df.index+1

    # Add in team mappings associated with the URL reference
    with open('data/abbrev_map.json', 'r') as file:
        team_mappings = json.load(file)

    df['team_url_id'] = df['team'].map(team_mappings.get)
    return df

def read_in_coach_ratings(file_loc:str) -> pd.DataFrame:
    """
    Reads in the output from coach_rating model
    """
    df = pd.read_csv(file_loc)
    df = df.rename(columns={'rating': 'coach_rating'})
    COACH_COLUMNS = [
        'coach_name',
        'coach_id',
        'team_id',
        'season',
        'event_date',
        'week',
        'coach_rating'
        ]
    return df[COACH_COLUMNS]


#%%
def replace_na_mean(df_col: pd.Series):
    """
    Reads in a series and returns the average for the column
    """
    return df_col.mean()

#%%
# 2. Teams table
def prep_games_df(file_loc: str) -> pd.DataFrame:
    """
    Read in teams.csv to merge
    """

    df = pd.read_csv(file_loc)

    # Drop records for duplicate games
    df = df.drop_duplicates(['boxscore_stub'])

    # calculate home_team_id and opp_team_id based on boxscore_stub
    df['boxscore_team_abbr'] = df['boxscore_stub'].str.split("/", expand=False).str[-1].str.replace(".htm", "").str[-3:]
    df['temp_results'] = df.apply(lambda row: ((row['team_id'], row['score_team']), (row['opp_team_id'], row['score_opp'])) if row['boxscore_team_abbr']==row['team_id'] else ((row['opp_team_id'], row['score_opp']), (row['team_id'], row['score_team'])), axis=1)
    df[['home_team_id', 'opp_team_id']] = pd.DataFrame(df['temp_results'].tolist(), index=df.index)
    df[['home_team_id', 'score_team']] = pd.DataFrame(df['home_team_id'].tolist(), index=df.index)
    df[['opp_team_id', 'score_opp']] = pd.DataFrame(df['opp_team_id'].tolist(), index=df.index)

    # Create a team_name dict
    # team_dict = (df.groupby('team_name')
    #              .agg({'team_id': 'first'})['team_id']
    #              .to_dict()
    #             )

    # # Add in opposiing team name
    # df['opp_team_id'] = df['opp'].apply(lambda x: team_dict.get(x))
    # df.rename(columns={'team_abbr': 'home_team_id'}, inplace=True)

    # print(df[['score_team', 'score_opp', 'boxscore_stub', 'home_team_id', 'opp_team_id']])

    # Split the record
    df[['wins', 'losses', 'ties']] = df['rec'].str.split('-', expand=True)
    df[['over_under_value', 'over_under_cat']] = (df['over_under']
                                                  .str
                                                  .extract(r'(\d+\.?\d*) \((\w+)\)'))

    # Convert number columns to numbers
    # There were 2 games in 2022 that were cancelled
    # Remove those columns
    #non_numeric_mask = ~df['off_yds_total'].str.replace('.', '', regex=False).str.isnumeric()
    #non_numeric_values = df.loc[non_numeric_mask, ]

    score_columns = ['score_team', 'score_opp']
    numeric_columns = ['off_first_downs', 'off_yds_total',
                       'off_yds_pass', 'off_yds_rush', 'off_timeout', 'def_first_downs',
                       'def_yds_total', 'def_yds_pass', 'def_yds_rush', 'def_timeout',
                       'off_exp_pts', 'def_exp_pts', 'sptm_exp_pts']

    # If there is no score, we can't have a prediction
    df[score_columns] = df[score_columns].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=score_columns, inplace=True)

    # Clean up other numeric columns - fill with NA
    # For offensive groups - get the average of the teams NA
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # def_timeouts and off_timeouts have nulls (should be 0)
    df.loc[df['off_timeout'].isna(), 'off_timeout'] = 0
    df.loc[df['def_timeout'].isna(), 'def_timeout'] = 0

    # Convert categorical values to numbers
    days_of_week = {
        'Mon': 1,
        'Sun': 2,
        'Sat': 3,
        'Fri': 4,
        'Thu': 5,
        'Wed': 6,
        'Tue': 7
    }

    roof_types = {
        'outdoors': 1,
        'dome': 2,
        'retractable roof (closed)': 3,
        'retractable roof (open)': 4
    }
    df['day_int'] = df['day'].map(days_of_week.get)
    df['roof_type_int'] = df['roof_type'].map(roof_types.get)
    # A separate analysis shows most frequent roof type = 1 (outdoor)
    df['roof_type_int'] = (df['roof_type_int']
                           .fillna(1))
    

    # Replace NAs with the mean
    na_cols = ['humidity_pct',
               'wind_speed',
               'temperature',
               'duration',
               'attendance'
               ]
    df[na_cols] = df[na_cols].fillna(df[na_cols].mean())
    week_ind_map = {'1':1, '2':2, '3':3, '4': 4,
                      '5':5, '6':6, '7':7, '8':8,
                      '9':9, '10':10, '11':11, '12':12,
                      '13':13, '14':14, '15':15, '16':16,
                      '17':17, '18':18, 'Wild Card': 20,
                      'Division': 21, 'Conf. Champ': 22,
                      'SuperBowl': 23}
    
    df['week_ind'] = df['week'].map(week_ind_map.get)
    

    
    return df

#%%
# 3. Assign coach to the teams
def team_coach_merge(game_df: pd.DataFrame,
                     coach_df: pd.DataFrame
                    ) -> pd.DataFrame:
    """
    This takes 2 dataframes and merges them together to make a combined dataset
    for the teams
    args:
        - game_df: output from the prep_games_df
            - key: ['[home|opp]_team_id', 'year']
        - coach_df: output from the prep_coaches_df
            - key: ['team', 'season']

    output:
        - merged dataframe based on coach information
    """
    # Home team
    tdf = pd.merge(
        game_df,
        coach_df,
        how='left',
        left_on = ['home_team_id', 'event_date'],
        right_on = ['team_id', 'event_date'],
        suffixes=[None, '_home']
    )

    # Opposing team
    df = pd.merge(
        tdf,
        coach_df,
        how='left',
        left_on = ['opp_team_id', 'event_date'],
        right_on = ['team_id', 'event_date'],
        suffixes=[None, '_opp']
    )

    return df
# %%
if __name__ == '__main__':
    #coach_df = prep_coaches_df('data/coaches.csv')
    coach_df = read_in_coach_ratings('data/coach_ratings.csv')
    game_df = prep_games_df('data/teams.csv')
    game_df = team_coach_merge(game_df=game_df, coach_df=coach_df)
    try:
        game_df.drop(columns=['Unnamed: 0'])
    except KeyError as e:
        print(e)
    print(f'Successfully processed {len(game_df)} games.')
    game_df.to_csv('data/intermediate/games_df.csv',index=False)
# %%
