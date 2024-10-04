#%%
import pandas as pd
import random
import re

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
    df['team'] = df['team'].str.lower()
    return df


#%%
# 2. Teams table
def prep_games_df(file_loc: str) -> pd.DataFrame:
    """
    Read in teams.csv to merge
    """

    df = pd.read_csv(file_loc)

    # Create a team_name dict
    team_dict = (df.groupby('team_name')
                 .agg({'team_id': 'first'})['team_id']
                 .to_dict()
                )

    # Add in opposiing team name
    df['opp_team_id'] = df['opp'].apply(lambda x: team_dict.get(x))
    df.rename(columns={'team_abbr': 'home_team_id'}, inplace=True)

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
        left_on = ['home_team_id', 'year'],
        right_on = ['team', 'season'],
        suffixes=[None, '_home']
    )

    # Opposing team
    df = pd.merge(
        tdf,
        coach_df,
        how='left',
        left_on = ['opp_team_id', 'year'],
        right_on = ['team', 'season'],
        suffixes=[None, '_opp']
    )

    return df
# %%
if __name__ == '__main__':
    coach_df = prep_coaches_df('data/coaches.csv')
    game_df = prep_games_df('data/teams.csv')
    game_df = team_coach_merge(game_df=game_df, coach_df=coach_df)
    try:
        game_df.drop(columns=['Unnamed: 0'])
    except KeyError as e:
        print(e)
    print(f'Successfully processed {len(game_df)} games.')
    game_df.to_csv('data/intermediate/games_df.csv',index=False)