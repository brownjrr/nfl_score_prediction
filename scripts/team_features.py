import pandas as pd
import random
import re


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



# 2. Teams table
def prep_teams_df(file_loc: str) -> pd.DataFrame:
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
    df.dropna(subset=numeric_columns, inplace=True)

    return df





# 5. Assign players to the teams
def team_coach_merge(team_df: pd.DataFrame,
                     coach_df: pd.DataFrame
                    ) -> pd.DataFrame:
    """
    This takes 2 dataframes and merges them together to make a combined dataset
    for the teams
    args:
        - team_df: output from the prep_teams_df
        - coach_df: output from the prep_coaches_df

    output:
        - merged dataframe based on coach information
    """
    df = pd.merge(
        team_df,
        coach_df,
        left_on = ['home_team_id', 
    )