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
    return coach_df

# 2. Players table
def prep_rosters_df(file_loc:str) -> pd.DataFrame:
    """
    Read in roster information containing players per team per year
    Need to add in key team_id -> e.g. Cardinal: crd
    #TODO: For now - we'll use a dummy variable from "dummy_players_rank.csv"
    Important column will be player_rank
    """

    df = pd.read_csv(file_loc)
    # For testing purposes
    random.seed(42)

    # Generate a dummy ranking for each player on the roster
    roster_players = roster_df.shape[0]
    df['player_rank'] = [round(random.uniform(1, 50), 1) for _ in range(roster_players]

    # Add in the team_id using regex
    pattern = r'^.*/teams/([a-z]{3}).*roster.htm$'
    rosters_df['team_id'] = (roster_df['roster_link']
                             .apply(lambda x: 
                                    re.findall(pattern, x)[0]
                                   )
                            )

# 3. Teams table
def prep_teams_df(file_loc: str) -> pd.DataFrame:
    """
    Read in teams_df to merge
    """

    df = pd.read_csv(file_loc)

    # Create a team_name dict
    team_dict = (df.groupby('team_name')
                 .agg({'team_id': 'first'})['team_id']
                 .to_dict()
                )

    # Add in opposiing team name
    df['opp_id'] = df['opp'].apply(lambda x: team_dict.get(x))

    # Split the record
    df[['wins', 'losses', 'ties']] = teams_df['rec'].str.split('-', expand=True)
    df[['over_under_value', 'over_under_cat']] = (teams_df['over_under']
                                                  .str
                                                  .extract(r'(\d+\.?\d*) \((\w+)\)'))
    return df
    
    
    