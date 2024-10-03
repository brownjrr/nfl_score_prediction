import json
import pandas as pd

#TODO Consider making this file a top-level file and the constants nested
with open('../data/position_alias.json', 'r') as file:
    position_alias = json.load(file)

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
    roster_players = df.shape[0]
    df['player_rank'] = [round(random.uniform(1, 50), 1) for _ in range(roster_players)]

    # Add in the team_id using regex
    pattern = r'^.*/teams/([a-z]{3}).*roster.htm$'
    df['team_id'] = (df['roster_link']
                             .apply(lambda x: 
                                    re.findall(pattern, x)[0]
                                   )
                            )

    # Clean up years_played column
    df.loc[df['years_played']=='Rook', 'years_played'] = 0
    df['years_played'] = df['years_played'].astype(int)
    df = df.loc[df['years_played']>=0,]
    
    return df


# 3. Trim our roster features
def yearly_team_roster(roster_df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects columns necessary to use for the roster
    args:
      roster_df - output from prep_rosters_df
          - key column ['roster_year', 'team_id']
          - player rank is either 'player_rank' or 'approx_value'
    """
    roster_cols = ['roster_year', 'team_id',
                   'player_id', 'player_name',
                   'approx_value', 'years_played',
                   'is_starter', 'age', 'position',
                   'weight', 'games_started', 'games_played']
    roster_df['position_alias'] = roster_df['position'].map(lambda x: position_alias.get(x))
    return roster_df[roster_cols]