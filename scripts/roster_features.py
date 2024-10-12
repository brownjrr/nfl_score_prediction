#%%
import json
import pandas as pd
import random
import re

#TODO Consider making this file a top-level file and the constants nested
def json_loader(file_loc:str) -> dict[str: str]:
    """
    Reads in some json files
    """
    with open(file_loc, 'r') as file:
        return json.load(file)
    

position_alias = json_loader('data/position_alias.json')
team_dict = json_loader('data/team_dict.json')
position_df_tmp = pd.read_csv('data/positions.csv')
position_category = dict(
    zip(position_df_tmp['position'], 
        position_df_tmp['position_type']
        )
        )

def team_only_dict(team_alias_dict: dict[str: str]) -> dict[str: str]:
    """
    We want to be able to update the names
    """
    team_only_names = [t.split()[-1] for t in team_alias_dict.keys()]
    
    # We need to rename the 'Team' to 'Football Team' since that was a season of a rename
    team_only_names[-2] = 'Football Team'
    team_only_dict =  {team_only_key: team_dict[team_key] 
                       for (team_key, team_only_key) in 
                       zip(team_alias_dict.keys(), team_only_names)}
    
    return team_only_dict

team_only_alias_dict = team_only_dict(team_dict)

#%%
# 2. Players table
def prep_rosters_df(file_loc:str) -> pd.DataFrame:
    """
    Read in roster information containing players per team per year
    Default file loc: data/rosters.csv
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

#%%
# 3. Trim our roster features
def yearly_team_roster(roster_df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects columns necessary to use for the roster
    args:
      roster_df - output from prep_rosters_df
          - key column ['roster_year', 'team_id']
          - player rank is either 'player_rank' or 'approx_value'
    """
    roster_df['position_alias'] = roster_df['position'].map(lambda x: position_alias.get(x))
    roster_cols = ['roster_year', 'team_id',
                   'player_id', 'player_name',
                   'approx_value', 'years_played',
                   'is_starter', 'age', 'position', 'position_alias',
                   'weight', 'games_started', 'games_played']

    return roster_df[roster_cols]
# %%
# 4. Ingest our per-game roster
def per_game_roster(file_loc: str) -> pd.DataFrame:
    """
    This reads in the base data that has game level data
    file_loc: data/game_starters_all.csv
    output: dataframe with keycols:
        - player_id (like FitzLa00)
        - player position and position_alias
        - team_id
        - event_date
    """
    df = pd.read_csv(file_loc)
    df.drop_duplicates(subset=['Date', 'Player_id', 'Team'], keep='first', inplace=True)
    df.rename(columns={'Player_id': 'player_id', 'Position': 'position'}, inplace=True)

    df['team_id'] = df['Team'].map(team_only_alias_dict.get)
    df['event_date'] = df['Date'].astype(str)
    df['roster_year'] = df['event_date'].apply(lambda x: x.split('-')[0]).astype(int)
    df['position_alias'] = df['position'].map(position_alias.get)
    # Fill in  missing positions with NA
    df['position_alias'] = df['position_alias'].fillna('UNKNOWN')
    df['position_cat'] = df['position'].map(position_category.get)

    return df[['team_id', 'player_id', 'position', 
               'position_alias', 'event_date', 
               'roster_year', 'position_cat']]
    
# %%
if __name__ == '__main__':
    yearly_roster = yearly_team_roster(prep_rosters_df('data/rosters.csv'))
    # We can join the two using [roster_year, player_id]
    game_roster = per_game_roster('data/game_starters_all.csv')

    # Assign the player's yearly value to the main roster
    roster_df = (game_roster
                 .merge(
                     yearly_roster[[
                         'roster_year',
                         'player_id',
                         'approx_value'
                         ]],
                         how='left',
                         on=['roster_year', 'player_id']
                        )
                )
    roster_df.to_csv('data/intermediate/roster_df.csv', index=False)
    print(f'Successfully processed {len(roster_df)} players for the roster.')
