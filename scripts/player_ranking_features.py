#%%
import pandas as pd
from random import random

# Trim our games_df to retain game level information
# First set up our home team and players - split into offense and defense
def home_away_games(df: pd.DataFrame,
                    player_df: pd.DataFrame,
                    cols: list[str, str, str]
                    ) -> pd.DataFrame:
    """
    ARGS:
        df -- Takes in the intermediate step from the games section `team_features.py`
            This then breaks the games dataframe into either home (0) or away (1).
        player_df -- This also requires the roster information 
            from the intermediate step from `roster_features.py`
        cols -- The columns needed for a join.
            ['game_id', '[home|opp]_team_id', 'event_date']
        home_away -- specifies if we are outputting the home or away data frame
    **
    ROSTER INFORMATION MUST CONTAIN PLAYER RANK
    **

    Returns a dataframe that has adjusted column names and merges the players and games
    """

    loc_games_df = df.loc[:, cols]
    loc_players = loc_games_df.merge(
        player_df,
        left_on=cols[1:],
        right_on=['team_id', 'event_date']
    )

    loc_players_off = loc_players.loc[loc_players['position_cat'] == 'offense']
    loc_players_def = loc_players.loc[loc_players['position_cat'] == 'defense']

    return loc_players_off, loc_players_def

def rename_position_alias(df: pd.DataFrame, new_column: str) -> pd.DataFrame:
    """
    Simply renames the position_alias column
    """
    return df.rename(columns={'position_alias': new_column})
#%%

#%%
# Determine the players, positions, and scores, per game
# We can roll this up by game by using:
#    - groupby('game_id', '[home|opp]_team_id)
#    - aggregate by 'player_id', '[off|def]_position', 'ranking|expected_value'

home_off_players_positional_ranking = (home_players_off
                                   .groupby(['game_id', 'home_team_id'])
                                   .agg({
                                       'player_id': list,
                                       'off_position': list,
                                       'approx_value': list
                                   })
)
home_off_players_positional_ranking = (home_off_players_positional_ranking
                                        .rename(columns={
                                            'player_id': 'home_off_player_id',
                                            'approx_value': 'home_off_rank'
                                            }
                                        )
                                    )

opp_def_players_positional_ranking = (opp_players_def
                                   .groupby(['game_id', 'opp_team_id'])
                                   .agg({
                                       'player_id': list,
                                       'def_position': list,
                                       'approx_value': list
                                   })
)

opp_def_players_positional_ranking = (opp_def_players_positional_ranking
                                        .rename(columns={
                                            'player_id': 'opp_def_player_id',
                                            'approx_value': 'opp_def_rank'
                                            }
                                        )
                                    )

#%%
home_team_off_interaction = home_off_players_positional_ranking.merge(
    opp_def_players_positional_ranking,
    on='game_id'
)
opp_team_off_interaction = opp_off_players_positional_ranking.merge(
    opp_def_players_positional_ranking,
    on='game_id'
)
# Game and player match on ['home|opp_team_id', event_date]
random_game = home_team_off_interaction.sample(1)

#%%
def probability_dictionary(file_loc: str) -> dict:
    '''
    Reads in the probability dataframe and converts to a dictionary
    '''
    prob_df = pd.read_csv(file_loc, index_col=0)
    prob_dict = {f'{row}{col}': prob_df.loc[row, col]
                 for row in prob_df.index
                 for col in prob_df.columns}
    
    return prob_dict

#%%
def position_player_interaction(pos_A, rank_A, pos_B, rank_B, pos_p):
    """
    Takes in the different positions from our dataframe, basically the lineup for the game
    Compares the different interaction results and returns the probability * rank

    args:
        pos_A - Offensive positions for the team. e.g., [QB, RB, WR, WR, TE]
        pos_B - Defensive positions for the team. e.g., [LB, LB, LB, DB, DB]
        pos_p (dict) - Probability index of interaction between the two positions
        rank_A - ranking of the offesnive player, e.g. [6.0, 10.0, 11.0, 13.0, 8.0]
        rank_B - ranking of the defensive player, e.g. [13.0, 5.0, 9.0, 6.0, 7.0]
    """

    team_A_rankings = {}
    for p_A, r_A in zip(pos_A, rank_A):
        pos_rank = 0
        for p_B, r_B in zip(pos_B, rank_B):
            pos_combo = p_A + p_B # This will give us the OFF-DEF interaction pair
            p_AB = pos_p.get(pos_combo)
            rank_difference = r_A - r_B
            weighted_rank = p_AB*rank_difference
            pos_rank += weighted_rank
            #print(f'Measuring rank between {p_B} -- rank difference of {rank_difference}')
            #print(f'Weighted rank is {weighted_rank}')
        #print(f'Final ranking between {p_A} and other team is {pos_rank}')
        team_A_rankings[p_A] = team_A_rankings.get(p_A, 0) + pos_rank

    return team_A_rankings

#%%
# Test cell
interaction_columns = ['off_position', 'home_off_rank', 'def_position', 'opp_def_rank']
prob_dict = probability_dictionary('../data/interaction_prob.csv')
random_game['off_pos_strength'] = (random_game[interaction_columns]
                                   .apply(lambda i_cols: 
                                          position_player_interaction(i_cols.iloc[0],
                                                                      i_cols.iloc[1],
                                                                      i_cols.iloc[2],
                                                                      i_cols.iloc[3],
                                                                      prob_dict
                                                                      ),
                                          axis=1
                                          )
                                    )

# %%
# Apply the interaction to all games to get a "home_rank" and "opp_rank"
home_team_off_interaction['home_strength'] = (home_team_off_interaction[interaction_columns]
                                                .apply(lambda i_cols:
                                                        position_player_interaction(i_cols.iloc[0],
                                                                                    i_cols.iloc[1],
                                                                                    i_cols.iloc[2],
                                                                                    i_cols.iloc[3],
                                                                                    prob_dict
                                                                                    ),
                                                        axis=1
                                                    )
                                                )

if __name__ == '__main__':
    # First read in the games (output from step 1)
    model_games_df = pd.read_csv('../data/intermediate/games_df.csv')
    players_df = pd.read_csv('../data/intermediate/roster_df.csv')
    model_games_df['game_id'] = model_games_df[
        ['home_team_id',
        'opp_team_id',
        'event_date'
        ]].apply(lambda x: "_".join(x.values.astype(str)), axis=1)
    
    # Determine which players are playing in what games
    # Split up into offense and defense
    #%%
    home_players_off, home_players_def = home_away_games(
        df=model_games_df,
        player_df=players_df,
        cols=['game_id', 'home_team_id', 'event_date']
    )

    opp_players_off, opp_players_def = home_players_def = home_away_games(
        df=model_games_df,
        player_df=players_df,
        cols=['game_id', 'opp_team_id', 'event_date']
    )
    #%%
## Rename columns for merge
home_players_off = rename_position_alias(home_players_off, 'off_position')
home_players_def = rename_position_alias(home_players_def, 'def_position')
opp_players_off = rename_position_alias(opp_players_off, 'off_position')
opp_players_def = rename_position_alias(opp_players_def, 'def_position')