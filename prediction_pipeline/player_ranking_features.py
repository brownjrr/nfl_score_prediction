#%%
import numpy as np
import pandas as pd
#from random import random

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
    #print(f'Offense is of type {type(loc_players_off)}')
    #print(f'Defense is of type {type(loc_players_def)}')

    return (loc_players_off, loc_players_def)

def rename_position(df: pd.DataFrame, new_column: str) -> pd.DataFrame:
    """
    Simply renames the position column
    """
    return df.rename(columns={'position': new_column})


# Determine the players, positions, and scores, per game
# We can roll this up by game by using:
#    - groupby('game_id', '[home|opp]_team_id)
#    - aggregate by 'player_id', '[off|def]_position', 'ranking|expected_value'

def player_position_group(df: pd.DataFrame,
                          home_away_switch: int=0,
                          off_def_switch = 0
                         ) -> pd.DataFrame:
    """
    Transforms our game-players list to be one line per game

    Args:
    - df: dataframe that's a combination of the player-game function.
        - Output after `player_game_position`
    - home_away_switch: boolean integer describing home (0) or away (1)
    - off_def_switch: boolean integer describing offense (0) or defense(1)
    Returns:
    - player_id is a list of all players ['TaylTr0', 'BrayQu00', 'WattJ.00']
    - position is the position they are in (aliased): ['QB', 'WR', 'TE']
    - approx_value/rank is a list of the players rank: [14.5, 1.5, 20.1]
        We are using the player's rating from the previous game `prior_rating`
    """

    team_id_col = 'home_team_id' if home_away_switch == 0 else 'opp_team_id'

    if home_away_switch == 0:
        if off_def_switch == 0:
            player_col = 'home_off_player_id'
            rank_col = 'home_off_rank'
        elif off_def_switch == 1:
            player_col = 'home_def_player_id'
            rank_col = 'home_def_rank'
    elif home_away_switch == 1:
        if off_def_switch == 0:
            player_col = 'opp_off_player_id'
            rank_col = 'opp_off_rank'
        elif off_def_switch == 1:
            player_col = 'opp_def_player_id'
            rank_col = 'opp_def_rank'
    else:
        raise TypeError
        print('Switch values need to be 1 or 0')

    grouped_df = (df
                  .groupby(['game_id', team_id_col])
                  .agg({
                      'player_id': list,
                      'position': list,
                      'prior_rating': list
                  })
                )
    grouped_ranking = (grouped_df
                       .rename(columns={
                           'player_id': player_col,
                           'prior_rating': rank_col
                       })
                    )
    
    return grouped_ranking


def game_player_matchup_merge(df_off: pd.DataFrame,
                              df_def: pd.DataFrame
                              ) -> pd.DataFrame:
    
    """
    Combines two ranking dataframes into one

    args:
    df_off -- Offensive dataframe with one row per game
    of_def -- Defensive dataframe with one row per game

    output:
    Dataframe that contains the merged columns with off vs defense
    """
    return df_off.merge(df_def, on='game_id')


def probability_dictionary(file_loc: str) -> dict:
    '''
    Reads in the probability dataframe and converts to a dictionary
    '''
    prob_df = pd.read_csv(file_loc, index_col=0)
    prob_dict = {f'{row}{col}': prob_df.loc[row, col]
                 for row in prob_df.index
                 for col in prob_df.columns}
    
    return prob_dict


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
        if np.isnan(r_A):
            r_A = 0
            
        for p_B, r_B in zip(pos_B, rank_B):
            pos_combo = p_A + p_B # This will give us the OFF-DEF interaction pair
            p_AB = pos_p.get(pos_combo)

            # NA Handling
            if np.isnan(r_B):
                r_B=0
            rank_difference = r_A - r_B
            weighted_rank = p_AB*rank_difference
            pos_rank += weighted_rank
            #print(f'Measuring rank between {p_B} -- rank difference of {rank_difference}')
            #print(f'Weighted rank is {weighted_rank}')
        #print(f'Final ranking between {p_A} and other team is {pos_rank}')
        team_A_rankings[p_A] = team_A_rankings.get(p_A, 0) + pos_rank

    return team_A_rankings

if __name__ == '__main__':
    # First read in the games (output from step 1)
    model_games_df = pd.read_csv('data/intermediate/games_df.csv')
    players_df = pd.read_csv('data/intermediate/roster_df.csv')
    prob_dict = probability_dictionary('data/interaction_prob.csv')
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

    opp_players_off, opp_players_def = home_away_games(
        df=model_games_df,
        player_df=players_df,
        cols=['game_id', 'opp_team_id', 'event_date']
    )

    # Join by group and get the players, positions, and rankings per game
    home_off_players_positional_ranking = player_position_group(
        home_players_off,
        home_away_switch = 0,
        off_def_switch=0
    )

    home_def_players_positional_ranking = player_position_group(
        home_players_def,
        home_away_switch=0,
        off_def_switch=1
    )

    opp_off_players_positional_ranking = player_position_group(
        opp_players_off,
        home_away_switch=1,
        off_def_switch=0
    )

    opp_def_players_positional_ranking = player_position_group(
        opp_players_def,
        home_away_switch=1,
        off_def_switch=1
    )

    ## Rename columns for merge
    home_off_players_positional_ranking = rename_position(
        home_off_players_positional_ranking,
        'off_position'
        )
    home_def_players_positional_ranking = rename_position(
        home_def_players_positional_ranking,
        'def_position'
        )
    opp_off_players_positional_ranking = rename_position(
        opp_off_players_positional_ranking,
        'off_position'
        )
    opp_def_players_positional_ranking = rename_position(
        opp_def_players_positional_ranking,
        'def_position'
        )
    

    ## Merge the two dataframes together
    home_team_matchup = game_player_matchup_merge(
        home_off_players_positional_ranking,
        opp_def_players_positional_ranking
        )
    opp_team_matchup = game_player_matchup_merge(
        opp_off_players_positional_ranking,
        home_def_players_positional_ranking
        )

    ## Setup our interaction probability
    home_matchup_cols = ['off_position', 'home_off_rank', 'def_position', 'opp_def_rank']
    home_team_matchup['home_strength'] = (
        home_team_matchup[home_matchup_cols]
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

    away_matchup_cols = ['off_position', 'opp_off_rank', 'def_position', 'home_def_rank']
    opp_team_matchup['opp_strength'] = (
        opp_team_matchup[away_matchup_cols]
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

    print(home_team_matchup.head())
    print(opp_team_matchup.head())
# %%
    home_team_summary = home_team_matchup['home_strength'].apply(lambda x: sum(x.values()))
    opp_team_summary = opp_team_matchup['opp_strength'].apply(lambda x: sum(x.values()))
    game_strength_summary = pd.merge(home_team_summary, opp_team_summary, on='game_id').reset_index()
    game_strength_summary.to_csv('data/intermediate/game_rank_matchup.csv', index=False)
# %%
