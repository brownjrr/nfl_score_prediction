import pandas as pd
import glob
import re


aliases = {
    'WR': {'WR', 'R', 'FL', 'SE'},
    'LB': {
        'LB', 'MIKE', 'WLB', 'JACK', 'WILL', 'SAM', 'SLB', 
        'OLB', 'ILB', 'LILB', 'RILB', 'RLB', 'LLB', 'MLB', 'LOLB', 'ROLB',
        'RUSH', 'BLB'
    },
    'DB': {'DB', 'LCB', 'RCB', 'CB', 'FS', 'SS', 'S', 'RS', 'NB'},
    'TE': {'TE'},
    'K': {'K', 'PK'},
    'QB': {'QB'},
    'RB': {'RB', 'HB', 'FB', 'B', 'H', 'F'},
    'OL': {'OL', 'OG', 'T', 'RT', 'ROT', 'LT', 'OT', 'G', 'LG', 'RG', 'C', 'LS'},
    'P': {'P'},
    'DL': {'DL', 'NG', 'DE', 'LDE', 'RDE', 'DT', 'NT', 'LDT', 'RDT'},
    'PR': {'PR'},
    'UNKNOWN': {'D', 'L', 'KB', 'LA', 'BT', 'LD', 'TOG', 'GOC'},
}

# inverting aliases dictionary
inv_alias_dict = dict()
# inverting aliases dictionary
inv_alias_dict = dict()

for i in aliases:
    for j in aliases[i]:
        inv_alias_dict[j] = i
for i in aliases:
    for j in aliases[i]:
        inv_alias_dict[j] = i

print(f"inv_alias_dict:\n{inv_alias_dict}")

def check_all_pos_mapped():
    """
    Run a check to make sure all positions can be mapped to the aliases dictionary
    """
print(f"inv_alias_dict:\n{inv_alias_dict}")

def check_all_pos_mapped():
    """
    Run a check to make sure all positions can be mapped to the aliases dictionary
    """

    players_df = pd.read_csv("../data/players.csv")

    print(f"players_df:\n{players_df}")

    stat_files = glob.glob("../data/PlayerStats/*.csv")

    print(f"stat_files: {stat_files}")

    for stat_file in stat_files[:]:
        print(f"stat_file: {stat_file}")

        stat_df = pd.read_csv(stat_file)
        stat_df = stat_df[(~stat_df['pos'].astype(str).str.contains("Missed season")) & (~stat_df['pos'].astype(str).str.contains("Did not play"))]

        print(stat_df.columns)

        positions = stat_df['pos'].unique()
    
        missing_positions = []
        for i in positions:
            if isinstance(i, str):
                for j in re.split(r'[^a-zA-Z]', i):
                    missing_positions.append("" if j in inv_alias_dict else j)

        # missing_positions = ["" if j in inv_alias_dict else j for j in re.split(r'[^a-zA-Z]', i) for i in positions if isinstance(i, str)]

        # print([(i, type(i)) for i in positions])
        # print(positions)
        # print(missing_positions)

        for i in missing_positions:
            assert i == ''

        # final_list = []
        # for i in positions:

        print()

def transform_position_col(x):
    # extracting position label(s) from string
    positions = re.split(r'[^a-zA-Z]', x)
    positions = list({inv_alias_dict[i] for i in positions})

    return positions

def preprocess_stats_data():
    stat_files = glob.glob("../data/PlayerStats/*.csv")

    print(f"stat_files: {stat_files}")
    
    dfs = []
    for stat_file in stat_files:
        print(f"stat_file: {stat_file}")

        stat_df = pd.read_csv(stat_file)

        before_shape = stat_df.shape

        # drop rows where position is NaN
        stat_df = stat_df.dropna(subset=['pos'])

        # dropping rows where position value contains "Missed season" or "Did not play" 
        stat_df = stat_df[(~stat_df['pos'].astype(str).str.contains("Missed season")) & (~stat_df['pos'].astype(str).str.contains("Did not play"))]

        stat_df['pos'] = stat_df['pos'].apply(lambda x: transform_position_col(x))
        
        # dropping rows where player has more than one position listed
        stat_df = stat_df[~(stat_df['pos'].str.len()>1)]

        # converting "pos" column to str
        stat_df['pos'] = stat_df['pos'].apply(lambda x: x[0])
        
        # dropping rows where position = "UNKNOWN"
        stat_df = stat_df[~(stat_df['pos']=='UNKNOWN')]
        
        after_shape = stat_df.shape

        print(f"stat_df.shape (BEFORE|AFTER): {before_shape}|{after_shape}")
        stat_df.to_csv(stat_file, index=False)
        print("Finished Saving")

def update_player_positions():
    """
    Run this function to update player positions in the stats data. 
    """

    stat_files = glob.glob("../data/PlayerStats/*.csv")

    dfs = []
    pid_dfs = []
    for stat_file in stat_files:
        print(f"stat_file: {stat_file}")

        stat_df = pd.read_csv(stat_file)
        stat_df = stat_df.dropna(subset="player_id")

        pid_dfs.append(stat_df[['player_id', 'pos']])

        temp_df = stat_df.groupby(['player_id', 'pos']).size()
        temp_df.name = 'row_count'
        temp_df = temp_df.to_frame().reset_index()

        dfs.append(temp_df)

    pid_df = pd.concat(pid_dfs)
    df = pd.concat(dfs)

    df['pos'] = df['pos'].apply(lambda x: inv_alias_dict[x])

    print(df)
    
    position_df = df.groupby(['player_id'])['pos'].apply(list).to_frame()
    
    position_df['one_position_found'] = position_df['pos'].apply(lambda x: len(set(x))==1)
    
    print(position_df[position_df['one_position_found']==False])

    # print(multiple_pos_index)
    
    # print(f"total pid_df:\n{pid_df}")
    # print(f"total pid_df after dropping dups:\n{pid_df.dropna(subset=['pos']).drop_duplicates(subset=['player_id'])}")

    # print(f"total df:\n{df}")
    # print(f"total df after dropping dups:\n{df.dropna(subset=['pos']).drop_duplicates(subset=['player_id'])}")

    # print(f"unique player id:\n{pid_df.drop_duplicates(['player_id'])}")
    # print(f"unique player id, pos:\n{pid_df.drop_duplicates(['player_id', 'pos'])}")


        # temp_df = stat_df.groupby(['player_id', 'pos']).size()
        # temp_df.name = 'row_count'
        # temp_df = temp_df.to_frame().reset_index()

        # dfs.append(temp_df)

    # temp_df = pd.concat(dfs) 

    # print(f"temp_df:\n{temp_df}")

    # idx = temp_df.groupby(['player_id', 'pos'])['row_count'].idxmax()

    # print(idx[idx.duplicated()])

    # max_reps_df = temp_df.loc[idx][['player_id', 'pos']]

    # print(f"max_reps_df:\n{max_reps_df}")

    # player_pos_dict = max_reps_df.set_index(['player_id']).to_dict()['pos']
    
    # updated_stat_df = stat_df.copy()
    # updated_stat_df['pos'] = updated_stat_df.apply(lambda row: player_pos_dict[row['player_id']], axis=1)

    # print(f"Total player_id/pos pairs after: {updated_stat_df.drop_duplicates(['player_id', 'pos']).shape}")
    # print(f"Total player_id/pos pairs before: {stat_df.drop_duplicates(['player_id', 'pos']).shape}")

    # assert updated_stat_df.drop_duplicates(['player_id', 'pos']).shape[0] <= stat_df.drop_duplicates(['player_id', 'pos']).shape[0]

    # print(f"Before Shape: {stat_df.shape}")
    # print(f"After Shape: {updated_stat_df.shape}")

    # assert stat_df.shape == updated_stat_df.shape

    # updated_stat_df.to_csv(stat_file, index=False)

def update_players_table():
    stat_files = glob.glob("../data/PlayerStats/*.csv")

    player_pos_dfs = []

    for stat_file in stat_files:
        print(f"stat_file: {stat_file}")

        stat_df = pd.read_csv(stat_file)

        player_pos_dfs.append(stat_df[['player_id', 'pos']])
    
    pos_df = pd.concat(player_pos_dfs).drop_duplicates()

    print(pos_df)
    print(pos_df.drop_duplicates(subset=['player_id']))

    # # print(pos_df)
    # print(pos_df[pos_df.duplicated(subset=['player_id'])])

    # players_file_name = "../data/players.csv"

    # players_df = pd.read_csv(players_file_name)

    # print(players_df[players_df.duplicated(subset=['player_id'])])

    # merged_df = pos_df.merge(players_df, on=['player_id'], how='outer')

    # print(merged_df.tail(40))

    # print(pos_df[pos_df['player_id'].isin(players_df['player_id'])])
    # print(players_df[players_df['player_id'].isin(pos_df['player_id'])])

def transform_position_col(x):
    # extracting position label(s) from string
    positions = re.split(r'[^a-zA-Z]', x)
    positions = list({inv_alias_dict[i] for i in positions})

    return positions

def preprocess_stats_data():
    stat_files = glob.glob("../data/PlayerStats/*.csv")

    print(f"stat_files: {stat_files}")
    
    dfs = []
    for stat_file in stat_files:
        print(f"stat_file: {stat_file}")

        stat_df = pd.read_csv(stat_file)

        before_shape = stat_df.shape

        # drop rows where position is NaN
        stat_df = stat_df.dropna(subset=['pos'])

        # dropping rows where position value contains "Missed season" or "Did not play" 
        stat_df = stat_df[(~stat_df['pos'].astype(str).str.contains("Missed season")) & (~stat_df['pos'].astype(str).str.contains("Did not play"))]

        stat_df['pos'] = stat_df['pos'].apply(lambda x: transform_position_col(x))
        
        # dropping rows where player has more than one position listed
        stat_df = stat_df[~(stat_df['pos'].str.len()>1)]

        # converting "pos" column to str
        stat_df['pos'] = stat_df['pos'].apply(lambda x: x[0])
        
        # dropping rows where position = "UNKNOWN"
        stat_df = stat_df[~(stat_df['pos']=='UNKNOWN')]
        
        after_shape = stat_df.shape

        print(f"stat_df.shape (BEFORE|AFTER): {before_shape}|{after_shape}")
        stat_df.to_csv(stat_file, index=False)
        print("Finished Saving")

def update_player_positions():
    """
    Run this function to update player positions in the stats data. 
    """

    stat_files = glob.glob("../data/PlayerStats/*.csv")
    dfs = []
    for stat_file in stat_files:
        print(f"stat_file: {stat_file}")

        stat_df = pd.read_csv(stat_file)
        stat_df = stat_df.dropna(subset="player_id")

        dfs.append(stat_df[['player_id', 'pos']])
    
    temp_df = pd.concat(dfs).groupby(['player_id', 'pos']).size()
    temp_df.name = 'row_count'
    temp_df = temp_df.to_frame().reset_index()

    idx = temp_df.groupby(['player_id'])['row_count'].idxmax()
    max_reps_df = temp_df.loc[idx][['player_id', 'pos']]

    player_pos_dict = max_reps_df.set_index(['player_id']).to_dict()['pos']
    
    for stat_file in stat_files:
        print(f"stat_file: {stat_file}")

        stat_df = pd.read_csv(stat_file)
        stat_df = stat_df.dropna(subset="player_id")

        updated_stat_df = stat_df.copy()
        updated_stat_df['pos'] = updated_stat_df.apply(lambda row: player_pos_dict[row['player_id']], axis=1)

        print(updated_stat_df)

        print(f"Total player_id/pos pairs after: {updated_stat_df.drop_duplicates(['player_id', 'pos']).shape}")
        print(f"Total player_id/pos pairs before: {stat_df.drop_duplicates(['player_id', 'pos']).shape}")

        assert updated_stat_df.drop_duplicates(['player_id', 'pos']).shape[0] <= stat_df.drop_duplicates(['player_id', 'pos']).shape[0]

        print(f"Before Shape: {stat_df.shape}")
        print(f"After Shape: {updated_stat_df.shape}")

        assert stat_df.shape == updated_stat_df.shape

        # updated_stat_df.to_csv(stat_file, index=False)

    # updating players table
    players_df = pd.read_csv("../data/players.csv")
    players_df['position'] = players_df['player_id'].apply(lambda x: player_pos_dict[x] if x in player_pos_dict else None)

    print(players_df)

    players_df.to_csv("../data/players.csv", index=False)


if __name__ == '__main__':
    # check_all_pos_mapped()
    # preprocess_stats_data()
    # update_player_positions()
    # update_players_table()
    pass