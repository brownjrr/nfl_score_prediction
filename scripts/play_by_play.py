import pandas as pd
import json
import glob
import warnings
import re


warnings.simplefilter(action='ignore', category=FutureWarning)

position_type_dict = {
    'offense': {'QB', 'RB', 'WR', 'TE', 'OL',},
    'defense': {'LB', 'DB', 'DL',},
    'special_teams': {'PR', 'P', 'K',},
}

# inverting aliases dictionary
inv_position_type_dict = dict()

for i in position_type_dict:
    for j in position_type_dict[i]:
        inv_position_type_dict[j] = i

print(f"inv_position_type_dict:\n{inv_position_type_dict}")

with open("../data/abbrev_map.json", "r") as f:
    abbrev_map = json.load(f)

print(abbrev_map)

def combine_jsons():
    """
    combining all play by play data (json to pandas df)
    """
    column_len_set = set()
    column_set = set()
    dfs = []
    for df_json_file in [i.replace("\\", "/") for i in glob.glob("C:/Users/brown/OneDrive/pbp_data_files/dataframes/*.txt")]:
        with open(df_json_file, "r") as f:
            lines = f.readlines()

            for line in lines:
                json_obj = json.loads(line)

                link = json_obj['index']
                data = json_obj['data']

                df = pd.read_json(data)

                for j, col in enumerate(df.columns):
                    if col in abbrev_map:
                        team1_idx = j
                        team1_name = df.columns[j]

                        team2_idx = j+1
                        team2_name = df.columns[j+1]
                        break
                    
                df = df.rename(columns={team1_name: "team_1", team2_name: "team_2"})
                df['team_1_name'] = abbrev_map[team1_name]
                df['team_2_name'] = abbrev_map[team2_name]
                df['link'] = link
                df['boxscore_id'] = df['link'].str.split("/").str[-1].str.replace(".htm", "")
                
                column_len_set.add(len(df.columns))
                column_set |= set(df.columns)

                dfs.append(df)

    df = pd.concat(dfs).drop_duplicates()

    print(df)
    print(f"Num Unique Links: {len(df['link'].unique())}")

    pattern = r'<a href.*?/a>'
    df['player'] = df['Detail'].apply(lambda x: re.findall(pattern, x))

    # discard plays where no players were found AND
    # discard plays where only 1 player is part of the interaction
    # (these are often kicking/punting plays)
    df = df[df['player'].str.len() > 1]
    df['player'] = df['player'].apply(lambda x: [re.findall(r"\".*?\"", i)[0].replace('"', '') for i in x])
    df = df.explode('player')
    df['player'] = df['player'].apply(lambda x: "https://www.pro-football-reference.com"+x)

    # creating event_date columns
    df['event_date'] = pd.to_datetime(df['boxscore_id'].str[:-4])

    # converting team names to team abbreviations for consistency with other tables
    # df['team_1_name'] = df['team_1_name'].apply(lambda x: abbrev_map[x])
    # df['team_2_name'] = df['team_2_name'].apply(lambda x: abbrev_map[x])
    
    print(f"Min Date: {df['event_date'].min()} - Max Date: {df['event_date'].max()}")
    print(df)

    df.to_csv("../data/play_by_play.csv", index=False)

    # print(f"column_len_set: {column_len_set}")
    # print(f"column_set: {column_set}")
    # print(f"column_set length: {len(column_set)}")

def get_play_by_play_df():
    df = pd.read_csv("../data/play_by_play.csv")
    players_df = pd.read_csv("../data/players.csv")

    df = df.merge(players_df[['link', 'player_id', 'position']], left_on=['player'], right_on=['link'], how='inner').dropna(subset=['position'])
    df = df[df['position']!="UNKNOWN"]

    return df

def get_all_interaction_probabilities():
    offensive_positions = pd.Series([i for i in inv_position_type_dict if inv_position_type_dict[i]=="offense"])
    offensive_positions = offensive_positions.to_frame(name="pos_1")

    defensive_positions = pd.Series([i for i in inv_position_type_dict if inv_position_type_dict[i]=="defense"])
    defensive_positions = defensive_positions.to_frame(name="pos_2")

    df = pd.merge(offensive_positions, defensive_positions, how='cross')
    temp_df = df[['pos_2', 'pos_1']].rename(columns={'pos_1': 'pos_2', 'pos_2': 'pos_1'})
    df = pd.concat([df, temp_df])

    pbp_df = get_play_by_play_df()

    def get_interaction_prob(row):
        print(f"Position 1: {row['pos_1']} | Position 2: {row['pos_2']}")
        
        pos1 = row['pos_1']
        pos2 = row['pos_2']

        temp_df = pbp_df.groupby(['Quarter', 'Time', 'Down', 'ToGo', 'Location', 'link_x', "team_1", "team_2"])['position'].apply(list).reset_index()
        pos1_rows = temp_df[[(pos1 in x) for x in temp_df.position]].shape[0]
        num_p1_p2_rows = temp_df[[(pos1 in x) and (pos2 in x) for x in temp_df.position]].shape[0]
        
        return num_p1_p2_rows / pos1_rows

    df['interaction_prob'] = df.apply(lambda row: get_interaction_prob(row), axis=1)
    
    print(df)

    df.to_csv("../data/interaction_prob.csv", index=False)

if __name__ == "__main__":
    # combine_jsons()
    # add_id_col_pbp()
    # get_play_by_play_df()
    # get_interaction_prob(get_play_by_play_df(), "RB", "DB")
    get_all_interaction_probabilities()