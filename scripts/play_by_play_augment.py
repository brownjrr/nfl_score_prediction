import pandas as pd
import re
import os


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

print(f"script_dir: {script_dir}")

def get_position_type_dicts():
    position_type_dict = {
        'offense': {'QB', 'RB', 'WR', 'TE', 'OL', 'P', 'K',},
        'defense': {'LB', 'DB', 'DL', 'PR'},
    }

    # inverting aliases dictionary
    inv_position_type_dict = dict()

    for i in position_type_dict:
        for j in position_type_dict[i]:
            inv_position_type_dict[j] = i

    print(f"inv_position_type_dict:\n{inv_position_type_dict}")

    return position_type_dict, inv_position_type_dict

def prepocess_text(x, player_pos_dict, inv_pos_type_dict, adj_pos_dict):
    pattern = r'<a href.*?/a>'

    players_refs = [i for i in re.findall(pattern, x) if "/players/" in i]

    for ref in players_refs:
        # print(ref)
        pid = re.search(r'".*?"', ref).group().strip("\"").split('/')[-1].replace(".htm", "")

        if pid in player_pos_dict:
            pos = player_pos_dict[pid]
        else:
            return x

        pos = pos.replace("/", "-")
        pos = pos.replace(",", "-")

        if "-" in pos:
            positions = list(set([adj_pos_dict[i] for i in pos.split("-")]))
            pos = positions[0]

        pos = adj_pos_dict[pos]
        pos_type = inv_pos_type_dict[pos]

        if pid in player_pos_dict:
            x = x.replace(ref, f"<{pid},{pos},{pos_type}>")
        else:
            x = x.replace(ref, f"<PLAYER_NOT_FOUND:{pid}>")

    return x

def get_player_pos_dict():
    df = pd.read_csv(script_dir+"../data/player_positions.csv")
    df = df.drop_duplicates(subset=['player_id'])

    return df.set_index("player_id").to_dict()['position']

def preprocess_play_text(df):
    inv_pos_type_dict = get_position_type_dicts()[1]
    player_pos_dict = get_player_pos_dict()
    adj_pos_dict = pd.read_csv(script_dir+"../data/positions.csv")[['position', 'adj_pos']].set_index(['position']).to_dict()['adj_pos']

    df['tokenized_play_text'] = df['Detail'].apply(prepocess_text, args=(player_pos_dict,inv_pos_type_dict,adj_pos_dict,),)

    return df

def add_coach_data_play_by_play():
    df = pd.read_csv(script_dir+"../data/play_by_play_extended.csv")
    coach_df = pd.read_csv(script_dir+"../data/game_level_coach_data.csv")
    coach_dict = coach_df[['boxscore_id', 'team_id', 'coach_id']].set_index(['boxscore_id', 'team_id']).to_dict()['coach_id']
    
    def get_coach(row, team1=True):
        boxscore = row['boxscore_id']

        if team1:
            team_1_id = row['team_1_name']
            coach_1_id = coach_dict[(boxscore, team_1_id)]
            return coach_1_id
        else:
            team_2_id = row['team_2_name']
            coach_2_id = coach_dict[(boxscore, team_2_id)]
            return coach_2_id

    df['team_1_coach'] = df.apply(lambda row: get_coach(row), axis=1)
    df['team_2_coach'] = df.apply(lambda row: get_coach(row, team1=False), axis=1)

    return df

def convert_date_to_season(x):
    if x.month <= 6:
        return x.year - 1
    else:
        return x.year

def get_roster_data():
    df = pd.read_csv(script_dir+"../data/rosters.csv")
    df['team_id'] = df['team_link'].str.split("/", expand=False).str[-2]

    return df

def add_offensive_team_play_by_play():
    df = add_coach_data_play_by_play()

    print(f"Initial Frame Size: {df.shape}")

    df = preprocess_play_text(df)

    # creating season column
    df['season'] = pd.to_datetime(df['event_date']).apply(convert_date_to_season)

    roster_df = get_roster_data()

    def find_offense_defensive_team(row):
        pattern = r'<.*?>'

        off_players_refs = [i for i in re.findall(pattern, row['tokenized_play_text']) if "offense" in i]

        # print(f"off_players_refs: {off_players_refs}")

        if len(off_players_refs) == 0:
            return None, None
        
        # find team of one offensive player
        player_ref = off_players_refs[0][1:-1]
        
        team_1_id = row['team_1_name']
        team_2_id = row['team_2_name']

        pid = player_ref.split(',')[0]
        season = row['season']
        temp_df = roster_df[(roster_df['player_id']==pid) & (roster_df['roster_year']==season) & (roster_df['team_id'].isin([team_1_id, team_2_id]))]
        
        if temp_df.empty:
            return None, None
        
        team = temp_df['team_id'].values[0]

        assert team == team_1_id or team == team_2_id, f"team={team} | team_1_id={team_1_id} | team_2_id={team_2_id}\n{temp_df}"

        if team == team_1_id:
            offense_team, defense_team = team_1_id, team_2_id
        else:
            offense_team, defense_team = team_2_id, team_1_id

        return offense_team, defense_team


    df['offense_defense'] = df.apply(find_offense_defensive_team, axis=1)
    df[['offensive_team', 'defensive_team']] = pd.DataFrame(df['offense_defense'].tolist(), index=df.index)
    df = df.drop(columns=['offense_defense', 'tokenized_play_text'])

    print(df)

    df.to_csv(script_dir+"../data/play_by_play_extended_v2.csv", index=False)

if __name__ == "__main__":
    add_offensive_team_play_by_play()
