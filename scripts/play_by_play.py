import pandas as pd
import json
import glob
import warnings
import re


warnings.simplefilter(action='ignore', category=FutureWarning)

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

def add_id_col_pbp():
    df = pd.read_csv("../temp_data/combined_pbp.csv")

    # print(f"Min Date: {df['event_date'].min()} - Max Date: {df['event_date'].max()}")

    # games_df = pd.read_csv("../data/raw_teams_ads.csv")
    # games_df['event_date'] = pd.to_datetime(games_df['event_date'])

    # print(f"{df.columns}\n{df[['team_1_name', 'team_2_name',]]}")
    # print(f"{games_df.columns}\n{games_df[['team_name', 'team_abbr']]}")

    # print(df.merge(games_df, left_on=['team_1_name', 'event_date'], right_on=['team_abbr', 'event_date']))

    # FINISH ME: Find a way to join pbp data and game data
    
if __name__ == "__main__":
    combine_jsons()
    # add_id_col_pbp()