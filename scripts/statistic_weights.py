import pandas as pd
from functools import reduce
import re


# Get game level statistic files
stats_files = {
    'game_starters_all.csv': "gs",
    'injuries_all.csv': "inj",
    'player_defense_all.csv': "def",
    'player_kicking_all.csv': "kick",
    'player_offense_all.csv': "off",
    'player_returns_all.csv': "ret",
    'snap_counts_all.csv': "snap",
}

def transform_date_injuries(row):
    season = int(row['inj_season'])
    month_day = row['date'].replace("/", "-")
    
    if month_day.split("-")[0] in ["01", "02", "03", "04", "05", "06"]:
        text = f"{season+1}-{month_day}"
    else:
        text = f"{season}-{month_day}"

    return text

    # lambda row: str(row['inj_season']) + '-' + str(row['date']).replace("/", "-")

def make_season_col(x):
    year = x.date().year
    month = x.date().month

    if month in [1,2,3,4,5,6]:
        season = year - 1
    else:
        season = year

    return season

def get_stats_table():
    folder = "../data/"
    dfs = []

    for i in stats_files:
        print(f"File: {i}")

        file = folder + i
        
        df = pd.read_csv(file)

        # convert all columns to lowercase and add prefix
        df.columns = [f"{stats_files[i]}_{j.lower()}" if j.lower() not in ['date', 'player_id', 'team'] else j.lower() for j in df.columns]
        
        if "team" in df.columns:
            df = df.drop(columns=['team'])

        if "injuries_all" in i:
            df['date'] = df.apply(transform_date_injuries, axis=1)

        # convert date to datetime
        df['date'] = pd.to_datetime(df['date'])

        # df = df[df['player_id']=="MannPe00"]

        # print columns
        print(df.columns)

        before_df = df
        after_df = df.dropna(subset=['date', 'player_id'])

        # show shape of df
        print(before_df.shape)

        # show shape of df after dropping null rows
        print(after_df.shape)
        print()

        assert before_df.shape == after_df.shape
        assert "date" in df.columns
        assert "player_id" in df.columns

        # dfs.append(df.set_index(["date", "player_id", "team"]))
        dfs.append(df)
    
    dfs = sorted(dfs, key=lambda x: x.shape[0], reverse=True)

    df_merged = dfs[0]

    for i in dfs[1:]:
        df_merged = pd.merge(df_merged, i, on=['date', 'player_id'], how='outer')
    
    df_merged = df_merged.drop_duplicates(subset=['date', 'player_id'])

    df_merged['season'] = df_merged['date'].apply(make_season_col)

    # pruning columns that we don't want in our data
    delete_cols = [
        'snap_player', 
        'snap_position',
        'gs_starter', 
        'gs_position',
        'inj_names',
        'inj_opp_team',
        'def_player',
        'off_player_name',
        'kick_player_name',
        'ret_player_name',
        'inj_injury_type',
    ]
    
    df_merged = df_merged.drop(columns=delete_cols)

    # converting _pct columns from str to float
    pct_cols = [i for i in df_merged.columns if "_pct" in i]

    for col in pct_cols:
        df_merged[col] = df_merged[col].apply(lambda x: int(x.replace("%", ""))/100 if isinstance(x, str) else x)

    def transform_active_inactive(x):
        if isinstance(x, str):
            if x.lower() == "active":
                return 1
            else:
                return 0
        else:
            return x

    df_merged['inj_active'] = df_merged['inj_active_inactive'].apply(transform_active_inactive)
    df_merged = df_merged.drop(columns=['inj_active_inactive'])

    injured_status_dict = {
        "available": {"Available"},
        "out": {'Out'},
        "probable": {'Probable'},
        "doubtful": {'Doubtful'},
        "injured_reserve": {'Injured Reserve', 'Injured Rese'},
        "questionable": {'Questionable'},
        "unable_to_perform": {'Physically Unable To Perform'},
        "suspended": {'Suspended'},
        "day_to_day": {'Day-To-Day'},
        "reserve": {'Reserve/Future', 'Reserve/Injured From Waived/Injured'},
        "unknown": {'Unknown'},
    }

    def transform_status_col(x, terms):
        if isinstance(x, str):
            if x in terms:
                return 1
            else:
                return 0
        else:
            return x

    for i in injured_status_dict:
        new_col = f"inj_game_status_{i}"
        df_merged[new_col] = df_merged["inj_game_status"].apply(lambda x: transform_status_col(x, injured_status_dict[i]))
    
    df_merged = df_merged.drop(columns=['inj_game_status'])

    print(df_merged)
    print(df_merged.shape)
    print(df_merged.drop_duplicates(subset=['player_id']).shape)
    print(df_merged.info(verbose=True))

    # print(f"{df_merged['inj_active_inactive'].unique()}")
    # print(f"{df_merged['inj_game_status'].unique()}")


def get_yearly_AV():
    pass


if __name__ == "__main__":
    get_stats_table()