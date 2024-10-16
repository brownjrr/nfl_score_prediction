import pandas as pd
from functools import reduce
import re
import os
import glob
from bs4 import BeautifulSoup
import requests
import json
import datetime
from pytz import timezone
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import os


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

print(f"script_dir: {script_dir}")


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

def get_current_time():
    tz = timezone('EST')
    current_time = datetime.datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')

    return current_time

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

def get_stats_table(save=False, verbose=False):
    folder = script_dir+"../data/"
    dfs = []

    for i in stats_files:
        if verbose: print(f"File: {i}")
        
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
        if verbose: print(df.columns)

        before_df = df
        after_df = df.dropna(subset=['date', 'player_id'])

        # show shape of df
        if verbose: print(before_df.shape)

        # show shape of df after dropping null rows
        if verbose: print(after_df.shape)
        if verbose: print()

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

    if verbose:
        print(df_merged)
        print(df_merged.shape)
        print(df_merged.drop_duplicates(subset=['player_id']).shape)
        print(df_merged.info(verbose=True))
        print(df_merged['season'])
        # print(f"{df_merged['inj_active_inactive'].unique()}")
        # print(f"{df_merged['inj_game_status'].unique()}")

    if save:
        df_merged.to_csv(script_dir+"../data/raw_combined_player_stats.csv", index=False)

    return df_merged

def look_for_av_values(pid, tables, data):
    av_found = False
    for table in tables:
        temp_df = pd.read_html(table.prettify())[0]
        
        if temp_df.columns.nlevels > 1:
            temp_df.columns = temp_df.columns.droplevel()

        temp_df.columns = [i.lower() for i in temp_df.columns]

        if "av" in temp_df.columns and ("year" in temp_df.columns or "season" in temp_df.columns):
            # defining time (year) column
            if "year" in temp_df.columns:
                time_col = "year"
            elif "season" in temp_df.columns:
                time_col = "season"
            else:
                time_col = None
            
            # convert time column to numeric type. Drop any rows with null values
            temp_df[time_col] = pd.to_numeric(temp_df[time_col], errors="coerce")
            temp_df = temp_df.dropna(subset=[time_col])
            temp_df = temp_df[temp_df[time_col].astype(int)!=2024]

            for ind in temp_df.index:
                av_found = True
                year = temp_df[time_col][ind]
                av = temp_df["av"][ind]
                table_name = table.get("id")

                data.append((pid, table_name, year, av))
    
    return data, av_found

def get_yearly_AV(get_missing=False):
    dir_name = "C:/Users/Robert Brown/OneDrive/player_data_files/"
    subfolders= [f.path for f in os.scandir(dir_name) if f.is_dir()]

    print(f"subfolders: {len(subfolders)}")

    players_with_no_data = []
    players_with_data_no_av = []
    position_set = set()
    data = []

    if get_missing:
        missing_set = set(get_missing_player_data_from_raw_data())

    for idx, i in enumerate(subfolders[:]):
        if idx % 100 == 0:
            print(f"[{get_current_time()}] {idx}/{len(subfolders)} Folders Seen")
        
        file = glob.glob(i+"/*.txt")[0].replace("\\", "/")

        pid = i.split("/")[-1]


        if get_missing and pid not in missing_set:
            continue

        # print(f"[{get_current_time()}] {pid}")

        # print(f"File: {file}")
        
        with open(file, "r", encoding="utf8") as f:
            # get html from webpage
            soup = BeautifulSoup(f.read(), "lxml")
            
            # find tables
            tables = soup.find_all("table")
            table_types = [i.get("id") for i in tables]

            # print(f"table_types: {table_types}")

            # if no tables were found, go to the website and look for them
            if len(tables) == 0:
                link = f"https://www.pro-football-reference.com/players/{pid[0].upper()}/{pid}.htm"

                page = requests.get(link)

                # define BeautifulSoup object
                soup = BeautifulSoup(page.content, "lxml")

                # get html as str
                html = str(soup.prettify())

                tables = soup.find_all("table")
                table_types = [i.get("id") for i in tables]

                # if no tables were found after going to the website again,
                # add this pid to the list and continue
                if len(tables) == 0:
                    players_with_no_data.append(pid)
                    continue
            
            data, av_found = look_for_av_values(pid, tables, data)
            
            # if AV was not found in the tables, go to the 
            # website and look for the tables again
            if not av_found:
                link = f"https://www.pro-football-reference.com/players/{pid[0].upper()}/{pid}.htm"

                page = requests.get(link)

                # define BeautifulSoup object
                soup = BeautifulSoup(page.content, "lxml")

                # get html as str
                html = str(soup.prettify())

                tables = soup.find_all("table")
                table_types = [i.get("id") for i in tables]

                print(f"[{get_current_time()}] redone table_types: {table_types}")

                data, av_found = look_for_av_values(pid, tables, data)
            
            # if AV was still not found after looking again,
            # add the pid to the list and continue
            if not av_found:
                players_with_data_no_av.append(pid)
                continue

            
    df = pd.DataFrame(data, columns=['player_id', 'table', 'season', 'AV'])
    df["season"] = df["season"].astype(int)

    df.reset_index().to_csv(script_dir+"../data/raw_player_yearly_av.csv", index=False)

def examine_raw_yearly_df():
    df = pd.read_csv(script_dir+"../data/raw_player_yearly_av.csv")

    print(df)

    print(len(df['player_id'].unique()))

def get_missing_player_data_from_raw_data():
    df = pd.read_csv(script_dir+"../data/raw_player_yearly_av.csv")

    player_ids_with_data = set(df['player_id'].unique())

    dir_name = "C:/Users/Robert Brown/OneDrive/player_data_files/"
    all_player_ids= [f.path.split('/')[-1] for f in os.scandir(dir_name) if f.is_dir()]

    no_data = []
    for i in all_player_ids:
        if i not in player_ids_with_data:
            no_data.append(i)
    
    return no_data

    print(f'len(no_data): {len(no_data)}')
    print(f'no_data: {no_data}')

def process_raw_yearly_av_data():
    df = pd.read_csv(script_dir+"../data/raw_player_yearly_av.csv")

    df['season'] = df['season'].astype(int)
    df['AV'] = pd.to_numeric(df['AV'], errors="coerce")

    df = df.dropna(subset=["AV"])

    df = df.groupby(["player_id", "season"]).agg({'AV': 'mean', 'table': list}).reset_index()

    df.to_csv(script_dir+"../data/player_yearly_av.csv", index=False)

def process_raw_combined_player_stats():
    df = pd.read_csv(script_dir+"../data/raw_combined_player_stats.csv")

    df = df.drop(columns={'date'}).groupby(["player_id", "season"]).sum().reset_index()

    df.to_csv(script_dir+"../data/combined_player_stats.csv", index=False)

def get_train_test_data_v1(start_year, end_year, use_standard_scaler=False):
    av_df = pd.read_csv(script_dir+"../data/player_yearly_av.csv")

    # filter by start and end year
    av_df = av_df[(av_df['season']>=start_year) & (av_df['season']<=end_year)]
    av_df = av_df.drop(columns=['table'])

    stats_df = pd.read_csv(script_dir+"../data/combined_player_stats.csv")

    player_pos_df = pd.read_csv(script_dir+"../data/player_positions.csv")
    player_pos_df['position'] = player_pos_df['position'].str.split("-")
    player_pos_df = player_pos_df.explode('position')

    pos_df = pd.read_csv(script_dir+"../data/positions.csv")

    all_positions = [i for i in pos_df['adj_pos'].unique() if i not in ["UNKNOWN", "PR"]]
    
    # remove players that have more than one position listed
    player_pos_df = player_pos_df.merge(pos_df, on=["position"], how="inner")
    player_pos_df = player_pos_df.drop_duplicates(subset=['player_id', 'adj_pos'])
    player_pos_df = player_pos_df.drop_duplicates(subset=['player_id',])

    # print(player_pos_df)

    final_df = av_df.merge(stats_df, on=["player_id", "season"], how='inner')
    final_df = player_pos_df.merge(final_df, on=['player_id'], how='inner')
    final_df = final_df.drop(columns=['position', 'position_type']).rename(columns={"adj_pos": "position"})

    results = dict()
    for i in all_positions:
        temp_df = final_df[final_df['position']==i]

        # drop id columns
        temp_df = temp_df.drop(columns=['player_id', 'position', 'season'])

        X = temp_df[[i for i in temp_df.columns if i != "AV"]]
        y = temp_df['AV']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if use_standard_scaler:
            scaler = StandardScaler().fit(X_train)
            X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        results[i] = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
        }
        
    return results

def get_train_test_data_v2(start_year, end_year, use_standard_scaler=False):
    av_df = pd.read_csv(script_dir+"../data/player_yearly_av.csv")

    # filter by start and end year
    av_df = av_df[(av_df['season']>=start_year) & (av_df['season']<=end_year)]
    av_df = av_df.drop(columns=['table'])

    stats_df = pd.read_csv(script_dir+"../data/combined_player_stats.csv")

    player_pos_df = pd.read_csv(script_dir+"../data/player_positions.csv")
    player_pos_df['position'] = player_pos_df['position'].str.split("-")
    player_pos_df = player_pos_df.explode('position')

    pos_df = pd.read_csv(script_dir+"../data/positions.csv")

    all_positions = [i for i in pos_df['adj_pos'].unique() if i not in ["UNKNOWN", "PR"]]
    
    # remove players that have more than one position listed
    player_pos_df = player_pos_df.merge(pos_df, on=["position"], how="inner")
    player_pos_df = player_pos_df.drop_duplicates(subset=['player_id', 'adj_pos'])
    player_pos_df = player_pos_df.drop_duplicates(subset=['player_id',])

    # print(player_pos_df)

    final_df = av_df.merge(stats_df, on=["player_id", "season"], how='inner')
    final_df = player_pos_df.merge(final_df, on=['player_id'], how='inner')
    final_df = final_df.drop(columns=['position', 'position_type']).rename(columns={"adj_pos": "position"})

    # One-hot Encode position column
    positions = list(final_df['position'].unique())

    for pos in positions:
        final_df[f'is_{pos}'] = final_df['position'].apply(lambda x: 1 if x==pos else 0)

    # drop id columns
    final_df = final_df.drop(columns=['player_id', 'position', 'season'])

    X = final_df[[i for i in final_df.columns if i != "AV"]]
    y = final_df['AV']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if use_standard_scaler:
        scaler = StandardScaler().fit(X_train)
        X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    results = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
    }
        
    return results

def ridge_regression_v1(verbose=False):
    results = dict()

    for use_standard_scaler in [False,]:
        if verbose: print(f"use_standard_scaler: {use_standard_scaler}")

        train_test_data = get_train_test_data_v1(start_year=2013, end_year=2014, use_standard_scaler=use_standard_scaler)

        for i in train_test_data:
            if verbose: print(f"POSITION: {i}")
            X_train = train_test_data[i]['X_train']
            X_test = train_test_data[i]['X_test']
            y_train = train_test_data[i]['y_train']
            y_test = train_test_data[i]['y_test']
            lowest_mae = 1000

            for fit_intercept in [True, False]:
                if verbose: print(f"fit_intercept: {fit_intercept}")
                reg = Ridge(fit_intercept=fit_intercept).fit(X_train, y_train)
                train_score = reg.score(X_train, y_train)
                test_score = reg.score(X_test, y_test)

                if verbose: print(f"train_score: {train_score}")
                if verbose: print(f"test_score: {test_score}")

                y_pred = reg.predict(X_test)

                mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)

                if verbose: print(f"MAE: {mae}")

                if mae < lowest_mae:
                    coefficients = reg.coef_
                    features = reg.feature_names_in_
                    feature_importances = dict(zip(features, coefficients))

                    results[i] = feature_importances

            if verbose: print("=========================================")
        if verbose: print("________________________________________________________________")

    X_train.to_csv('data/intermediate/model_ridge__position__x_train.csv')
    X_test.to_csv('data/intermediate/model_ridge__position__x_test.csv')
    return results

def ridge_regression_v2(verbose=False):
    results = None

    for use_standard_scaler in [False, True]:
        if verbose: print(f"use_standard_scaler: {use_standard_scaler}")

        train_test_data = get_train_test_data_v2(start_year=2013, end_year=2014, use_standard_scaler=use_standard_scaler)


        X_train = train_test_data['X_train']
        X_test = train_test_data['X_test']
        y_train = train_test_data['y_train']
        y_test = train_test_data['y_test']
        lowest_mae = 1000

        for fit_intercept in [True, False]:
            if verbose: print(f"fit_intercept: {fit_intercept}")
            reg = Ridge(fit_intercept=fit_intercept).fit(X_train, y_train)
            train_score = reg.score(X_train, y_train)
            test_score = reg.score(X_test, y_test)

            if verbose: print(f"train_score: {train_score}")
            if verbose: print(f"test_score: {test_score}")

            y_pred = reg.predict(X_test)

            mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)

            if verbose: print(f"MAE: {mae}")

            if mae < lowest_mae:
                coefficients = reg.coef_
                features = reg.feature_names_in_
                feature_importances = dict(zip(features, coefficients))

                results = feature_importances

        if verbose: print("=========================================")

    X_train.to_csv('data/intermediate/model_ridge__weights_overall__x_train.csv')
    X_test.to_csv('data/intermediate/model_ridge__weights_overall__x_train.csv')

    return results

def create_player_stat_weights_df(partion_by_position=False):
    inv_stats_files = {stats_files[i]:i for i in stats_files}

    print(f"inv_stats_files: {inv_stats_files}")
    
    if partion_by_position:
        weights_dict = ridge_regression_v1(verbose=False)

        data = []

        for pos in weights_dict:
            for stat in weights_dict[pos]:
                stat_source = stat.split("_")[0]
                stat_weight = weights_dict[pos][stat]

                data.append((pos, "_".join(stat.split("_")[1:]), inv_stats_files[stat_source], stat_weight))
        
        df = pd.DataFrame(data, columns=['position', 'statistic', 'stat_source', 'weight'])
        
        print(df)

        df.to_csv(script_dir+"../data/stat_weights_by_position.csv", index=False)
    else:
        weights_dict = ridge_regression_v2(verbose=False)
        
        data = []
        for stat in weights_dict:
            stat_source = stat.split("_")[0]

            if stat_source in inv_stats_files:
                stat_weight = weights_dict[stat]
                data.append(("_".join(stat.split("_")[1:]), inv_stats_files[stat_source], stat_weight))
        
        df = pd.DataFrame(data, columns=['statistic', 'stat_source', 'weight'])
        
        print(df)

        df.to_csv(script_dir+"../data/stat_weights_overall.csv", index=False)


if __name__ == "__main__":
    # get_yearly_AV(get_missing=False)
    
    get_stats_table(save=True, verbose=True)
    process_raw_yearly_av_data()

    for i in [True, False]:
        create_player_stat_weights_df(partion_by_position=i)
