import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil import parser
import json
import time
import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from pytz import timezone
import re
import timeout_decorator

def get_current_time():
    tz = timezone('EST')
    current_time = datetime.datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')

    return current_time

def df_to_json(df, filename):
    json_df = json.dumps(df.to_json())+"\n"

    # print(type(json_df))
    # print(json_df)

    with open(filename, "a+") as f:
        f.write(json_df)

def dict_to_json(my_dict, filename):
    obj = json.dumps(my_dict)+"\n"

    with open(filename, "a+") as f:
        f.write(obj)

def get_box_score_objs():
    # setting base url
    base_website = "https://www.pro-football-reference.com/boxscores/game-scores.htm"

    # get response from websit
    page = requests.get(base_website)

    # define BeautifulSoup object
    soup = BeautifulSoup(page.content, "html.parser")

    # get div
    results = soup.find(id="all_games")

    # creating dataframe from div
    df = pd.read_html(results.prettify(), extract_links='body')[0]

    # cleaning up the data (visual aid)
    df = df.applymap(lambda x: x[0] if x[1] is None else x)

    # print(df)

    # get link for all games of each score
    scores = {}

    for ind in df.index:
        tup = df["Unnamed: 7"][ind]
        score = df["Score"][ind]

        scores[score] = tup[1]

    # print(scores)

    base_url = "https://www.pro-football-reference.com"

    for score in scores:
        relative_url = scores[score]

        full_url = base_url + relative_url

        print(full_url)

        page = requests.get(full_url)

        # define BeautifulSoup object
        soup = BeautifulSoup(page.content, "html.parser")

        # get div
        results = soup.find(id="all_games")

        # creating dataframe from div
        df = pd.read_html(results.prettify(), extract_links='body')[0]

        # cleaning up the data (visual aid)
        df = df.applymap(lambda x: x[0] if x[1] is None else x)
        
        # print(df.columns)

        df['Date'] = pd.to_datetime(df['Date'])

        df = df[(df['Date'] >= "2013-06-01") & (df['Date'] <= "2024-06-01")]

        # print(df)
        
        df_to_json(df, "../data/game_level_data.txt")
        
        time.sleep(1)

    print(results.prettify())

    print(results.prettify())

    print(df)

def get_play_by_play_dfs():
    options = Options()
    options.add_argument('--headless=new')
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(20)

    urls = []
    col_set = set()
    base_url = "https://www.pro-football-reference.com"
    
    seen_links = get_seen_pbp_dfs()
    all_box_score_links = get_seen_pbp_links()

    print(f"[{get_current_time()}] Total Num Seen Links: {len(seen_links)}")
    print(f"[{get_current_time()}] Total Box Score Links: {len(all_box_score_links)}")

    with open("../data/game_level_data.txt", "r") as f:
        lines = f.readlines()

        for j, line in enumerate(lines):
            data = pd.read_json(json.loads(line))

            data = data.rename(columns={'Unnamed: 8': 'box_score'})
            data['box_score'] = data['box_score'].apply(lambda x: base_url+x[1])

            col_set |= set(data.columns)

            # print(data)

            for i in data['box_score']:
                # if i in seen_links:
                #     continue

                # print(f"[{get_current_time()}] link: {i}")

                # get response from websit
                # page = requests.get(i)

                print(f"[{get_current_time()}] Going to URL")

                try:
                    driver.get(i)
                    print(f"[{get_current_time()}] URL Reached")
                except Exception as e:
                    print(f"Error Encountered: {e}")
                    continue

                # print(f"[{get_current_time()}] Looking for DIV")
                pbp_div = driver.find_element(By.ID, "all_pbp")

                # print(f"[{get_current_time()}] Getting html from DIV")
                table_html = pbp_div.get_attribute('outerHTML')

                # print(f"[{get_current_time()}] Reading table from DIV")
                df = pd.read_html(table_html, flavor='bs4', extract_links='all')[0]

                pd.set_option('display.max_columns', None)
                pd.set_option('display.max_rows', None)
                pd.set_option('display.max_colwidth', None)

                print(df)
                
                # print(f"[{get_current_time()}] Running Applymap")
                # cleaning up the data (visual aid)
                df = df.applymap(lambda x: x[0] if x[1] is None else x)
                
                # print(df)

                # print(f"[{get_current_time()}] Converting table to json")
                save_dict = {
                    'index': i,
                    'data': df.to_json()
                }

                dict_to_json(save_dict, "../data/play_by_play_dfs.txt")
        
                time.sleep(1)

                print(f"[{get_current_time()}] Finishing Up {i}")

def get_seen_pbp_dfs():
    seen_set = set()
    with open("../data/play_by_play_dfs.txt", "r") as f:
        lines = f.readlines()

        for i, line in enumerate(lines):
            data = json.loads(line)
            seen_set.add(data['index'])

    print(len(seen_set))

    return seen_set

def get_seen_pbp_links():
    base_url = "https://www.pro-football-reference.com"

    all_box_score_links = set()

    with open("../data/game_level_data.txt", "r") as f:
        lines = f.readlines()

        for j, line in enumerate(lines):
            data = pd.read_json(json.loads(line))
            data = data.rename(columns={'Unnamed: 8': 'box_score'})
            data['box_score'] = data['box_score'].apply(lambda x: base_url+x[1])

            all_box_score_links |= set(list(data['box_score']))
    
    print(f"Num Found Links: {len(all_box_score_links)}")

    return all_box_score_links

def list_col_select(x, idx):
    if isinstance(x, list) or isinstance(x, tuple):
        return x[idx]
    else:
        return x

def concat_pbp_tables():
    data_dict = dict()

    with open("../data/play_by_play_dfs.txt", 'r') as f:
        lines = f.readlines()

        print(len(lines))

        index_set = set()

        for line in lines:
            data = json.loads(line)
            link = data['index']
            temp_df = pd.read_json(data['data'])

            index_set.add(link)

            if link not in data_dict:
                data_dict[link] = temp_df
            
            break

        print(len(index_set))
    
    with open("../data/abbrev_map.json", 'r') as f:
        abbrev_dict = json.load(f)

    for link in data_dict:
        temp_df = data_dict[link]
        team_cols = []
        new_cols = []
        for i, col in enumerate(temp_df.columns):
            if col in abbrev_dict:
                new_cols.append(abbrev_dict[col])
                team_cols.append((i, abbrev_dict[col]))
            else:
                new_cols.append(col)
        
        temp_df.columns = new_cols

        away_team, home_team = tuple([i[1] for i in sorted(team_cols, key=lambda x: x[0])])

        temp_df['home_team'] = home_team
        temp_df['away_team'] = away_team

        temp_df = temp_df.rename(columns={away_team: 'away_team_score', home_team: 'home_team_score'})
        
        for i in temp_df.columns:
            contains_list = False

            for val in temp_df[i]:
                if isinstance(val, list) or isinstance(val, tuple):
                    contains_list=True

            if contains_list and i != "Detail":
                temp_df[i] = temp_df[i].apply(lambda x: list_col_select(x, 0))

        for i in temp_df['Detail']:
            print(i)


        # print(temp_df.duplicated(keep=False))


    # print(data_dict)
    # print(data_dict[list(data_dict.keys())[0]].columns)

    # 

def get_webdriver():
    options = Options()
    options.add_argument('--headless=new')
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(20)

    return driver

timeout_decorator.timeout(60)
def get_pbp_tables_with_links(driver, site, html_save_file, data_save_file):
    try:
        driver.get(site)
        print(f"[{get_current_time()}] URL Reached")
    except Exception as e:
        print(f"Error Encountered: {e}")
        return

    pbp_div = driver.find_element(By.ID, "all_pbp")

    pbp_table = pbp_div.find_element(By.TAG_NAME, 'table')

    table_html = pbp_table.get_attribute('outerHTML')

    # write html for this page to file
    with open(html_save_file, "w+") as f:
        f.write(table_html)
    
    # print(pbp_table.get_attribute('outerHTML'))
    
    headers = []
    data = []

    # grabbing rows
    rows = pbp_table.find_elements(By.TAG_NAME, "tr")

    # getting column names from first row
    columns = [i.text for i in rows[0].find_elements(By.TAG_NAME, 'th')]
    
    print(f"columns: {columns}")

    # finding index of the Detail header
    detail_idx = None
    for i, col in enumerate(columns):
        if col == 'Detail':
            detail_idx = i

    # getting table data
    for row in rows[1:]:
        th_elmts = row.find_elements(By.TAG_NAME, 'th')
        td_elmts = row.find_elements(By.TAG_NAME, 'td')

        headers.append([i.text for i in th_elmts])

        if (len(td_elmts) == 1) or (len(td_elmts) == 0): # skip 
            continue

        row_data = []

        for idx, i in enumerate(th_elmts + td_elmts):

            # preserving links in Detail column
            if idx == detail_idx:
                p = re.compile(r'<a name.*?/a>')
                inner_html = i.get_attribute("innerHTML")
                inner_html = p.sub('', inner_html)

                row_data.append(inner_html)
            else: # grabbing only text from cell
                row_data.append(i.text)
        
        data.append(row_data)

    df = pd.DataFrame(data, columns=columns)

    save_dict = {
        'index': site,
        'data': df.to_json()
    }

    dict_to_json(save_dict, data_save_file)

    time.sleep(1)

    print(f"[{get_current_time()}] Finishing Up {i}")

if __name__ == '__main__':
    # get_box_score_objs()
    # get_play_by_play_dfs()
    # concat_pbp_tables()
    # get_pbp_tables_with_links(
    #     driver=get_webdriver(), 
    #     site="https://www.pro-football-reference.com/boxscores/201309220oti.htm",
    #     html_save_file="../temp_data/201309220oti.txt",
    # )

    pass