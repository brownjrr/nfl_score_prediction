import requests
import pandas as pd
from bs4 import BeautifulSoup
import glob
import time
import os
import multiprocessing as mp
from dateutil.parser import parse
import os


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")+"/"

print(f"script_dir: {script_dir}")

def get_boxscores_html():
    base_dir = "C:/Users/Robert Brown/OneDrive/"
    save_folder = f"{base_dir}boxscore_data_files/"
    files = [i.replace("\\", "/") for i in glob.glob(f"{base_dir}pbp_data_files/*.txt")]
    boxscores = [i.split("/")[-1].replace(".txt", "") for i in files]
    urls = [
        f"https://www.pro-football-reference.com/boxscores/{i}.htm"
        for i in boxscores
    ]

    seen_boxscores = {str(f.path).split('/')[-1] for f in os.scandir(save_folder) if f.is_dir()}

    print(f"seen_boxscores: {len(seen_boxscores)}")
    
    step = 0
    for boxscore, i in list(zip(boxscores, urls)):
        if step % 100 == 0:
           print(f"Step {step} of {len(urls)}")

        step += 1

        if boxscore in seen_boxscores: continue

        # print(f"URL: {i}")

        page = requests.get(i)

        # define BeautifulSoup object
        soup = BeautifulSoup(page.content, "lxml")

        # get html as str
        html = str(soup.prettify())

        # print(html)

        boxscore_folder = f"{save_folder}{boxscore}/"
        os.makedirs(boxscore_folder, exist_ok=True)

        with open(f"{boxscore_folder}page_source.txt", "w+", encoding="utf-8") as f:
            f.write(html)

        time.sleep(2)
    
def create_coaches_dataframe(boxscore_folder):
    """
    Using html gathered from pro-football-reference.com
    to create a table showing the coaches in each game
    """

    # getting all boxscores (games)
    all_boxscores_folders = [f.path.replace("\\", "/") for f in os.scandir(boxscore_folder) if f.is_dir()]

    print(f"Num Games: {len(all_boxscores_folders)}")

    data = []

    for step, folder in enumerate(all_boxscores_folders):
        if step % 100 == 0:
            print(f"Step Number: {step} of {len(all_boxscores_folders)}")

        html_file = f"{folder}/page_source.txt"

        with open(html_file, "r") as f:
            html_str = f.read()

            # define BeautifulSoup object
            soup = BeautifulSoup(html_str, "html.parser")

            scorebox_div = soup.find("div", {"class": "scorebox"})

            datapoint_divs = scorebox_div.find_all("div", {"class": "datapoint"})

            away_team_found = False
            for i in datapoint_divs:
                if "coach" in i.text.lower():
                    boxscore = folder.split("/")[-1]

                    if away_team_found: team = "home"
                    else: 
                        team = "away"
                        away_team_found = True

                    # extract coach link
                    coach_href = i.find("a").get("href")
                    coach_name = i.find('a').text.strip()

                    # extract team link
                    parent = i.parent
                    a_tags = parent.find_all("a")
                    a_tags = [i for i in a_tags if "/teams/" in i.get("href")]

                    assert len(a_tags)==1

                    team_link = a_tags[0].get("href")

                    # extract game day
                    date_str = scorebox_div.find("div", {"class": "scorebox_meta"}).find("div").text.strip()
                    date_obj = parse(date_str, fuzzy=False)
                    
                    data.append((coach_name, coach_href, team_link, team, boxscore, date_obj))
    
    df = pd.DataFrame(data, columns=['coach_name', 'coach_link', 'team_link', 'home_away', 'boxscore_id', 'date'])

    print(df)

    df.to_csv(script_dir+"../data/raw_game_level_coach_data.csv", index=False)

def convert_date_to_season(x):
    if x.month <= 6:
        return x.year - 1
    else:
        return x.year

def process_game_level_coach_data():
    df = pd.read_csv(script_dir+"../data/raw_game_level_coach_data.csv")

    # get coach ids
    df['coach_id'] = df['coach_link'].str.split("/", expand=False).str[-1].str.replace(".htm", "")
    
    # get team_ids
    df['team_id'] = df['team_link'].str.split("/", expand=False).str[2]

    # get season 
    df['season'] = pd.to_datetime(df['date']).apply(convert_date_to_season)

    print(df)

    df.to_csv(script_dir+"../data/game_level_coach_data.csv", index=False)

if __name__ == '__main__':
    # get_boxscores_html()
    # create_coaches_dataframe(boxscore_folder="C:/Users/Robert Brown/OneDrive/boxscore_data_files")
    process_game_level_coach_data()
