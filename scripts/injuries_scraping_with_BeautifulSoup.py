
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import re
import time
import random
import os

teams = [
    "crd", "atl", "rav", "buf", "car", "chi", "cin", "cle", "dal", "den",
    "det", "gnb", "htx", "clt", "jax", "kan", "sdg", "ram", "rai", "mia",
    "min", "nor", "nwe", "nyg", "nyj", "phi", "pit", "sea", "sfo", "tam",
    "oti", "was"
]

years = ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020","2021" , "2022", "2023"]  # Add more years as needed

def scrape_pfr_injuries(url):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    
    # Get player names, dates, and opponents
    players = [player.get_text() for player in soup.select('.left')][1:-1]
    player_id = [th.get('data-append-csv') for th in soup.select('th[data-stat="player"]')][1:]
    dates = [th.get_text(strip=True).split()[0][:5] for th in soup.find_all('th', class_='poptip center')]
    opp = [th.get_text(strip=True).split()[1] for th in soup.find_all('th', class_='poptip center')]
    
    # Get team and season information
    team = soup.select_one('#meta div:nth-of-type(2) h1 span:nth-of-type(2)').get_text()
    season = soup.select_one('#meta div:nth-of-type(2) h1 span:nth-of-type(1)').get_text()

    # Create a DataFrame to store roster info
    df_roster = pd.DataFrame(players, columns=["names"])
    df_roster["team"] = team
    df_roster["season"] = season
    df_roster['player_id'] = player_id

    # Set the number of players and games
    num_players = len(df_roster)
    num_games = len(dates)
    
    # Create blank DataFrames to store injury info
    df_active = pd.DataFrame(index=range(num_players), columns=range(num_games))
    df_weeks = pd.DataFrame(index=range(num_players), columns=range(num_games))
    df_injuries = pd.DataFrame(index=range(num_players), columns=range(num_games))


    df_opp = pd.DataFrame(index=range(num_players), columns=opp)
    df_date = pd.DataFrame(index=range(num_players), columns=dates)


    # Iterate over players and games
    for x in range(num_players):
        for y in range(num_games):
            try:
                injury_info = soup.select_one(f'#team_injuries tbody tr:nth-of-type({x+1}) td:nth-of-type({y+1})')
                if injury_info:
                    class_attr = injury_info.get('class', [])
                    df_active.iat[x, y] = 'dnp' if 'dnp' in class_attr else None
                    df_weeks.iat[x, y] = injury_info.get('data-stat')
                    injury_tip = injury_info.get('data-tip')
                    
                    # If the injury_tip contains digits, replace it with None
                    if injury_tip and re.search(r'\d', injury_tip):
                        df_injuries.iat[x, y] = None
                    else:
                        df_injuries.iat[x, y] = injury_tip
            except Exception as e:
                pass

    # Reshape the data for easier interpretation
    df_active_melt = df_active.melt(ignore_index=False, var_name='Week', value_name='active_inactive')
    df_weeks_melt = df_weeks.melt(ignore_index=False, var_name='Week', value_name='Week_Info')
    df_injuries_melt = df_injuries.melt(ignore_index=False, var_name='Week', value_name='Injury_Info')
    df_opp_melt = df_opp.melt(ignore_index=False, var_name='opp_team', value_name='opp_value')
    df_date_melt = df_date.melt(ignore_index=False, var_name='date', value_name='date_value')


    df_roster_expanded = pd.concat([df_roster] * len(dates), ignore_index=True)
    
    # Combine all data
    df_final = pd.concat([df_roster_expanded.reset_index(drop=True), df_active_melt.reset_index(drop=True),
                          df_weeks_melt.reset_index(drop=True), df_injuries_melt.reset_index(drop=True), 
                          df_opp_melt.reset_index(drop=True),df_date_melt.reset_index(drop=True)], axis=1)
    

    df_final = df_final.drop(columns=['Week','opp_value','date_value'])
    
    # Process injury and availability information
    df_final['active_inactive'] = df_final['active_inactive'].replace('dnp', 'Out').fillna('Active')
    df_final['injury_type'] = df_final['Injury_Info'].apply(
        lambda x: 'Healthy' if pd.isna(x) else x.split(':')[-1].strip().title() if isinstance(x, str) else 'Healthy'
    ) 

    df_final['game_status'] = df_final['Injury_Info'].apply(
        lambda x: 'Available' if pd.isna(x) else x.split(':')[0].strip().title() if isinstance(x, str) else 'Available'
    )
    
    # Remove unnecessary columns and filter valid week numbers
    df_final = df_final.drop(columns=['Week_Info', 'Injury_Info'])

    column_order = ['player_id','names', 'team', 'season', 'date', 'opp_team', 'active_inactive', 'injury_type', 'game_status']
    df_final = df_final[column_order]

    return df_final


# url = f"https://www.pro-football-reference.com/teams/{team}/{year}_injuries.htm"

# for year in years:
#     all_data = []

#     # Loop through teams for the current year and scrape data
#     for team in tqdm(teams):
#         url = f"https://www.pro-football-reference.com/teams/{team}/{year}_injuries.htm"
        
#         # Scrape the data from the URL
#         data = scrape_pfr_injuries(url)
#         all_data.append(data)

#         # Sleep for a random time between 3.5 to 5.5 seconds
#         time.sleep(random.uniform(3.5, 5.5))

#     # Combine all the data into a single DataFrame for the current year
#     final_df = pd.concat(all_data, ignore_index=True)