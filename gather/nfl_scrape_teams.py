# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 22:53:55 2024

@author: masca
"""

import pandas as pd
#import nflscraPy as nfl
from bs4 import BeautifulSoup
import requests
import re
from alive_progress import alive_bar
import time


url = 'https://www.pro-football-reference.com'
years = range(2013, 2024)


# Column header rename

UPDATED_COLUMNS = {
    ("Score", "Tm") : ("Score", "score_team"),
    ("Score", "Opp"): ("Score", "score_opp"),
    ("Offense", "1stD"): ("Offense", "off_first_downs"),
    ("Offense", "TotYd"): ("Offense", "off_yds_total"),
    ("Offense", "PassY"): ("Offense", "off_yds_pass"),
    ("Offense", "RushY"): ("Offense", "off_yds_rush"),
    ("Offense", "TO"): ("Offense", "off_timeout"),
    ("Defense", "1stD"): ("Defense", "def_first_downs"),
    ("Defense", "TotYd"): ("Defense", "def_yds_total"),
    ("Defense", "PassY"): ("Defense", "def_yds_pass"),
    ("Defense", "RushY"): ("Defense", "def_yds_rush"),
    ("Defense", "TO"): ("Defense", "def_timeout"),
    ("Expected Points", "Offense"): ("Expected Points", "off_exp_pts"),
    ("Expected Points", "Defense"): ("Expected Points", "def_exp_pts"),
    ("Expected Points", "Sp. Tms"): ("Expected Points", "sptm_exp_pts")
    }


def read_boxscore_stub(stub:str):
    '''
    Takes in the current team and year's stub and pulls the href tag
    that refers to the boxscore
    '''
    r = requests.get(url + stub)
    temp_games = BeautifulSoup(r.content, 'html.parser')
    # Pull the html that makes up the season for the team in question
    game_results_html = temp_games.find_all('table')[1]
    
    # Convert each table row to a list - skip the top 2 headers
    game_rows = game_results_html.find_all('tr')[2:]

    # Save the href tag for the boxscores
    game_score_stub = []

    # example stub is: "/boxscores/201401110nwe.htm"
    for game_row in game_rows:
        boxscore_stub = game_row.find('td', attrs={'data-stat': 'boxscore_word'})
        if boxscore_stub.a is None:
            game_score_stub.append("")
        else:
            game_score_stub.append(
                boxscore_stub.a.get('href')
            )
    return game_score_stub


# We'll use the fantasy listing to get all players per year
def scrape_team_years(year:int)-> pd.DataFrame:
    '''
    Parameters
    ----------
    year : INT
        the year to be returned.

    Returns
    -------
    DataFrame.

    This function does the majority of the scraping from the website
    'https://www.pro-football-reference.com'. The goal is to scrape each
    team's performance over the years
    '''
    url = 'https://www.pro-football-reference.com'
    r = requests.get(url + '/years/' + str(year) + '/')
    
    # STARTER MESSAGE
    print(f'Beginning process for {year} using the url: \
          {url + "/years/" + str(year)}')
    soup = BeautifulSoup(r.content, 'html.parser')
    afc_table = soup.find_all('table')[0]
    nfc_table = soup.find_all('table')[1]
    
    df = []
    
    # Make our team list
    team_list = afc_table.find_all('tr')[2:] + \
                nfc_table.find_all('tr')[2:]
    total_list_len = len(team_list)
    
    with alive_bar(total_list_len) as bar:
        for i, row in enumerate(team_list):
            try:
                dat = row.find('th', attrs={'data-stat': 'team'})
                team_name = dat.a.get_text()
                print(f'Evaluating {team_name} - team number {i}')
                
                # Use the a-href to get the URL stub of the team
                stub = dat.a.get('href')
                stub = stub[:-4] + '.htm'
                
                # Now read in the gamelog for this year for this team
                tdf = pd.read_html(url + stub)[1]

                # Store boxscore stubs per game
                boxscore_stubs = read_boxscore_stub(stub)
                tdf['boxscore_stub'] = boxscore_stubs
                tdf['boxscore_link'] = [url + stub for stub in boxscore_stubs]
                
                # Use our dictionary to rename the columns
                # We'll only be keeping the headers at index 1 eventually
                tdf.columns = pd.MultiIndex.from_tuples(
                    tdf.set_axis(tdf.columns.values, axis=1)
                        .rename(columns=UPDATED_COLUMNS)
                    )
                
                # Get rid of multiindex, just keep the last row
                tdf.columns = tdf.columns.get_level_values(-1)
                
                # Fix the away/home column
                tdf = tdf.rename(columns={'Unnamed: 8_level_1': 'Away'})
                tdf.Away = [1 if r=='@' else 0 for r in tdf['Away']]
    
                # Change column name for result
                tdf.rename(columns={'Unnamed: 5_level_1': 'result'}, inplace=True)
    
                # Drop boxscore descriptor
                tdf.drop('Unnamed: 4_level_1', axis=1, inplace=True)
                tdf.OT = tdf.OT.apply(lambda x: 1 if x == 'OT' else 0)
                
                # Capture Playoff information (if it exists)
                tdf["playoffs"] = 0
                playoff_marker = tdf.index[
                    tdf.Date.str.contains("Playoffs", na=False)
                    ]
                
                # Adjust all games after the playoff marker to flag=1
                if not playoff_marker.empty:
                    playoff_index = playoff_marker[0]
                    tdf.loc[tdf.index > playoff_index, 'playoffs'] = 1
    
                # Clean up the table
                tdf = tdf.dropna(subset=['Day'])
                
                # Add Remaining info
                tdf['team_name'] = team_name
                tdf['team_id'] = re.search(r'/teams/([a-z]{3})/', stub).group(1)
                tdf['year'] = year
                tdf['team_stub'] = stub
                tdf['team_link'] = url + stub
                
                # Make our dataframe
                df.append(tdf)
                time.sleep(5)
            except:
                pass
            
            bar()
            
    df = pd.concat(df) 
    return df

#scrape_team_years(2013)