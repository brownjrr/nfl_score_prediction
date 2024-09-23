# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 00:37:15 2024

@author: masca
"""

import nfl_scrape_teams
import pandas as pd
import time

years = range(2013, 2024)
teams_df = []

for year in years:
    tdf = nfl_scrape_teams.scrape_team_years(year)
    time.sleep(30)
    teams_df.append(tdf)
    
    print(f'Finished scraping the year {year}')
    
teams_df = pd.concat(teams_df)