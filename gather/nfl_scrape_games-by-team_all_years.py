# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 00:37:15 2024

@author: masca
"""

import gather.nfl_scrape_teams as nfl_scrape
import pandas as pd
import time
from pathlib import Path
import os

years = range(2013, 2024)
teams_df = []

for year in years:
    tdf = nfl_scrape.scrape_team_years(year)
    tdf.to_csv(f'tmp_{year}.csv')
    time.sleep(30)
    teams_df.append(tdf)
    
    print(f'Finished scraping the year {year}')
    
teams_df = pd.concat(teams_df)
teams_df.to_csv("data/raw_teams.csv")

# Delete our temp files
#tmp_path = Path('.')
#for f in tmp_path.glob('tmp*.csv'):
#    os.remove(f)