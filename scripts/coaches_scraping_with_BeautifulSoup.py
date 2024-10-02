
import os
import time
import random
import requests
from bs4 import BeautifulSoup
import pandas as pd


def scrape_coaches_data(url,year):
    # Fetch the page content
    season = year
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    
    # Extract coach names and team names
    coaches = [coach.get_text() for coach in soup.select('th[data-stat="coach"] a')]
    coaches_id = [coach['href'][-11:-4] for coach in soup.select('th[data-stat="coach"] a')] 
    teams = [team.get_text() for team in soup.select('td[data-stat="team"] a')]
    
    # Define the stats we want to extract
    stats_to_extract = {
        'Season_Games': 'g',
        'Season_Wins': 'wins',
        'Season_Losses': 'losses',
        'Season_Ties': 'ties',
        'Playoff_Games': 'g_playoffs',
        'Playoff_Wins': 'wins_playoffs',
        'Playoff_Losses': 'losses_playoffs',
    }
    
    # Extract the stats in a loop to avoid redundancy
    extracted_data = {}
    for stat_name, stat_selector in stats_to_extract.items():
        extracted_data[stat_name] = [stat.get_text() for stat in soup.select(f'td[data-stat="{stat_selector}"]')]
    
    # Construct the DataFrame
    df_coaches = pd.DataFrame({
        'Coach': coaches,
        'Coach_id':coaches_id,
        'Team': teams,
        'Season':season,
        **extracted_data  # Unpack the dictionary to include all extracted stats
    })
    
    # Convert numeric columns to appropriate types
    numeric_columns = list(stats_to_extract.keys())
    df_coaches[numeric_columns] = df_coaches[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # Return the final DataFrame
    return df_coaches

# years = list(range(2013,2024))
# # Loop over each year and scrape the data
# for year in years:
#     url = f"https://www.pro-football-reference.com/years/{year}/coaches.htm"
    
#     # Scrape the data for the given year
#     df = scrape_coaches_data(url,year)
    
#     # Sleep for a random time between 3.5 to 5.5 seconds to avoid overwhelming the server
#     time.sleep(random.uniform(3.5, 5.5))
