
import os
import time
import random
import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd

def parse_weather(weather_text):
    # Parse the weather information to extract temperature, humidity, and wind speed
    temperature = None
    humidity_pct = None
    wind_speed = None

    parts = weather_text.split(", ")
    for part in parts:
        if 'degrees' in part:
            temperature = float(part.split()[0])
        if 'humidity' in part:
            humidity_pct = float(part.split()[2].replace('%', ''))
        if 'mph' in part:
            wind_speed = float(part.split()[1])
    
    return temperature, humidity_pct, wind_speed

def date_func(url):
    event_date = f'{url[-16:-12]}-{url[-12:-10]}-{url[-10:-8]}'
    return event_date

def game_starters(comments,event_date):
    # Initialize a list to store table data
    table_data = [] 

    # Iterate over each comment and parse the one that contains 'home_starters' or 'vis_starters'
    for comment in comments:
        if 'id="home_starters"' in comment:
            comment_soup = BeautifulSoup(comment, 'html.parser')
            table = comment_soup.find('table')
            
            if table:
                # Extract the team name from the caption dynamically
                team_name = table.find('caption').get_text().split(' Starters Table')[0]
                rows = table.find_all('tr')[1:]  # Skip header row

                # Loop through each row and extract player data
                for row in rows:
                    player_tag = row.find('th', {'data-stat': 'player'})
                    if player_tag:
                        player_id = player_tag['data-append-csv']
                        player_name = player_tag.find('a').get_text()
                        position = row.find('td', {'data-stat': 'pos'}).get_text()
                        home = 1
                        date = event_date
                        table_data.append([date, team_name, player_id, player_name, position, home])
        elif 'id="vis_starters"' in comment:
            comment_soup = BeautifulSoup(comment, 'html.parser')
            table = comment_soup.find('table')
            
            if table:
                # Extract the team name from the caption dynamically
                team_name = table.find('caption').get_text().split(' Starters Table')[0]
                rows = table.find_all('tr')[1:]  # Skip header row

                # Loop through each row and extract player data
                for row in rows:
                    player_tag = row.find('th', {'data-stat': 'player'})
                    if player_tag:
                        player_id = player_tag['data-append-csv']
                        player_name = player_tag.find('a').get_text()
                        position = row.find('td', {'data-stat': 'pos'}).get_text()
                        home = 0
                        date = event_date
                        table_data.append([date, team_name, player_id, player_name, position, home])

    # Convert the list to a DataFrame
    df = pd.DataFrame(table_data, columns=["Date","Team", "Player_id", "Starter", "Position", "Home"])
    return df


def snap_counts(comments,event_date):
    # Initialize a list to store table data
    snap_data = [] 

    # Iterate over each comment and parse the one that contains 'home_starters' or 'vis_starters'
    for comment in comments:
        if 'id="home_snap_counts"' in comment:
            comment_soup = BeautifulSoup(comment, 'html.parser')
            table = comment_soup.find('table')
            
            if table:
                # Extract the team name from the caption dynamically
                team_name = table.find('caption').get_text().split(' Snap Counts Table')[0]
                rows = table.find_all('tr')[2:]  # Skip header row

                for row in rows:
                    # Extract each field (skip rows without enough columns)
                    columns = row.find_all('td')

                    # extract player data
                    player_tag = row.find('th', {'data-stat': 'player'})
                    if player_tag:
                        player_id = player_tag['data-append-csv']
                        player_name = player_tag.find('a').get_text()

                    # Extract required data
                    pos = columns[0].text.strip()
                    off_num = columns[1].text.strip()
                    off_pct = columns[2].text.strip()
                    def_num = columns[3].text.strip()
                    def_pct = columns[4].text.strip()
                    st_num = columns[5].text.strip()
                    st_pct = columns[6].text.strip()
                    date = event_date
                    snap_data.append([date, team_name, player_id, player_name, pos, off_num, off_pct, def_num, def_pct, st_num, st_pct])
                    
        if 'id="vis_snap_counts"' in comment:
            comment_soup = BeautifulSoup(comment, 'html.parser')
            table = comment_soup.find('table')
            
            if table:
                # Extract the team name from the caption dynamically
                team_name = table.find('caption').get_text().split(' Snap Counts Table')[0]
                rows = table.find_all('tr')[2:]  # Skip header row

                for row in rows:
                    # Extract each field (skip rows without enough columns)
                    columns = row.find_all('td')

                    # extract player data
                    player_tag = row.find('th', {'data-stat': 'player'})
                    if player_tag:
                        player_id = player_tag['data-append-csv']
                        player_name = player_tag.find('a').get_text()

                    # Extract required data
                    pos = columns[0].text.strip()
                    off_num = columns[1].text.strip()
                    off_pct = columns[2].text.strip()
                    def_num = columns[3].text.strip()
                    def_pct = columns[4].text.strip()
                    st_num = columns[5].text.strip()
                    st_pct = columns[6].text.strip()
                    date = event_date
                    snap_data.append([date, team_name, player_id, player_name, pos, off_num, off_pct, def_num, def_pct, st_num, st_pct])

    # Convert the list to a DataFrame
    df = pd.DataFrame(snap_data, columns=["date","team", "player_id", "player", "position", "off_num","off_pct","def_num","def_pct","st_num","st_pct"])
    return df


def player_offense(soup,event_date):
    # Find the table by ID
    table = soup.find('table', id='player_offense')

    # Initialize a list to hold player data
    offense_data = []

    # Loop through the rows in the table body
    for row in table.find('tbody').find_all('tr'):
        # Create a dictionary to store the player's stats
        player_stats = {}
        
        data_tag = row.find_all('td')
        if not data_tag:
            # Skip rows that have fewer columns (short rows)
            continue
        # Extract the player's name
        try:
            player_tag = row.find('th', {'data-stat': 'player'})
            if player_tag:
                player_id = player_tag['data-append-csv']
                player_name = player_tag.find('a').get_text()

            team = data_tag[0].text.strip()
            pass_cmp = data_tag[1].text.strip()
            pass_att = data_tag[2].text.strip()
            pass_yds = data_tag[3].text.strip()
            pass_td = data_tag[4].text.strip()
            pass_int = data_tag[5].text.strip()
            pass_sacked = data_tag[6].text.strip()
            pass_sacked_yds = data_tag[7].text.strip()
            pass_long = data_tag[8].text.strip()
            pass_rating = data_tag[9].text.strip()
            rush_att = data_tag[10].text.strip()
            rush_yds = data_tag[11].text.strip()
            rush_td = data_tag[12].text.strip()
            rush_long = data_tag[13].text.strip()
            rec_tgt = data_tag[14].text.strip()
            rec_rec = data_tag[15].text.strip()
            rec_yds = data_tag[16].text.strip()
            rec_td = data_tag[17].text.strip()
            rec_long = data_tag[18].text.strip()
            fmb = data_tag[19].text.strip()
            fmb_lost = data_tag[20].text.strip()
        except:
            pass
        clmn = [event_date,player_id,player_name,team,pass_cmp,pass_att,pass_yds,pass_td,pass_int,pass_sacked,pass_sacked_yds,pass_long,pass_rating,
                rush_att,rush_yds,rush_td,rush_long,rec_tgt,rec_rec,rec_yds,rec_td,rec_long,fmb,fmb_lost]
        offense_data.append(clmn)
    df = pd.DataFrame(offense_data,columns=['date','player_id', 'player_name', 'team', 'pass_cmp', 'pass_att', 'pass_yds', 'pass_td', 'pass_int', 'pass_sacked', 'pass_sacked_yds',
    'pass_long', 'pass_rating', 'rush_att', 'rush_yds', 'rush_td', 'rush_long', 'rec_tgt', 'rec_rec', 'rec_yds', 'rec_td', 
    'rec_long', 'fmb', 'fmb_lost'])
    return df

    
def player_defense(comments,event_date):
    # Initialize a list to store table data
    diff_data = [] 

    # Iterate over each comment and parse the one that contains 'home_starters' or 'vis_starters'
    for comment in comments:
        if 'id="player_defense"' in comment:
                comment_soup = BeautifulSoup(comment, 'html.parser')
                table = comment_soup.find('table')

                if table:
                    # Extract the team name from the caption dynamically
                    rows = table.find_all('tr')[2:]  # Skip header row
                    # Extract each field (skip rows without enough columns)
                    for row in rows:
                        data_tag = row.find_all('td')
                        if not data_tag:
                            # Skip rows that have fewer columns (short rows)
                            continue
    
                        # extract player data
                        player_tag = row.find('th', {'data-stat': 'player'})
                        if player_tag:
                            player_id = player_tag['data-append-csv']
                            player_name = player_tag.find('a').get_text()
                        team = data_tag[0].text.strip()
                        def_int = data_tag[1].text.strip()
                        def_int_yds = data_tag[2].text.strip()
                        def_int_td = data_tag[3].text.strip()
                        def_int_long = data_tag[4].text.strip()
                        pass_defended = data_tag[5].text.strip()
                        sacks = data_tag[6].text.strip()
                        tackles_combined = data_tag[7].text.strip()
                        tackles_solo = data_tag[8].text.strip()
                        tackles_assists = data_tag[9].text.strip()
                        tackles_loss = data_tag[10].text.strip()
                        qb_hits = data_tag[11].text.strip()
                        fumbles_rec = data_tag[12].text.strip()
                        fumbles_rec_yds = data_tag[13].text.strip()
                        fumbles_rec_td = data_tag[14].text.strip()
                        fumbles_forced = data_tag[15].text.strip()
                        diff_data.append([event_date,player_id, player_name, team,def_int,def_int_yds,def_int_td,def_int_long,pass_defended,sacks,tackles_combined,tackles_solo,
                                    tackles_assists,tackles_loss,qb_hits,fumbles_rec,fumbles_rec_yds,fumbles_rec_td,fumbles_forced])
                    
    df = pd.DataFrame(diff_data,columns=['date','player_id','player','team', 'def_int', 'def_int_yds', 'def_int_td', 'def_int_long', 'pass_defended', 'sacks', 'tackles_combined', 'tackles_solo',
                                                    'tackles_assists', 'tackles_loss', 'qb_hits', 'fumbles_rec', 'fumbles_rec_yds', 'fumbles_rec_td', 'fumbles_forced'])
    return df             

def player_returns(comments,event_date):
    # Initialize a list to store table data
    returns_data = [] 

    # Iterate over each comment and parse the one that contains 'home_starters' or 'vis_starters'
    for comment in comments:
        if 'id="returns"' in comment:
                comment_soup = BeautifulSoup(comment, 'html.parser')
                table = comment_soup.find('table')

                if table:
                    # Extract the team name from the caption dynamically
                    rows = table.find_all('tr')[2:]  # Skip header row
                    # Extract each field (skip rows without enough columns)
                    for row in rows:
                        data_tag = row.find_all('td')
                        if not data_tag:
                            # Skip rows that have fewer columns (short rows)
                            continue

                        # extract player data
                        player_tag = row.find('th', {'data-stat': 'player'})
                        if player_tag:
                            player_id = player_tag['data-append-csv']
                            player_name = player_tag.find('a').get_text()
                        team = data_tag[0].text.strip() 
                        kick_ret = data_tag[1].text.strip()
                        kick_ret_yds = data_tag[2].text.strip() 
                        kick_ret_yds_per_ret = data_tag[3].text.strip() 
                        kick_ret_td = data_tag[4].text.strip()
                        kick_ret_long = data_tag[5].text.strip()
                        punt_ret = data_tag[6].text.strip()
                        punt_ret_yds = data_tag[7].text.strip()
                        punt_ret_yds_per_ret = data_tag[8].text.strip() 
                        punt_ret_td = data_tag[9].text.strip()
                        punt_ret_long= data_tag[10].text.strip()
                        returns_data.append([event_date,player_id,player_name,team,kick_ret,kick_ret_yds,kick_ret_yds_per_ret,kick_ret_td,kick_ret_long,
                                            punt_ret,punt_ret_yds,punt_ret_yds_per_ret,punt_ret_td,punt_ret_long])
    df = pd.DataFrame(returns_data,columns=['date','player_id', 'player_name', 'team', 'kick_ret', 'kick_ret_yds', 'kick_ret_yds_per_ret', 
                                                    'kick_ret_td', 'kick_ret_long','punt_ret', 'punt_ret_yds', 'punt_ret_yds_per_ret', 'punt_ret_td', 'punt_ret_long'])
    return  df

def player_kicking(comments,event_date):
    # Initialize a list to store table data
    kicking_data = [] 

    # Iterate over each comment and parse the one that contains 'home_starters' or 'vis_starters'
    for comment in comments:
        if 'id="kicking"' in comment:
                comment_soup = BeautifulSoup(comment, 'html.parser')
                table = comment_soup.find('table')

                if table:
                    # Extract the team name from the caption dynamically
                    rows = table.find_all('tr')[2:]  # Skip header row
                    # Extract each field (skip rows without enough columns)
                    for row in rows:
                        data_tag = row.find_all('td')
                        if not data_tag:
                            # Skip rows that have fewer columns (short rows)
                            continue
                        # extract player data
                        player_tag = row.find('th', {'data-stat': 'player'})
                        if player_tag:
                            player_id = player_tag['data-append-csv']
                            player_name = player_tag.find('a').get_text()

                        team = data_tag[0].text.strip() 
                        scoring_xpm = data_tag[1].text.strip()
                        scoring_xpa = data_tag[2].text.strip() 
                        scoring_fgm = data_tag[3].text.strip() 
                        scoring_fga = data_tag[4].text.strip()
                        punt = data_tag[5].text.strip()
                        punt_yds = data_tag[6].text.strip()
                        punt_yds_per_punt = data_tag[7].text.strip()
                        punt_long = data_tag[8].text.strip() 
                        kicking_data.append([event_date,player_id, player_name, team,scoring_xpm,scoring_xpa,scoring_fgm,scoring_fga,punt,punt_yds,punt_yds_per_punt,punt_long])
    df = pd.DataFrame(kicking_data,columns=['date','player_id', 'player_name', 'team', 'scoring_xpm', 'scoring_xpa', 'scoring_fgm', 'scoring_fga',
                                                        'punt', 'punt_yds', 'punt_yds_per_punt', 'punt_long'])
    return df
                    

def player_advanced_passing(comments,event_date):
    # Initialize a list to store table data
    advanced_passing_data = [] 

    # Iterate over each comment and parse the one that contains 'home_starters' or 'vis_starters'
    for comment in comments:
        if 'id="passing_advanced"' in comment:
                comment_soup = BeautifulSoup(comment, 'html.parser')
                table = comment_soup.find('table')

                if table:
                    # Extract the team name from the caption dynamically
                    rows = table.find_all('tr')[1:]  # Skip header row
                    # Extract each field (skip rows without enough columns)
                    for row in rows:
                        data_tag = row.find_all('td')
                        if not data_tag:
                            # Skip rows that have fewer columns (short rows)
                            continue
                        # extract player data
                        player_tag = row.find('th', {'data-stat': 'player'})
                        if player_tag:
                            player_id = player_tag['data-append-csv']
                            player_name = player_tag.find('a').get_text()
                        team = data_tag[0].text.strip()
                        pass_cmp = data_tag[1].text.strip()
                        pass_att = data_tag[2].text.strip()
                        pass_yds = data_tag[3].text.strip()
                        pass_first_down = data_tag[4].text.strip()
                        pass_first_down_pct = data_tag[5].text.strip()
                        pass_target_yds = data_tag[6].text.strip()
                        pass_tgt_yds_per_att = data_tag[7].text.strip()
                        pass_air_yds = data_tag[8].text.strip()
                        pass_air_yds_per_cmp = data_tag[9].text.strip()
                        pass_air_yds_per_att = data_tag[10].text.strip()
                        pass_yac = data_tag[11].text.strip()
                        pass_yac_per_cmp = data_tag[12].text.strip()
                        pass_drops = data_tag[13].text.strip()
                        pass_drop_pct = data_tag[14].text.strip()
                        pass_poor_throws = data_tag[15].text.strip()
                        pass_poor_throw_pct = data_tag[16].text.strip()
                        pass_sacked = data_tag[17].text.strip()
                        pass_blitzed = data_tag[18].text.strip()
                        pass_hurried = data_tag[19].text.strip()
                        pass_hits = data_tag[20].text.strip()
                        pass_pressured = data_tag[21].text.strip()
                        pass_pressured_pct = data_tag[22].text.strip()
                        rush_scrambles = data_tag[23].text.strip()
                        rush_scrambles_yds_per_att = data_tag[24].text.strip()

                        # Append the extracted stats to the player_stats_list
                        advanced_passing_data.append([event_date,player_id, player_name, team, pass_cmp, pass_att, pass_yds, pass_first_down, pass_first_down_pct, pass_target_yds, 
                                                pass_tgt_yds_per_att, pass_air_yds, pass_air_yds_per_cmp, pass_air_yds_per_att, pass_yac, 
                                                pass_yac_per_cmp, pass_drops, pass_drop_pct, pass_poor_throws, pass_poor_throw_pct, pass_sacked,
                                                pass_blitzed, pass_hurried, pass_hits, pass_pressured, pass_pressured_pct, rush_scrambles, 
                                                rush_scrambles_yds_per_att])

    # Create a DataFrame from the extracted data
    df = pd.DataFrame(advanced_passing_data, columns=[
                        'date','player_id', 'player_name', 'team', 'pass_cmp', 'pass_att', 'pass_yds', 'pass_first_down', 'pass_first_down_pct', 'pass_target_yds', 'pass_tgt_yds_per_att', 
                        'pass_air_yds', 'pass_air_yds_per_cmp', 'pass_air_yds_per_att', 'pass_yac', 'pass_yac_per_cmp', 'pass_drops', 'pass_drop_pct',
                        'pass_poor_throws', 'pass_poor_throw_pct', 'pass_sacked', 'pass_blitzed', 'pass_hurried', 'pass_hits', 'pass_pressured', 
                        'pass_pressured_pct', 'rush_scrambles', 'rush_scrambles_yds_per_att'])
    return df
                                        

def player_advanced_rushing(comments,event_date):
    # Initialize a list to store table data
    advanced_rushing_data = [] 

    # Iterate over each comment and parse the one that contains 'home_starters' or 'vis_starters'
    for comment in comments:
        if 'id="rushing_advanced"' in comment:
                comment_soup = BeautifulSoup(comment, 'html.parser')
                table = comment_soup.find('table')

                if table:
                    # Extract the team name from the caption dynamically
                    rows = table.find_all('tr')[1:]  # Skip header row
                    # Extract each field (skip rows without enough columns)
                    for row in rows:
                        data_tag = row.find_all('td')
                        if not data_tag:
                            # Skip rows that have fewer columns (short rows)
                            continue
                        # extract player data
                        player_tag = row.find('th', {'data-stat': 'player'})
                        if player_tag:
                            player_id = player_tag['data-append-csv']
                            player_name = player_tag.find('a').get_text()
                        team = data_tag[0].text.strip()
                        rush_att = data_tag[1].text.strip()
                        rush_yds = data_tag[2].text.strip()
                        rush_td = data_tag[3].text.strip()
                        rush_first_down = data_tag[4].text.strip()
                        rush_yds_before_contact = data_tag[5].text.strip()
                        rush_yds_bc_per_rush = data_tag[6].text.strip()
                        rush_yac = data_tag[7].text.strip()
                        rush_yac_per_rush = data_tag[8].text.strip()
                        rush_broken_tackles = data_tag[9].text.strip()
                        rush_broken_tackles_per_rush = data_tag[10].text.strip()

                        # Append the extracted stats to the rushing_stats_list
                        advanced_rushing_data.append([event_date,player_id, player_name, team, rush_att, rush_yds, rush_td, rush_first_down, rush_yds_before_contact, 
                                                rush_yds_bc_per_rush, rush_yac, rush_yac_per_rush, rush_broken_tackles, rush_broken_tackles_per_rush])

    # Create a DataFrame from the extracted data
    df = pd.DataFrame(advanced_rushing_data, columns=[
                        'date','player_id', 'player_name', 'team', 'rush_att', 'rush_yds', 'rush_td', 'rush_first_down', 'rush_yds_before_contact', 
                        'rush_yds_bc_per_rush', 'rush_yac', 'rush_yac_per_rush', 'rush_broken_tackles', 'rush_broken_tackles_per_rush'
                    ])
    return df

def player_advanced_receiving(comments,event_date):
    # Initialize a list to store table data
    advanced_receiving_data = [] 

    # Iterate over each comment and parse the one that contains 'home_starters' or 'vis_starters'
    for comment in comments:
        if 'id="receiving_advanced"' in comment:
                comment_soup = BeautifulSoup(comment, 'html.parser')
                table = comment_soup.find('table')

                if table:
                    # Extract the team name from the caption dynamically
                    rows = table.find_all('tr')[1:]  # Skip header row
                    # Extract each field (skip rows without enough columns)
                    for row in rows:
                        data_tag = row.find_all('td')
                        if not data_tag:
                            # Skip rows that have fewer columns (short rows)
                            continue
                        # extract player data
                        player_tag = row.find('th', {'data-stat': 'player'})
                        if player_tag:
                            player_id = player_tag['data-append-csv']
                            player_name = player_tag.find('a').get_text()
                        team = data_tag[0].text.strip()
                        targets = data_tag[1].text.strip()
                        rec = data_tag[2].text.strip()
                        rec_yds = data_tag[3].text.strip()
                        rec_td = data_tag[4].text.strip()
                        rec_first_down = data_tag[5].text.strip()
                        rec_air_yds = data_tag[6].text.strip()
                        rec_air_yds_per_rec = data_tag[7].text.strip()
                        rec_yac = data_tag[8].text.strip()
                        rec_yac_per_rec = data_tag[9].text.strip()
                        rec_adot = data_tag[10].text.strip()
                        rec_broken_tackles = data_tag[11].text.strip()
                        rec_broken_tackles_per_rec = data_tag[12].text.strip()
                        rec_drops = data_tag[13].text.strip()
                        rec_drop_pct = data_tag[14].text.strip()
                        rec_target_int = data_tag[15].text.strip()
                        rec_pass_rating = data_tag[16].text.strip()

                        # Append the extracted stats to the receiving_stats_list
                        advanced_receiving_data.append([event_date,player_id, player_name, team, targets, rec, rec_yds, rec_td, rec_first_down, rec_air_yds, 
                                                    rec_air_yds_per_rec, rec_yac, rec_yac_per_rec, rec_adot, rec_broken_tackles,
                                                    rec_broken_tackles_per_rec, rec_drops, rec_drop_pct, rec_target_int, rec_pass_rating])

    # Create a DataFrame from the extracted data
    df = pd.DataFrame(advanced_receiving_data, columns=[
                        'date','player_id', 'player_name', 'team', 'targets', 'rec', 'rec_yds', 'rec_td', 'rec_first_down', 'rec_air_yds', 'rec_air_yds_per_rec', 
                        'rec_yac', 'rec_yac_per_rec', 'rec_adot', 'rec_broken_tackles', 'rec_broken_tackles_per_rec', 'rec_drops',
                        'rec_drop_pct', 'rec_target_int', 'rec_pass_rating'
                    ])
    return df

def player_advanced_defense(comments,event_date):
    # Initialize a list to store table data
    advanced_defense_data = [] 

    # Iterate over each comment and parse the one that contains 'home_starters' or 'vis_starters'
    for comment in comments:
        if 'id="defense_advanced"' in comment:
                comment_soup = BeautifulSoup(comment, 'html.parser')
                table = comment_soup.find('table')

                if table:
                    # Extract the team name from the caption dynamically
                    rows = table.find_all('tr')[1:]  # Skip header row
                    # Extract each field (skip rows without enough columns)
                    for row in rows:
                        data_tag = row.find_all('td')
                        if not data_tag:
                            # Skip rows that have fewer columns (short rows)
                            continue
                        # extract player data
                        player_tag = row.find('th', {'data-stat': 'player'})
                        if player_tag:
                            player_id = player_tag['data-append-csv']
                            player_name = player_tag.find('a').get_text()
                        team = data_tag[0].text.strip()
                        def_int = data_tag[1].text.strip()
                        def_targets = data_tag[2].text.strip()
                        def_cmp = data_tag[3].text.strip()
                        def_cmp_perc = data_tag[4].text.strip()
                        def_cmp_yds = data_tag[5].text.strip()
                        def_yds_per_cmp = data_tag[6].text.strip()
                        def_yds_per_target = data_tag[7].text.strip()
                        def_cmp_td = data_tag[8].text.strip()
                        def_pass_rating = data_tag[9].text.strip()
                        def_tgt_yds_per_att = data_tag[10].text.strip()
                        def_air_yds = data_tag[11].text.strip()
                        def_yac = data_tag[12].text.strip()
                        blitzes = data_tag[13].text.strip()
                        qb_hurry = data_tag[14].text.strip()
                        qb_knockdown = data_tag[15].text.strip()
                        sacks = data_tag[16].text.strip()
                        pressures = data_tag[17].text.strip()
                        tackles_combined = data_tag[18].text.strip()
                        tackles_missed = data_tag[19].text.strip()
                        tackles_missed_pct = data_tag[20].text.strip()

                        # Append the extracted stats to the defensive_stats_list
                        advanced_defense_data.append([event_date,player_id, player_name, team, def_int, def_targets, def_cmp, def_cmp_perc, def_cmp_yds, def_yds_per_cmp, 
                                                    def_yds_per_target, def_cmp_td, def_pass_rating, def_tgt_yds_per_att, def_air_yds, 
                                                    def_yac, blitzes, qb_hurry, qb_knockdown, sacks, pressures, tackles_combined, 
                                                    tackles_missed, tackles_missed_pct])

    # Create a DataFrame from the extracted data
    df = pd.DataFrame(advanced_defense_data, columns=[
                        'date','player_id', 'player_name', 'team', 'def_int', 'def_targets', 'def_cmp', 'def_cmp_perc', 'def_cmp_yds', 'def_yds_per_cmp', 'def_yds_per_target',
                        'def_cmp_td', 'def_pass_rating', 'def_tgt_yds_per_att', 'def_air_yds', 'def_yac', 'blitzes', 'qb_hurry', 'qb_knockdown',
                        'sacks', 'pressures', 'tackles_combined', 'tackles_missed', 'tackles_missed_pct'
                    ])

    return df

def games_info(boxscore_link):
    # return game info dict
    page = requests.get(boxscore_link)
    soup = BeautifulSoup(page.content, 'html.parser')
    comments = soup.find_all(string = lambda text: isinstance(text, Comment))

    coach_tag = soup.find_all('div', class_='datapoint')
    coaches = [coach.find('a').text for coach in coach_tag]
    coaches = "-".join(coaches)

    event_date=date_func(boxscore_link)

    #individual game data
    i_starters = game_starters(comments,event_date)

    i_snap_counts = snap_counts(comments,event_date)

    i_player_offense = player_offense(soup,event_date)

    i_player_defense = player_defense(comments,event_date)

    i_player_returns = player_returns(comments,event_date)

    i_player_kicking = player_kicking(comments,event_date)

    i_player_advance_passing = player_advanced_passing(comments,event_date)

    i_player_advance_rushing = player_advanced_rushing(comments,event_date)

    i_player_advance_receiving = player_advanced_receiving(comments,event_date)

    i_player_advance_defense = player_advanced_defense(comments,event_date)


    game_info = {
        'won_toss': None,
        'won_toss_decision': None,
        'won_toss_overtime': None,
        'won_toss_overtime_decision': None,
        'attendance': None,
        'duration': None,
        'roof_type': None,
        'surface_type': None,
        'temperature': None,
        'humidity_pct': None,
        'wind_speed': None,
        'team_spread': None,
        'over_under': None,
        'coaches':coaches
    }

    # Extract relevant data from the comments
    for comment in comments:
        if 'id="game_info"' in comment:
            comment_soup = BeautifulSoup(comment, 'html.parser')
            table = comment_soup.find('table')
            if table:
                rows = table.find_all('tr')
                
                for row in rows:
                    header = row.find('th').text.strip() if row.find('th') else None
                    value = row.find('td').text.strip() if row.find('td') else None

                    # Map the data from the table to the appropriate field
                    if header == "Won Toss":
                        won_toss_text = value
                        if 'deferred' in won_toss_text:
                            game_info['won_toss'] = won_toss_text.split()[0]
                            game_info['won_toss_decision'] = 'deferred'
                        else:
                            game_info['won_toss'] = won_toss_text
                            game_info['won_toss_decision'] = 'accepted'
                    elif header == "Won OT Toss":
                        won_ot_toss_text = value
                        if 'deferred' in won_ot_toss_text:
                            game_info['won_toss_overtime'] = won_toss_text.split()[0]
                            game_info['won_toss_overtime_decision'] = 'deferred'
                        else:
                            game_info['won_toss_overtime'] = won_ot_toss_text
                            game_info['won_toss_overtime_decision'] = 'accepted'
                    elif header == "Roof":
                        game_info['roof_type'] = value
                    elif header == "Surface":
                        game_info['surface_type'] = value
                    elif header == "Duration":
                        game_info['duration'] = int(value.split(":")[0]) * 60 + int(value.split(":")[1])
                    elif header == "Attendance":
                        game_info['attendance'] = int(value.replace(",", ""))
                    elif header == "Weather":
                        game_info['temperature'], game_info['humidity_pct'], game_info['wind_speed'] = parse_weather(value)
                    elif header == "Vegas Line":
                        game_info['team_spread'] = value  # Extract the team spread
                    elif header == "Over/Under":
                        game_info['over_under'] = value  # Extract the over/under total
    return game_info, i_starters, i_snap_counts, i_player_offense, i_player_defense, i_player_returns, i_player_kicking, i_player_advance_passing, i_player_advance_rushing, i_player_advance_receiving, i_player_advance_defense
    
def scrape_game_data(url,year):
    '''return tuple of game_data as DataFrame and starters,snap_counts, and player stats as list (11 elements)'''
    # Fetch the page content
    season = year
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    # Find all the rows in the table
    rows = soup.find_all('tr')

    # Data to be extracted
    games_data = []

    # Initialize empty lists to store DataFrames
    starters_list =[]
    snap_counts_list = []
    player_offense_list = []
    player_defense_list = [] 
    player_returns_list = [] 
    player_kicking_list = [] 
    player_advance_passing_list = [] 
    player_advance_rushing_list = [] 
    player_advance_receiving_list = []
    player_advance_defense_list = []

    for row in rows:
        # Extract each field (skip rows without enough columns)
        columns = row.find_all('td')

        week = row.find_all('th')[0].text.strip()
        if len(columns) < 13 or columns[1].text.strip() == 'Playoffs':
            continue
        
        # Extract required data
        week_day = columns[0].text.strip()
        event_date = columns[1].text.strip()
        game_time = columns[2].text.strip()
        team_a = columns[3].text.strip()
        team_b = columns[5].text.strip()
        game_location = columns[4].text.strip()

        try:
            # Logic to determine game location
            if game_location == '@':
                location = " ".join(team_b.split()[:-1])
            elif game_location == 'N':
                location = "Niether"
            else:
                location = " ".join(team_a.split()[:-1])
        except:
            continue
        
        team_a_score = columns[7].text.strip()
        team_b_score = columns[8].text.strip()
        team_a_yards = columns[9].text.strip()
        team_a_turnover = columns[10].text.strip()
        team_b_yards = columns[11].text.strip()
        team_b_turnover = columns[12].text.strip()
        
        boxscore_link = columns[6].find('a')['href'] if columns[6].find('a') else ''
        boxscore_link = f"https://www.pro-football-reference.com{boxscore_link}" 

        i_gi, i_s, i_sc, i_po, i_pd, i_pr, i_pk, i_pap, i_paru, i_parc, i_pad = games_info(boxscore_link)
        starters_list.append(i_s)
        snap_counts_list.append(i_sc)
        player_offense_list.append(i_po)
        player_defense_list.append(i_pd)
        player_returns_list.append(i_pr)
        player_kicking_list.append(i_pk)
        player_advance_passing_list.append(i_pap)
        player_advance_rushing_list.append(i_paru)
        player_advance_receiving_list.append(i_parc)
        player_advance_defense_list.append(i_pad)
        
        # Append to list of games
        games_data.append({**{
            'season':season,
            'week': week,
            'week_day': week_day,
            'event_date': event_date,
            'game_time': game_time,
            'team_a': team_a,
            'team_b': team_b,
            'location': location,
            'team_a_score': team_a_score,
            'team_b_score': team_b_score,
            'team_a_yards': team_a_yards,
            'team_a_turnover': team_a_turnover,
            'team_b_yards': team_b_yards,
            'team_b_turnover': team_b_turnover,
            'boxscore_link': boxscore_link
        }, **i_gi})

        # Sleep for a random time between 3.5 to 5.5 seconds to avoid overwhelming the server
        time.sleep(random.uniform(3.5, 5.5))

    # Convert the data into a DataFrame
    df_games = pd.DataFrame(games_data)

    return df_games, starters_list, snap_counts_list, player_offense_list, player_defense_list, player_returns_list, player_kicking_list, player_advance_passing_list, player_advance_rushing_list, player_advance_receiving_list, player_advance_defense_list


# url = f"https://www.pro-football-reference.com/years/{year}/games.htm"
