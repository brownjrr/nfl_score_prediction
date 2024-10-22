# NFL Score Prediction

Files Required To Run Makefile

- /data/play_by_play.csv
- /data/players.csv
- /data/game_starters_all.csv
- /data/injuries_all.csv
- /data/player_defense_all.csv
- /data/player_kicking_all.csv
- /data/player_offense_all.csv
- /data/player_returns_all.csv
- /data/snap_counts_all.csv
- /data/raw_game_level_coach_data.csv
- /data/player_positions.csv
- /data/positions.csv
- /data/rosters.csv
- /data/teams.csv
- /data/coaches.csv
- /data/games_info_all.csv
- /data/game_starters_all.csv
- /data/team_dict.csv
- /data/team_dict_short.csv
- /data/position_alias.csv
- /data/custom_weights.csv

___
## How to create play_by_play.csv
In order to create the `play_by_play.csv` table, you must run `game_data_scraping.py`. This script visits the website for each NFL game in our study, extracts the HTML and creates `play_by_play.csv`. This is a long running script!

## How to create players.csv
In order to create the `players.csv` table, you must run the `PlayerDataCollection.ipynb` notebook. This is a long running script!

## How to create game_starters_all.csv
In order to create the 'game_starter_all.csv' table, you must run the 'all_games_scraping_with_BeautifulSoup.ipynb' notebook. This is a long running script!

## How to create injuries_all.csv
In order to create the 'injuries_all.csv' table, you must run the 'injuries_scraping_with_BeautifulSoup.py'.

## How to create player_defense_all.csv
In order to create the 'player_defense_all.csv' table, you must run the 'all_games_scraping_with_BeautifulSoup.ipynb' notebook. This is a long running script!

## How to create player_kicking_all.csv
In order to create the 'player_kicking_all.csv' table, you must run the 'all_games_scraping_with_BeautifulSoup.ipynb' notebook. This is a long running script!

## How to create player_offense_all.csv
In order to create the 'player_offense_all.csv' table, you must run the 'all_games_scraping_with_BeautifulSoup.ipynb' notebook. This is a long running script!

## How to create player_returns_all.csv
In order to create the 'player_returns_all.csv' table, you must run the 'all_games_scraping_with_BeautifulSoup.ipynb' notebook. This is a long running script!

## How to create snap_counts_all.csv
In order to create the 'snap_counts_all.csv' table, you must run the 'all_games_scraping_with_BeautifulSoup.ipynb' notebook. This is a long running script!

## How to create raw_game_level_coach_data.csv
In order to create `raw_game_level_coach_data.csv`, you must run `game_data_scraping.py`. This is a long running script!

## How to create player_positions.csv
In order to create `player_positions.csv`, you must run `create_player_position.py`.

## How to create rosters.csv
In order to create `rosters.csv`, you must run `RosterDataCollection.ipynb`.

## How to create teams.csv
This is created through the file through `gather/nfl_scrape_teams.py`

## How to create coaches.csv
In order to create the 'coaches.csv' table, you must run the 'coaches_scraping_with_BeautifulSoup.py'.

## How to create games_info_all.csv
In order to create the 'games_info_all.csv' table, you must run the 'all_games_scraping_with_BeautifulSoup.ipynb' notebook. This is a long running script!

## How to create game_starters_all.csv
In order to create the 'game_starters_all.csv' table, you must run the 'all_games_scraping_with_BeautifulSoup.ipynb' notebook. This is a long running script!

## team_dict.json,  team_dict_short.json, and position_alias.json
These are dictionaries with team information, and position information, created manually.

## How to create custom_weights.csv
[Enter Info Here]
