import pandas as pd
import json

# Find positions without a position category to them
with open('data/position_alias.json', 'r') as file:
    position_alias = json.load(file)

per_game_roster = pd.read_csv('data/game_starters_all.csv')

per_game_roster['position_alias'] = per_game_roster['Position'].map(position_alias.get)
#missing_positions = (per_game_roster['Position']
#                     .loc[per_game_roster['position_alias']
#                          .isna()].unique())

# TODO Enhance missing data with actual position
per_game_roster['position_alias'] = per_game_roster['position_alias'].fillna('UNKNOWN')