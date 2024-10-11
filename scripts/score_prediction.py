#%%
import pandas as pd

# Model libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

# Bring in our constants
#import constants

FEATURE_SELECTION = [
    'week_ind', 'day_int', 'OT',
    'away', 'attendance', 'roof_type_int',
    'humidity_pct', 'wind_speed',
    'temperature', 'duration',
    'coach_index', 'coach_index_opp'
    ]

#%%
# Read in our intermediate data
def model_data_read(file_loc: str) -> pd.DataFrame:
    """
    First step to load in the data
    Can be part of a pipeline or read in data here
    Test file: 'data/intermediate/games_df.csv'
    """
    return pd.read_csv(file_loc)


def tts_prep(df: pd.DataFrame, test_year: int=2023):
    y = df.loc[:,['score_team', 'score_opp']]
    X = df.drop(columns=['score_team', 'score_opp'])

    # Split our data manually
    train_mask, test_mask = X['year'] < test_year, X['year'] >= test_year
    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]

    return X_train, X_test, y_train, y_test


def baseline_rfr(model_df, random_state=42):
    X_train, X_test, y_train, y_test = tts_prep(model_df, test_year=2023)
    
    X_features_train = X_train[FEATURE_SELECTION]
    X_features_test = X_test[FEATURE_SELECTION]

    clf = RandomForestRegressor(random_state=random_state).fit(X_features_train, y_train)
    print(f'Trainng results: {clf.score(X_features_train, y_train):.3f}')
    score = clf.score(X_features_test, y_test)
    
    return (score)
#%%
if __name__ == '__main__':
    game_df = model_data_read('data/intermediate/games_df.csv')
    base_score = baseline_rfr(game_df)
    print(f'Without any team player indicators, \
          the model gives us a training score of {base_score}')