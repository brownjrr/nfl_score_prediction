#%%
import pandas as pd

# Model libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer

# Bring in our constants
#import constants

FEATURE_SELECTION = [
    'week_ind', 'day_int', 'OT',
    'away', 'attendance', 'roof_type',
    'humidity_pct', 'wind_speed',
    'temperature', 'duration',
    'coach_index', 'coach_index_opp',
    'home_strength', 'opp_strength'
    ]

#%%
# Read in our intermediate data
def model_data_read(file_loc: str) -> pd.DataFrame:
    """
    First step to load in the data
    Can be part of a pipeline or read in data here
    Test file: 'data/intermediate/games_df.csv'
    """
    df = pd.read_csv(file_loc)
    if 'game_id' not in df.columns:
        df['game_id'] = df[
            ['home_team_id',
            'opp_team_id',
            'event_date'
            ]].apply(
                lambda x: "_".join(x.values.astype(str)),
                axis=1
                )
    return df

def add_matchup_rank(df: pd.DataFrame, file_loc: str) -> pd.DataFrame:
    """
    Add the columns for the matchup score to our main dataframe
    """
    matchup_rank = pd.read_csv(file_loc)
    return df.merge(
        matchup_rank,
        on='game_id'
    )


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
    print(f'Training results: {clf.score(X_features_train, y_train):.3f}')
    score = clf.score(X_features_test, y_test)
    
    return (score)

def random_forest_pipeline_prediction():
    """
    Feeds in our  main model data into a pipeline for training
    and hyperparameter tuning, and the final output
    """
    numeric_features=[item for item in 
                      FEATURE_SELECTION if item != 'roof_type']
    categoric_features = ['roof_type']

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer()),
               ("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(
            strategy="constant",
            fill_value="missing")),
            ("onehot",
             OneHotEncoder(handle_unknown="ignore")
             )
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categoric_features)
        ]
    )


    # Define our regressor
    base_regressor = RandomForestRegressor(random_state=42)
    multi_target_regressor = MultiOutputRegressor(base_regressor)
    pipe = Pipeline(steps=[
        ("preprocessor", preprocess),
        ("regressor", multi_target_regressor)
    ])

    return pipe


def rfr_hyperparameter_tuning(pipe, x_train, y_train):
    """
    Takes in part of the pipe process and returns the best fit hyperparameters
    
    Args:
    ----
        - pipe: part of the sklearn pipeline
        - x_train: the training data
        - y_train: the training target
        - returns: GridSearchCV object that is best fit to the data
    """
    param_grid = {
        'regressor__estimator__n_estimators': [10, 100, 1000],
        'preprocessor__num__imputer__strategy': ['mean', 'median']
    }

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid = param_grid,
        cv = 5,
        verbose = 2
    )

    return grid_search.fit(x_train, y_train)

def train_random_forest(model_df: pd.DataFrame):
    X_train, X_test, y_train, y_test = tts_prep(model_df, test_year=2023)
    
    X_features_train = X_train[FEATURE_SELECTION]
    X_features_test = X_test[FEATURE_SELECTION]

    pipe = random_forest_pipeline_prediction()

    # Train our random forest
    print("Tuning model...")
    best_model = rfr_hyperparameter_tuning(pipe, X_features_train, y_train)
    print("Tuning complete")

    # Get the score on the test set
    y_pred = best_model.predict(X_features_test)
    best_model_mae = mean_absolute_error(y_test, y_pred)
    best_model_mape = mean_absolute_percentage_error(y_test, y_pred)
    final_model = X_features_test.copy()
    final_model[['score_home_test', 'score_opp_test']] = y_test
    final_model[['score_home_pred', 'score_opp_pred']] = y_pred
    final_model.to_csv('data/output/random_forest_model_output.csv')
    print(f"Random Forest's best MAE results: {best_model_mae}")
    print(f"Random Forest's best MAPE results: {best_model_mape}")
    return best_model

#%%
if __name__ == '__main__':
    game_df = model_data_read('data/intermediate/games_df.csv')
    game_match_df = add_matchup_rank(game_df, 'data/intermediate/game_rank_matchup.csv')
    #base_score = baseline_rfr(game_match_df)
    #print(f'Without any team player indicators, \
    #      the model gives us a training score of {base_score}')
    
    best_tuned_random_forest = train_random_forest(model_df=game_match_df)
# %%