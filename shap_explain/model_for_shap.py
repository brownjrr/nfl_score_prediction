import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import root_mean_squared_error
import pickle

FEATURE_SELECTION = [
    'week_ind', 'day_int',
    'attendance',
    'humidity_pct', 'wind_speed',
    'temperature', 'over_under_value',
    'spread_value', 'spread_home_away',
    'coach_rating', 'coach_rating_opp',
    'home_strength', 'opp_strength',
    'team_rating', 'team_rating_opp'
    ]

# IMPORTED FUNCTIONS ------------------
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
    Merging using 'inner' to ensure no NAs
    """
    matchup_rank = pd.read_csv(file_loc)
    return df.merge(
        matchup_rank,
        on='game_id',
        how='inner'
    )


def tts_prep(df: pd.DataFrame, test_year: int=2023):
    y = df.loc[:,['score_team', 'score_opp']]
    X = df.drop(columns=['score_team', 'score_opp'])

    # Split our data manually
    train_mask, test_mask = X['year'] < test_year, X['year'] >= test_year
    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]

    return X_train, X_test, y_train, y_test

def tts_split_data(df: pd.DataFrame, test_year: int=2023):
    X_train, X_test, y_train, y_test = tts_prep(df, test_year=test_year)
    key_cols = ['game_id', 'boxscore_stub',]

    X_train_keys = X_train[key_cols]
    X_features_train = X_train[FEATURE_SELECTION]
    X_test_keys = X_test[key_cols]
    X_features_test = X_test[FEATURE_SELECTION]

    return (
        X_features_train,
        X_features_test,
        y_train,
        y_test,
        X_train_keys,
        X_test_keys
    )
    
# ---- END OF IMPORTED FUNCTIONS

game_df = model_data_read('data/intermediate/games_df.csv')
    # Filter to match our player rating model
game_df = game_df.loc[game_df['season'] >= 2015]
game_match_df = add_matchup_rank(game_df, 'data/intermediate/game_rank_matchup.csv')


def single_model_for_shap(model_df: pd.DataFrame,
                          n_estimators: int,
                          max_depth: int,
                          min_samples_split: int
                          ):
    
    X_train, X_test, y_train, y_test, _, _= tts_split_data(model_df, test_year=2023)
    y_train_home = y_train.iloc[:, 0]
    y_train_opp = y_train.iloc[:, 1]
    y_test_home = y_test.iloc[:, 0]
    y_test_opp = y_test.iloc[:, 1]
    
    key_cols = ['game_id', 'boxscore_stub',]

    # Apply preprocess steps manually
    # Change root_type to be the int version
    numeric_features = FEATURE_SELECTION

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer()),
               ("scaler", StandardScaler())]
    )

    # Excluding the categorical transform
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
            ("num", numeric_transformer, numeric_features)
        ]
    )

    X_transformed = preprocess.fit_transform(X_train)
    rfr = RandomForestRegressor(random_state=42,
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                criterion="squared_error")
    rfr.fit(X_train, y_train_home)
    
    save_train = pd.DataFrame(X_train)
    save_train.to_csv("data/output/model_rfr__x_train.csv", index=False)
    X_test.to_csv('data/output/model_rfr__x_test.csv', index=False)

    # Error of this model
    y_pred_home = rfr.predict(X_test)
    rmse_rfr = root_mean_squared_error(y_test_home, y_pred_home)
    print(f"Error of stripped down model is {rmse_rfr:.3f}")
    return rfr


# Load in an existing model and run the single model
with open("data/model/single_target_random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

best_params = model.best_params_
shap_rfr = single_model_for_shap(game_match_df,
                                    n_estimators=best_params['regressor__n_estimators'],
                                    max_depth=best_params['regressor__max_depth'],
                                    min_samples_split=best_params['regressor__min_samples_split']
                                    )

with open("data/model/rfr_for_shap.pkl", "wb") as f:
    pickle.dump(shap_rfr, f)