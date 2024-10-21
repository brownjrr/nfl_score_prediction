#%%
import pandas as pd

# Model libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer

import numpy as np

# Bring in our constants
#import constants

FEATURE_SELECTION = [
    'week_ind', 'day_int',
    'attendance', 'roof_type',
    'humidity_pct', 'wind_speed',
    'temperature', 'over_under_value',
    'spread_value', 'spread_home_away',
    'coach_rating', 'coach_rating_opp',
    'home_strength', 'opp_strength',
    'team_rating', 'team_rating_opp'
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

def tts_split_data(df: pd.DataFrame,
                   test_year: int=2023,
                   features: list=FEATURE_SELECTION):
    X_train, X_test, y_train, y_test = tts_prep(df, test_year=test_year)
    key_cols = ['game_id', 'boxscore_stub',]

    X_train_keys = X_train[key_cols]
    X_features_train = X_train[features]
    X_test_keys = X_test[key_cols]
    X_features_test = X_test[features]

    return (
        X_features_train,
        X_features_test,
        y_train,
        y_test,
        X_train_keys,
        X_test_keys
    )
    

def baseline_rfr(model_df, random_state=42):
    feature_list = [
    'week_ind', 'day_int',
    'attendance', 'roof_type_int',
    'humidity_pct', 'wind_speed',
    'temperature', 'over_under_value',
    'spread_value', 'spread_home_away',
    'coach_rating', 'coach_rating_opp',
    'home_strength', 'opp_strength',
    'team_rating', 'team_rating_opp'
    ]
    
    X_train,X_test, y_train, y_test, x_train_keys, x_test_keys = tts_split_data(
        model_df,
        test_year=2023,
        features=feature_list)

    rfr = RandomForestRegressor(random_state=random_state).fit(X_train, y_train)
    print(f'Training results: {rfr.score(X_train, y_train):.3f}')
    
    y_pred = rfr.predict(X_test)
    rfr_mae = mean_absolute_error(y_test, y_pred)
    rfr_rmse = root_mean_squared_error(y_test, y_pred)
    print(f"Base random forest model MAE {rfr_mae:.3f}")
    print(f"Base random forest model RMSE {rfr_rmse:.3f}")
    return None


def model_pipeline_prediction(model_regressor, single_target: bool=False):
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
    base_regressor = model_regressor
    multi_target_regressor = MultiOutputRegressor(base_regressor)

    if single_target:
        estimator = base_regressor
    else:
        estimator = multi_target_regressor
    pipe = Pipeline(steps=[
        ("preprocessor", preprocess),
        ("regressor", estimator)
    ])

    return pipe


def rfr_hyperparameter_tuning(x_train, y_train, single_output_model: bool=False):
    """
    Takes in part of the pipe process and returns the best fit hyperparameters
    
    Args:
    ----
        - x_train: the training data
        - y_train: the training target
        - single_output_model: Determines whether or not to run a single model output
        - returns: GridSearchCV object that is best fit to the data
    """

    pipe = model_pipeline_prediction(RandomForestRegressor(random_state=42),
                                     single_target=single_output_model)
    
    if single_output_model:
        param_grid = {
            'regressor__n_estimators': [10, 100, 1000],
            'regressor__max_depth': [10, 25, 50, None],
            'regressor__min_samples_split': [2, 5, 10],
            'preprocessor__num__imputer__strategy': ['mean', 'median'],
        }
    else:
        param_grid = {
            'regressor__estimator__n_estimators': [10, 100, 1000],
            'regressor__estimator__max_depth': [10, 25, 50, None],
            'regressor__estimator__min_samples_split': [2, 5, 10],
            'preprocessor__num__imputer__strategy': ['mean', 'median'],
        }

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid = param_grid,
        cv = 5,
        verbose = 0
    )

    return grid_search.fit(x_train, y_train)


def gb_hyperparameter_tuning(x_train, y_train):
    """
    Takes in part of the pipe process and returns the best fit hyperparameters
    for a gradient boosted model
    
    Args:
    ----
        - pipe: part of the sklearn pipeline
        - x_train: the training data
        - y_train: the training target
        - returns: GridSearchCV object that is best fit to the data
    """

    pipe = model_pipeline_prediction(GradientBoostingRegressor(random_state=42))

    param_grid = {
        'regressor__estimator__learning_rate': [0.025, 0.075,0.2],
        'regressor__estimator__n_estimators': [100, 1000],
        'regressor__estimator__subsample': [0.5, 1.0],
        'preprocessor__num__imputer__strategy': ['mean', 'median']
    }

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid = param_grid,
        cv = 5,
        verbose = 0
    )

    return grid_search.fit(x_train, y_train)


def poisson_hyperparameter_tuning(x_train, y_train):
    """
    Takes in part of the pipe process and returns the best fit hyperparameters
    for a poisson regression model
    
    Args:
    ----
        - pipe: part of the sklearn pipeline
        - x_train: the training data
        - y_train: the training target
        - returns: GridSearchCV object that is best fit to the data
    """

    pipe = model_pipeline_prediction(PoissonRegressor())

    param_grid = {
        'regressor__estimator__alpha': [0.0, 0.2, 1.0, 10.0],
        'regressor__estimator__solver': ['lbfgs', 'newton-cholesky'],
        'regressor__estimator__max_iter': [100, 200, 300],
        'preprocessor__num__imputer__strategy': ['mean', 'median']
    }

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid = param_grid,
        cv = 5,
        verbose = 0
    )

    return grid_search.fit(x_train, y_train)


def train_random_forest(model_df: pd.DataFrame, output_data: bool=True):
    X_train, X_test, y_train, y_test = tts_prep(model_df, test_year=2023)
    
    key_cols = ['game_id', 'boxscore_stub',]

    X_train_keys = X_train[key_cols]
    X_features_train = X_train[FEATURE_SELECTION]
    X_test_keys = X_test[key_cols]
    X_features_test = X_test[FEATURE_SELECTION]

    # Expose X_features_test for SHAP
    if output_data:
        X_features_train.to_csv('data/intermediate/model_rfr__x_train.csv', index=False)
        X_features_test.to_csv('data/intermediate/model_rfr__x_test.csv', index=False)

    #pipe = random_forest_pipeline_prediction(model_type)

    # Train our random forest
    print("Tuning Random Forest model...")
    best_model = rfr_hyperparameter_tuning(X_features_train, y_train)
    print("Tuning complete")
    print("-------------------\n")

    # Get the score on the test set
    y_pred = best_model.predict(X_features_test)
    best_model_mae = mean_absolute_error(y_test, y_pred)
    best_model_rmse = root_mean_squared_error(y_test, y_pred)
    final_model = X_features_test.copy()
    final_model[['score_home_test', 'score_opp_test']] = y_test
    final_model[['score_home_pred', 'score_opp_pred']] = y_pred
    final_model = pd.concat([X_test_keys, final_model], axis=1)
    final_model.to_csv('data/output/random_forest_model_output.csv')
    print(f"Random Forest's best MAE results: {best_model_mae}")
    print(f"Random Forest's best RMSE results: {best_model_rmse}")
    return best_model


def train_gradient_boost(model_df: pd.DataFrame):
    X_train, X_test, y_train, y_test = tts_prep(model_df, test_year=2023)
    
    key_cols = ['game_id', 'boxscore_stub',]

    X_train_keys = X_train[key_cols]
    X_features_train = X_train[FEATURE_SELECTION]
    X_test_keys = X_test[key_cols]
    X_features_test = X_test[FEATURE_SELECTION]

    #pipe = random_forest_pipeline_prediction(model_type)

    # Train our random forest
    print("Tuning Gradient Boosted model...")
    best_model = gb_hyperparameter_tuning(X_features_train, y_train)
    print("Tuning complete")
    print("-------------------\n")

    # Get the score on the test set
    y_pred = best_model.predict(X_features_test)
    best_model_mae = mean_absolute_error(y_test, y_pred)
    best_model_rmse = root_mean_squared_error(y_test, y_pred)
    final_model = X_features_test.copy()
    final_model[['score_home_test', 'score_opp_test']] = y_test
    final_model[['score_home_pred', 'score_opp_pred']] = y_pred
    final_model = pd.concat([X_test_keys, final_model], axis=1)
    final_model.to_csv('data/output/gradient_boosted_model_output.csv')
    print(f"Gradient Boosted's best MAE results: {best_model_mae}")
    print(f"Gradient Boosted's best RMSE results: {best_model_rmse}")
    return best_model


def single_output_random_forest(
        model_df: pd.DataFrame,
        output_data: bool=False
        ) -> None:
    
    X_train, X_test, y_train, y_test = tts_prep(model_df, test_year=2023)
    y_train_home = y_train.iloc[:, 0]
    y_train_opp = y_train.iloc[:, 1]
    y_test_home = y_test.iloc[:, 0]
    y_test_opp = y_test.iloc[:, 1]

    
    key_cols = ['game_id', 'boxscore_stub',]

    X_train_keys = X_train[key_cols]
    X_features_train = X_train[FEATURE_SELECTION]
    X_test_keys = X_test[key_cols]
    X_features_test = X_test[FEATURE_SELECTION]

    #pipe = random_forest_pipeline_prediction(model_type)

    # Train our random forest
    print("Tuning Single Output Random Forest model...")
    home_team_model = rfr_hyperparameter_tuning(
        X_features_train,
        y_train_home,
        single_output_model=True
    )
    print("... now training the opp_team model ...")
    opp_team_model = rfr_hyperparameter_tuning(
        X_features_train,
        y_train_opp,
        single_output_model=True
    )
    print("Tuning complete")
    print("-------------------\n")

    # Get the score on the test set
    y_pred_home = home_team_model.predict(X_features_test)
    y_pred_opp = opp_team_model.predict(X_features_test)
    best_home_model_mae = mean_absolute_error(y_test_home, y_pred_home)
    best_opp_model_mae = mean_absolute_error(y_test_opp, y_pred_opp)
    best_home_model_rmse = root_mean_squared_error(y_test_home, y_pred_home)
    best_opp_model_rmse = root_mean_squared_error(y_test_opp, y_pred_opp)

    final_model = X_features_test.copy() 
    final_model['score_home_test'] = y_test_home
    final_model['score_opp_test'] = y_test_opp
    final_model['score_home_pred'] = y_pred_home
    final_model['score_opp_pred'] = y_pred_opp
    final_model = pd.concat([X_test_keys, final_model], axis=1)
    final_model.to_csv('data/output/single_random_forest_model_output.csv')
    print(f"Random Forest's best MAE results: {best_home_model_mae}")
    print(f"Random Forest's best RMSE results: {best_home_model_rmse}")
    print(f"Random Forest's best MAE results: {best_opp_model_mae}")
    print(f"Random Forest's best RMSE results: {best_opp_model_rmse}")
    return home_team_model


def single_model_for_shap(model_df: pd.DataFrame,
                          n_estimators: int,
                          max_depth: int,
                          min_samples_split: int,
                          input_strategy: str
                          ) -> None:
    
    X_train, X_test, y_train, y_test, _, _= tts_split_data(model_df, test_year=2023)
    
    key_cols = ['game_id', 'boxscore_stub',]

    # Apply preprocess steps manually
    # Change root_type to be the int version
    NEW_FEATURES = [f for f in FEATURE_SELECTION if f != "roof_type"]
    NEW_FEATURES.append("roof_type_int")
    numeric_features=NEW_FEATURES

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
    rfr.fit(X_transformed, y_train)
    
    save_train = pd.DataFrame(X_transformed)
    save_train.to_csv("data/output/model_rfr__x_train_transformed.csv")
    return rfr

    
def train_poisson_model(model_df: pd.DataFrame, output_data: bool=True):
    X_train, X_test, y_train, y_test = tts_prep(model_df, test_year=2023)
    
    key_cols = ['game_id', 'boxscore_stub',]

    X_train_keys = X_train[key_cols]
    X_features_train = X_train[FEATURE_SELECTION]
    X_test_keys = X_test[key_cols]
    X_features_test = X_test[FEATURE_SELECTION]

    # Expose X_features_test for SHAP
    if output_data:
        X_features_train.to_csv('data/intermediate/model_poisson__x_train.csv', index=False)
        X_features_test.to_csv('data/intermediate/model_poisson__x_test.csv', index=False)

    #pipe = random_forest_pipeline_prediction(model_type)

    # Train our random forest
    print("Tuning Random Forest model...")
    best_model = poisson_hyperparameter_tuning(X_features_train, y_train)
    print("Tuning complete")
    print("-------------------\n")

    # Get the score on the test set
    y_pred = best_model.predict(X_features_test)
    best_model_mae = mean_absolute_error(y_test, y_pred)
    best_model_rmse = root_mean_squared_error(y_test, y_pred)
    final_model = X_features_test.copy()
    final_model[['score_home_test', 'score_opp_test']] = y_test
    final_model[['score_home_pred', 'score_opp_pred']] = y_pred
    final_model = pd.concat([X_test_keys, final_model], axis=1)
    final_model.to_csv('data/output/poisson_model_output.csv')
    print(f"Poisson Regression's best MAE results: {best_model_mae}")
    print(f"Poisson Regression's best RMSE results: {best_model_rmse}")
    return best_model

def save_pickle_model(file_name: str, model) -> None:
    """
    Saves our model to the '/data/model/' location

    ARGS:
    ----
        - file_name: should just the name of the file to save
            - Can also accept full path
        - model: trained model object (from pipeline)
    """

    if "/" in file_name:
        print(f"Reverting to full path. Saving to {file_name}")
        file_loc = file_name
    else:
        file_loc = 'data/model/' + file_name


    if file_loc[-3:] != "pkl":
        print("File name does not contain pkl extension. Adding it.")
        file_loc += ".pkl"

    with open(file_loc, 'wb') as f:
        pickle.dump(model, f)

    return None

#%%
if __name__ == '__main__':
    import pickle
    game_df = model_data_read('data/intermediate/games_df.csv')
    # Filter to match our player rating model
    game_df = game_df.loc[game_df['season'] >= 2015]
    game_match_df = add_matchup_rank(game_df, 'data/intermediate/game_rank_matchup.csv')
    
    #original_model = baseline_rfr(game_match_df)
    #best_tuned_random_forest = train_random_forest(model_df=game_match_df)
    #best_tuned_gradient_boost = train_gradient_boost(model_df=game_match_df)
    #best_home_model = single_output_random_forest(model_df=game_match_df, output_data=True)
    best_poisson_model = train_poisson_model(model_df=game_match_df, output_data=True)
    

    # Save our models
    #save_pickle_model(file_name="random_forest_model.pkl",
    #                  model=best_tuned_random_forest)

    #save_pickle_model(file_name="gradient_boosted_model.pkl",
    #                  model=best_tuned_gradient_boost)
    
    #save_pickle_model(file_name="single_target_random_forest_model.pkl",
    #                  model=best_home_model)    

    save_pickle_model(file_name="poisson_regression_model.pkl",
                      model=best_poisson_model)
# %%
