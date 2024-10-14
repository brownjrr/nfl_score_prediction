import pickle
import seaborn
import pandas as pd
import shap

def read_model(file_name: str):
    if "/" in file_name:
        print(f"Reverting to full path. Saving to {file_name}")
        file_loc = file_name
    else:
        file_loc = 'data/model/' + file_name

    with open(file_loc, 'rb') as f:
        reg = pickle.load(f)

    return reg

rfr = read_model('random_forest_model.pkl')
x_test = pd.read_csv('data/intermediate/x_test.csv')
explainer = shap.TreeExplainer()
shap_values = explainer.shap_values(x_test)
shap.summary_plot(shap_values)