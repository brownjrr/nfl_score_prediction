import shap
import numpy as np
from sklearn.multioutput import MultiOutputRegressor

# Helper code for MultiOutputRegressor taken from
# SHAP github --  https://github.com/shap/shap/issues/1104

def build_explainer(model, *args, **kwargs):
    """
    If model is an instance of MultiOutputRegressor, use MultiOutputExplainer abstraction.
    Note: Explainer is automatically converted to TreeExplainer when mode is tree based
    """

    if isinstance(model, MultiOutputRegressor):
        explainer = MultiOutputExplainer(model, *args, **kwargs)
    else:
        explainer = shap.Explainer(model, *args, **kwargs)
    return explainer

class MultiOutputExplainer(shap.Explainer):
    """Abstraction of Explainer for MultiOutputRegressor model
    """
    def __init__(self, model, *args, **kwargs):
        assert isinstance(model, MultiOutputRegressor)
        self.explainers = []
        self.expected_value = []
        for estimator in model.estimators_:
            explainer = shap.Explainer(estimator, *args, **kwargs)
            self.explainers.append(explainer)
            self.expected_value.append(explainer.expected_value)

    def shap_values(self, *args, **kwargs):
        shap_values = []
        for explainer in self.explainers:
            shap_values.append(explainer.shap_values(*args, **kwargs))
        return np.array(shap_values)
    
    def shap_interaction_values(self, *args, **kwargs):
        shap_interaction_values = []
        for explainer in self.explainers:
            shap_interaction_values.append(explainer.shap_interaction_values)
        return np.array(shap_interaction_values)