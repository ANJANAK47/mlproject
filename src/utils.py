import os
import sys
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            dill.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        best_params = {}

        for name, model in models.items():
            grid = param.get(name, None)

            if grid and len(grid) > 0:
                gs = GridSearchCV(model, grid, cv=3, n_jobs=-1)
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)
                best_params[name] = gs.best_params_
            else:
                best_params[name] = {}

            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            test_score = r2_score(y_test, y_test_pred)

            report[name] = test_score

        return report, best_params
    except Exception as e:
        raise CustomException(e, sys)
