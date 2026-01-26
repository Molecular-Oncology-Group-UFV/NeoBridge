import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score
import numpy as np
import optuna
from datetime import datetime
import os

def save_optuna_result(model_name, metric, value, best_params,filepath,filename='results/reports/Optuna_Results/optuna_results_general.csv'):

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        'datetime': now,
        'model': model_name,
        'metric': metric,
        'result': value,
        'best_params': str(best_params),
        'dataset' : filepath
    }
    if not os.path.exists(filename):
        df = pd.DataFrame([entry])
        df.to_csv(filename, index=False)
    else:
        df = pd.read_csv(filename)
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
        df.to_csv(filename, index=False)