import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.Earthquake_prediction.logger import logging
from src.Earthquake_prediction.exception import customexception

from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error

import joblib

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".joblib":
            joblib.dump(obj, file_path)
        elif ext == ".pkl":
            with open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    except Exception as e:
        raise customexception(e, sys)

    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise customexception(e,sys)
    
def load_object(file_path):
    try:
        ext = os.path.splitext(file_path)[1].lower()
        print(ext)
        if ext == ".pkl":
            with open(file_path, 'rb') as file_obj:
                return pickle.load(file_obj)
        elif ext == ".joblib":
            with open(file_path, 'rb') as file_obj:
                return joblib.load(file_obj)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    except Exception as e:
        logging.info('Exception Occurred in load_object function utils')
        raise customexception(e, sys)

    