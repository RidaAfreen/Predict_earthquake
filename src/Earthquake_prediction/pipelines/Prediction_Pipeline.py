import os
import sys
import pandas as pd
from src.Earthquake_prediction.exception import customexception
from src.Earthquake_prediction.logger import logging
from src.Earthquake_prediction.utils.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            preprocessor_path=os.path.join("Artifacts","preprocessor.pkl")
            model_path=os.path.join("Artifacts","Model.pkl")
            
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            
            scaled_data=preprocessor.transform(features)
            
            pred=model.predict(scaled_data)
            
            return pred
            
            
        
        except Exception as e:
            raise customexception(e,sys)
    
    
    
class CustomData:
    def __init__(self,
                 latitude:float,
                 longitude:float,
                 depth:float,
                 mag:float,
                 hour:float):
        
        self.latitude=latitude
        self.longitude=longitude
        self.depth=depth
        self.mag=mag
        self.hour=hour
            
                
    def get_data_as_dataframe(self):
            try:
                custom_data_input_dict = {
                    'latitude':[self.latitude],
                    'longitude':[self.longitude],
                    'depth':[self.depth],
                    'mag':[self.mag],
                    'hour':[self.hour]
                }
                df = pd.DataFrame(custom_data_input_dict)
                logging.info('Dataframe Gathered')
                return df
            except Exception as e:
                logging.info('Exception Occured in prediction pipeline')
                raise customexception(e,sys)
