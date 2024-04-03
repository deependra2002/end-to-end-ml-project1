import sys
from dataclasses import dataclass
from src.utils import save_objact

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
import os

@dataclass
class DataTransformatiomConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformatiomConfig() 
          
    def get_data_transformer_object(self):
        try:
            numerical_columns=["writing_score","reading_score"]
            categorical_columns=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler(with_mean=False)),
                ]  
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_incoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info("numerical columns standard scaling completed")
            
            logging.info("categorical colum encoding completed")
            
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("read train and test data complated")
            logging.info("optaning preprocessor object")
            preprocessing_obj=self.get_data_transformer_object()
            target_column_name="math_score"
            numerical_columns=["writing_score","reading_score"]
            
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            logging.info("applying preprocessing object on traning dataframe")
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
            logging.info("saved preprocessing obj")
            save_objact(
                
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
             )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
            
            

            