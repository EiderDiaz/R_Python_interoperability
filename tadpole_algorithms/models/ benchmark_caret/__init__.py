# Benchmark entry added after the competition deadline. The entry simply uses the last known value.
# Based on an MATLAB script by Daniel Alexander and Neil Oxtoby.
# ============
# Authors:
#   Razvan Valentin-Marinescu

## Read in the TADPOLE data set and extract a few columns of salient information.
# Script requires that TADPOLE_D1_D2.csv is in the parent directory. Change if
# necessary

import pandas as pd
import numpy as np
import os
import sys

from tadpole_algorithms.models.tadpole_model import TadpoleModel


import logging


logger = logging.getLogger(__name__)
#this import is for use R code into Python
from rpy2 import robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector

class BenchmarkCaret(TadpoleModel):
    

    
    def caret_gmb_modelfitting_R(self):

        caret_gmb_modelfitting = ""
        with open('./R_scripts/caret_gmb', 'r') as file:
        #this file contains magic R scrpits 
            caret_gmb_modelfitting = file.read()
        gmb_model = robjects.r(caret_gmb_modelfitting)
        return gmb_model

    def preprocess_df_R(self,dataframe):
        #this function parse a python dataframe to a R dataframe
        feature_dict = {}
        for colname in dataframe.columns:
            # What happens if we pass the wrong type?
            feature_dict[colname] = robjects.FloatVector(dataframe[colname])
        
        dataframe_R = robjects.DataFrame(feature_dict)
        return dataframe_R
        
        
   
    def modelfitting_R(self, model,formula,dataframe):
        formula_R = robjects.Formula(formula) 
        r_model = robjects.r[model]
        model =  r_model(formula=formula_R, data=dataframe)

        return model

    def predict_R(self,model,test_df):
        r_predict = robjects.r["predict"]
        predictions = r_predict(model, test_df)
        return predictions

    def SVM_fitting_R(self,formula,dataframe):
        e1071 = importr('e1071')
        r_svm = robjects.r["svm"]
        #r_false = robjects.r["FALSE"]
        formula_R = robjects.Formula(formula) 
        model = r_svm(formula=formula_R, data=dataframe, kernel = "linear", cost = 10, scale = 0)
        return model



    



#end R functions
    def preprocess(self, train_df):
        logger.info("Pre-processing")
        train_df = train_df.copy()
        if 'Diagnosis' not in train_df.columns:
            train_df = train_df.replace({'DXCHANGE': {4: 2, 5: 3, 6: 3, 7: 1, 8: 2, 9: 1}})
            train_df = train_df.rename(columns={"DXCHANGE": "Diagnosis"})

        # Sort the dataframe based on age for each subject
        if 'Years_bl' in train_df.columns:
            train_df = train_df.sort_values(by=['RID', 'Years_bl'])

        # Ventricles_ICV = Ventricles/ICV_bl. So make sure ICV_bl is not zero to avoid division by zero
        icv_bl_median = train_df['ICV_bl'].median()
        train_df.loc[train_df['ICV_bl'] == 0, 'ICV_bl'] = icv_bl_median

        if 'Ventricles_ICV' not in train_df.columns:
            train_df["Ventricles_ICV"] = train_df["Ventricles"].values / train_df["ICV_bl"].values

        # Select features
        train_df = train_df[
            ["RID", "Diagnosis", "ADAS13", "Ventricles_ICV", "Ventricles", "ICV_bl"]
        ]

        # Force values to numeric
        train_df = train_df.astype("float64", errors='ignore')

        return train_df

    def train(self, train_df):
        logger.info("No training required")

    def predict(self, test_df):
        logger.info("Predicting")

        # select last row per RID
        test_df = test_df.sort_values(by=['EXAMDATE'])
        test_df = test_df.groupby('RID').tail(1)
        exam_dates = test_df['EXAMDATE']

        test_df = self.preprocess(test_df)
        
        # Select same columns as for traning for testing
        test_df = test_df[["RID", "Diagnosis", "ADAS13", "Ventricles_ICV"]]

        # Default values
        Ventricles_typical = 25000
        Ventricles_broad_50pcMargin = 20000  # +/- (broad 50% confidence interval)
        Ventricles_default_50pcMargin = 1000  # +/- (broad 50% confidence interval)
        ADAS13_typical = 12
        ADAS13_broad_50pcMargin = 10 
        ADAS13_default_50pcMargin = 1
        
        subjects = test_df["RID"].unique()
        diag_probas = np.zeros([len(subjects),3])
        adas_prediction = np.zeros(len(subjects))
        adas_ci = np.zeros(len(subjects))
        ventricles_prediction = np.zeros(len(subjects))
        ventricles_ci = np.zeros(len(subjects))
        for i, subject in enumerate(subjects):
            diag_probas[i, int(test_df.loc[test_df["RID"] == subject, "Diagnosis"].dropna().values.tolist()[-1])-1] = 1
            
            adas_prediction[i] = test_df.loc[test_df["RID"] == subject, "ADAS13"].dropna().values.tolist()[-1]
            if adas_prediction[i] > 0: 
                adas_ci[i] = ADAS13_default_50pcMargin
                adas_ci[i] = ADAS13_default_50pcMargin
            else:
                # Subject has no history of ADAS13 measurement, so we'll take a
                # typical score of 12 with wide confidence interval +/-10.
                adas_prediction[i] = ADAS13_typical 
                adas_ci[i] = ADAS13_broad_50pcMargin
            
            try:
                ventricles_prediction[i] = test_df.loc[test_df["RID"] == subject, "Ventricles_ICV"].dropna().values.tolist()[-1]    
            except IndexError:
                print(test_df.loc[test_df["RID"] == subject, "Ventricles_ICV"].dropna().values.tolist())
                
            if ventricles_prediction[i] > 0: 
                ventricles_ci[i] = Ventricles_default_50pcMargin
            else:
                # Subject has no imaging history, so we'll take a typical
                # ventricles volume of 25000 & wide confidence interval +/-20000
                ventricles_prediction[i] = Ventricles_typical
                ventricles_ci[i] = Ventricles_broad_50pcMargin 
        
        diag_probas_t = diag_probas.T.copy()

        def add_months_to_str_date(strdate, months=1):
            try:
                return (datetime.strptime(strdate, '%Y-%m-%d') + relativedelta(months=months)).strftime('%Y-%m-%d')
            except ValueError:
                return (datetime.strptime(strdate, '%d/%m/%Y') + relativedelta(months=months)).strftime('%d/%m/%Y')

        df = pd.DataFrame.from_dict({
            'RID': subjects,
            'month': 1,
            'Forecast Date': list(map(lambda x: add_months_to_str_date(x, 1), exam_dates.tolist())),
            'CN relative probability': diag_probas_t[0],
            'MCI relative probability': diag_probas_t[1],
            'AD relative probability': diag_probas_t[2],

            'ADAS13': adas_prediction,
            'ADAS13 50% CI lower': adas_prediction - adas_ci, # To do: Set to zero if best-guess less than 1.
            'ADAS13 50% CI upper': adas_prediction + adas_ci,

            'Ventricles_ICV': ventricles_prediction,
            'Ventricles_ICV 50% CI lower': ventricles_prediction - ventricles_ci,
            'Ventricles_ICV 50% CI upper': ventricles_prediction + ventricles_ci,
        })

        # copy each row for each month
        new_df = df.copy()
        for i in range(2, 12 * 10):
            df_copy = df.copy()
            df_copy['month'] = i
            df_copy['Forecast Date'] = df_copy['Forecast Date'].map(lambda x: add_months_to_str_date(x, i - 1))
            new_df = new_df.append(df_copy)

        return new_df
    