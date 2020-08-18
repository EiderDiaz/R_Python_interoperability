import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri

from tadpole_algorithms.models.tadpole_model import TadpoleModel

import logging

from datetime import datetime
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)

class BenchmarkModel(TadpoleModel):
    
    def R_df_to_python_df(self,R_df,csvname=""):
        #func to convert a R df into python and if a csvname is specified can be saved
        pandas2ri.activate()
        d_from_r_df = pd.DataFrame()
        with localconverter(robjects.default_converter + pandas2ri.converter):
            pd_from_r_df = robjects.conversion.rpy2py(R_df)
        if csvname :
            pd_from_r_df.to_csv(csvname)
        return pd_from_r_df

    def Python_df_to_R_df(self,Python_df):
        #func to convert a Python df into R DF
        pandas2ri.activate()
        r_from_pd_df = robjects.DataFrame({})
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_from_pd_df = robjects.conversion.py2rpy(Python_df)
        return r_from_pd_df
    
    def loading_data(self, path_D1_D2_Dict, path_D1_D2, path_D3, path_simple_submission):
        load_data = ""
        with open('load_data.txt', 'r') as file:
            load_data = file.read()
        robjects.r(load_data)

        f_load_data = robjects.r['load_data']
        output = f_load_data(path_D1_D2_Dict, path_D1_D2, path_D3, path_simple_submission)
        
        return output[0], output[1], output[2], output[3]
    
    def split(self, TADPOLE_D1_D2, TADPOLE_D3):
        train_test_split = ""
        with open('train_test_split.txt', 'r') as file:
            train_test_split = file.read()
        robjects.r(train_test_split)

        f_train_test_split = robjects.r['train_test_split']
        output = f_train_test_split(TADPOLE_D1_D2, TADPOLE_D3)

        return output[0], output[1], output[2]
    
    def data_conditioning_for_D2(self, TrainingSet, D2TesingSet):
        conditioning_D2 = ""
        with open('conditioning_D2.txt', 'r') as file:
            conditioning_D2 = file.read()
        robjects.r(conditioning_D2)

        f_conditioning_D2 = robjects.r['conditioning_D2']
        output = f_conditioning_D2(TrainingSet, D2TesingSet)

        return output[0], output[1]
    
    def data_preprocessing(self,TrainingSet,D2TesingSet,TADPOLE_D1_D2_Dict,minvisit,colimputedthresh,rowimputedthresh,includeid):
        dataTADPOLEPreprocesing = ""
        with open('dataTADPOLEPreprocesing.txt', 'r') as file:
            dataTADPOLEPreprocesing = file.read()
        robjects.r(dataTADPOLEPreprocesing)

        f_dataTADPOLEPreprocesing = robjects.r['dataTADPOLEPreprocesing']
        dataTadpole = f_dataTADPOLEPreprocesing(TrainingSet,
                                                D2TesingSet,
                                                TADPOLE_D1_D2_Dict,
                                                MinVisit=minvisit,
                                                colImputeThreshold=colimputedthresh,
                                                rowImputeThreshold=rowimputedthresh,
                                                includeID=includeid)
        return dataTadpole
    
    def Train_25_Models_D2_subjects(self, dataTadpole, method, b_delta):
        # Preparation R to python data
        col_names = list(dataTadpole[0].columns)
        vec_predictors = np.array("AGE", "PTGENDE", col_names[-22:])
        vec_predictors = robjects.vectors.StrVector(vec_predictors)
        
        TrainTadpoleClassModels = ""
        with open('TrainTadpoleClassModels.txt', 'r') as file:
            TrainTadpoleClassModels = file.read()
        robjects.r(TrainTadpoleClassModels)

        f_TrainTadpoleClassModels = robjects.r['TrainTadpoleClassModels']
        CognitiveClassModels = f_TrainTadpoleClassModels(dataTadpole[0],
                                predictors=vec_predictors,
                                numberOfRandomSamples=25,
                                delta=b_delta,
                                MLMethod=method,
                                NumberofRepeats = 1)
        
        return CognitiveClassModels
    
    def predict_D2_subjects(self, CognitiveClassModels, dataTadpole):
        forecastCognitiveStatus = ""
        with open('forecastCognitiveStatus.txt', 'r') as file:
            forecastCognitiveStatus = file.read()
        robjects.r(forecastCognitiveStatus)

        f_forecastCognitiveStatus = robjects.r['forecastCognitiveStatus']
        predictADNI = f_forecastCognitiveStatus(CognitiveClassModels,dataTadpole[1])
        
        return predictADNI
    
    def get_D3_data(self, TrainingSet):
        original_data_D3_train = ""
        with open('original_data_D3_train.txt', 'r') as file:
            original_data_D3_train = file.read()
        robjects.r(original_data_D3_train)

        f_original_data_D3_train = robjects.r['original_data_D3_train']
        dataTadpole = f_original_data_D3_train(TrainingSet)
        
        return dataTadpole
    
    def Train_50_Models_D1_data(self, dataTadpole, model):
        # Preparation R to python data
        col_names = list(dataTadpole[0].columns)
        vec_predictors = np.array("AGE", "PTGENDE", col_names[-22:])
        vec_predictors = robjects.vectors.StrVector(vec_predictors)
        
        TrainTadpoleRegresionModels = ""
        with open('TrainTadpoleRegresionModels.txt', 'r') as file:
            TrainTadpoleRegresionModels = file.read()
        robjects.r(TrainTadpoleRegresionModels)

        f_TrainTadpoleRegresionModels = robjects.r['TrainTadpoleRegresionModels']
        CognitiveRegresModels = f_TrainTadpoleRegresionModels(dataTadpole[0],
                                                        predictors=vec_predictors,
                                                        numberOfRandomSamples=50,
                                                        MLMethod=model,
                                                        NumberofRepeats = 1)
        return CognitiveRegresModels
    
    def ventr_adas_data_preparation(self, dataTadpole):
        ventr_adas_preparation = ""
        with open('ventr_adas_preparation.txt', 'r') as file:
            ventr_adas_preparation = file.read()
        robjects.r(ventr_adas_preparation)

        f_ventr_adas_preparation = robjects.r['ventr_adas_preparation']
        ltptf = f_ventr_adas_preparation(dataTadpole)
        
        return ltptf
    
    def five_year_for_cast(self, predictADNI, ltptf, CognitiveRegresModels, submissionTemplate):
        FiveYearForeCast = ""
        with open('FiveYearForeCast.txt', 'r') as file:
            FiveYearForeCast = file.read()
        robjects.r(FiveYearForeCast)

        f_FiveYearForeCast = robjects.r['FiveYearForeCast']
        forecast = f_FiveYearForeCast(predictADNI,
                                      testDataset=ltptf,
                                      ADAS_Ventricle_Models=CognitiveRegresModels,
                                      Subject_datestoPredict=submissionTemplate)
    
        return forecast
    
    def remove_D2_from_train(self, TADPOLE_D3, TrainingSet):
        remove_D2_train = ""
        with open('remove_D2_train.txt', 'r') as file:
            remove_D2_train = file.read()
        robjects.r(remove_D2_train)

        f_remove_D2_train = robjects.r['remove_D2_train']
        output = f_remove_D2_train(TADPOLE_D3, TrainingSet)

        return output[0], output[1]
    
    def train_D3_corr_vdas_ventr(self, D3TrainingSet):
        trainD3CorrelationAdasVentr = ""
        with open('trainD3CorrelationAdasVentr.txt', 'r') as file:
            trainD3CorrelationAdasVentr = file.read()
        robjects.r(trainD3CorrelationAdasVentr)

        f_trainD3CorrelationAdasVentr = robjects.r['trainD3CorrelationAdasVentr']
        dataTadpoleD3 = f_trainD3CorrelationAdasVentr(D3TrainingSet)
        
        return dataTadpoleD3
    
    def last_time_D3(self, D3TrainingSet):
        last_time_D3_point = ""
        with open('last_time_D3_point.txt', 'r') as file:
            last_time_D3_point = file.read()
        robjects.r(last_time_D3_point)

        f_last_time_D3_point = robjects.r['last_time_D3_point']
        ltptf = f_last_time_D3_point(D3TrainingSet)
        
        return ltptf
