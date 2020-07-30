
##Load the Datasets
library(readxl)


TADPOLE_D1_D2_Dict <- read.csv("C:/Users/jtame/Dropbox (Personal)/Documents/FRESACAD/TADPOLE/TADPOLE/TADPOLE_D1_D2_Dict.csv", na.strings=c("NA",-4,"-4.0",""," "))
TADPOLE_D1_D2 <- read.csv("C:/Users/jtame/Dropbox (Personal)/Documents/FRESACAD/TADPOLE/TADPOLE/TADPOLE_D1_D2.csv", na.strings=c("NA",-4,"-4.0",""," "))
TADPOLE_D3 <- read.csv("C:/Users/jtame/Dropbox (Personal)/Documents/FRESACAD/TADPOLE/TADPOLE/TADPOLE_D3.csv", na.strings=c("NA",-4,"-4.0",""," ","NaN"))
TADPOLE_D4_corr <- read.csv("~/GitHub/R_Python_interoperability/data/TADPOLE_D4_corr.csv")

submissionTemplate <- read_excel("TADPOLE_Simple_Submission_TeamName.xlsx")

submissionTemplate$`Forecast Date` <- as.Date(paste(submissionTemplate$`Forecast Date`,"-01",sep=""))
submissionTemplate$`CN relative probability` <- as.numeric(nrow(submissionTemplate))
submissionTemplate$`MCI relative probability` <-  as.numeric(nrow(submissionTemplate))
submissionTemplate$`AD relative probability` <-  as.numeric(nrow(submissionTemplate))

TADPOLE_D1_D2$EXAMDATE <- as.Date(TADPOLE_D1_D2$EXAMDATE)
TADPOLE_D3$EXAMDATE <- as.Date(TADPOLE_D3$EXAMDATE)

#DataSplit


TrainingSet <- subset(TADPOLE_D1_D2,D1==1)
D2TesingSet <- subset(TADPOLE_D1_D2,D2==1)

#DataProcessing

source('~/GitHub/TADPOLE/dataPreprocessing.R')
source('~/GitHub/TADPOLE/TADPOLE_Train.R')
source('~/GitHub/TADPOLE/predictCognitiveStatus.R')
source('~/GitHub/TADPOLE/FiveYearForecast.R')



dataTadpole <- dataTADPOLEPreprocesing(TrainingSet,D2TesingSet,TADPOLE_D1_D2_Dict,MinVisit=36,colImputeThreshold=0.25,rowImputeThreshold=0.25)

save(dataTadpole,file="D2DataFrames.RDATA")
load(file="D2DataFrames.RDATA")

CognitiveClassModels <- TrainTadpoleClassModels(dataTadpole$AdjustedTrainFrame,
                        predictors=c("AGE","PTGENDER",colnames(dataTadpole$AdjustedTrainFrame)[-c(1:22)]),
                        numberOfRandomSamples=25,
                        delta=TRUE,
                        MLMethod=BSWiMS.model,
                        NumberofRepeats = 1)

save(CognitiveClassModels,file="CognitiveClassModels_25.RDATA")
load(file="CognitiveClassModels_10b.RDATA")


predictADNI <- forecastCognitiveStatus(CognitiveClassModels,dataTadpole$testingFrame)

forecast <- FiveYearForeCast(predictADNI,Subject_datestoPredict=submissionTemplate)
write.csv(forecast,file="forecastJTP.csv")

#D3 Cross sectional

## First Remove D2 subjects from Training Set
D3IDS <- TADPOLE_D3$RID
D3TrainingSet <- TrainingSet[!(TrainingSet$RID %in% D3IDS),]

## Conditioning the datasets
dataTadpoleD3 <- dataTADPOLEPreprocesing(D3TrainingSet,TADPOLE_D3,TADPOLE_D1_D2_Dict,MinVisit=18,colImputeThreshold=0.15,rowImputeThreshold=0.25)
save(dataTadpoleD3,file="D3DataFrames.RDATA")
load(file="D3DataFrames.RDATA")

## Build the predictive models
D3CognitiveClassModels <- TrainTadpoleClassModels(dataTadpoleD3$AdjustedTrainFrame,
                                                predictors=c("AGE","PTGENDER",colnames(dataTadpoleD3$AdjustedTrainFrame)[-c(1:22)]),
                                                numberOfRandomSamples=25,
                                                MLMethod=BSWiMS.model,
                                                NumberofRepeats = 1)
save(D3CognitiveClassModels,file="D3CognitiveClassModels_25.RDATA")
#load(file="D3CognitiveClassModels_5.RDATA")


## Predict all the testing set
predictADNID3 <- forecastCognitiveStatus(D3CognitiveClassModels,dataTadpoleD3$testingFrame)

## Forecast the testing set
forecastD3 <- FiveYearForeCast(predictADNID3,Subject_datestoPredict=submissionTemplate)
write.csv(forecastD3,file="forecastJTPD3.csv")

