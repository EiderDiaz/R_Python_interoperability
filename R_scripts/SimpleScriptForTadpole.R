
##Load the Datasets
library(readxl)


TADPOLE_D1_D2_Dict <- read.csv("C:/Users/jtame/Dropbox (Personal)/Documents/FRESACAD/TADPOLE/TADPOLE/TADPOLE_D1_D2_Dict.csv", na.strings=c("NA",-4,"-4.0",""," "))
TADPOLE_D1_D2 <- read.csv("C:/Users/jtame/Dropbox (Personal)/Documents/FRESACAD/TADPOLE/TADPOLE/TADPOLE_D1_D2.csv", na.strings=c("NA",-4,"-4.0",""," "))
TADPOLE_D3 <- read.csv("C:/Users/jtame/Dropbox (Personal)/Documents/FRESACAD/TADPOLE/TADPOLE/TADPOLE_D3.csv", na.strings=c("NA",-4,"-4.0",""," ","NaN"))

submissionTemplate <- read_excel("TADPOLE_Simple_Submission_TeamName.xlsx")

submissionTemplate$`Forecast Date` <- as.Date(paste(submissionTemplate$`Forecast Date`,"-01",sep=""))
submissionTemplate$`CN relative probability` <- as.numeric(nrow(submissionTemplate))
submissionTemplate$`MCI relative probability` <-  as.numeric(nrow(submissionTemplate))
submissionTemplate$`AD relative probability` <-  as.numeric(nrow(submissionTemplate))

#DataSplit

TrainingSet <- subset(TADPOLE_D1_D2,D1==1)
D2TesingSet <- subset(TADPOLE_D1_D2,D2==1)

#DataProcessing

source('~/GitHub/TADPOLE/dataPreprocessing.R')
source('~/GitHub/TADPOLE/TADPOLE_Train.R')
source('~/GitHub/TADPOLE/predictCognitiveStatus.R')



dataTadpole <- dataTADPOLEPreprocesing(TrainingSet,TADPOLE_D3,TADPOLE_D1_D2_Dict,MinVisit=36,colImputeThreshold=0.25,rowImputeThreshold=0.25)


save(dataTadpole,file="D3DataFrames.RDATA")
load(file="D3DataFrames.RDATA")

D3Testing <- dataTadpole$testingFrame

dataTadpole <- dataTADPOLEPreprocesing(TrainingSet,D2TesingSet,TADPOLE_D1_D2_Dict,MinVisit=36,colImputeThreshold=0.25,rowImputeThreshold=0.25)


save(dataTadpole,file="D2DataFrames.RDATA")
load(file="D2DataFrames.RDATA")


CognitiveClassModels <- TrainTadpoleClassModels(dataTadpole$AdjustedTrainFrame,
                        predictors=c("AGE","PTGENDER",colnames(dataTadpole$AdjustedTrainFrame)[-c(1:22)]),
                        MLMethod=BSWiMS.model,
                        NumberofRepeats = 5)

#CognitiveClassModelsSVM <- TrainTadpoleClassModels(dataTadpole$AdjustedTrainFrame,
#                                                predictors=c("AGE","PTGENDER",colnames(dataTadpole$AdjustedTrainFrame)[-c(1:22)]),
#                                                MLMethod=e1071::svm,
#                                                asFactor=TRUE,
#                                                )


save(CognitiveClassModels,file="CognitiveClassModels.RDATA")

load(file="CognitiveClassModels.RDATA")

dataTadpole$testingFrame$EXAMDATE <- as.Date(dataTadpole$testingFrame$EXAMDATE)

predictADNI <- forecastCognitiveStatus(CognitiveClassModels,dataTadpole$testingFrame)

statusLO <- (predictADNI$lastKownDX == "NL" | predictADNI$lastKownDX == "MCI to NL") + 
  2*(predictADNI$lastKownDX == "Dementia to MCI" | predictADNI$lastKownDX == "NL to MCI" | predictADNI$lastKownDX == "MCI") + 
  3*(predictADNI$lastKownDX == "MCI to Dementia" | predictADNI$lastKownDX == "Dementia")


table(predictADNI$crossprediction$pDX,statusLO)


forecast <- FiveYearForeCast(predictADNI,Subject_datestoPredict=submissionTemplate)
  
