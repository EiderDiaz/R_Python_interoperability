
##Load the Datasets
library(readxl)


TADPOLE_D1_D2_Dict <- read.csv("C:/Users/jtame/Dropbox (Personal)/Documents/FRESACAD/TADPOLE/TADPOLE/TADPOLE_D1_D2_Dict.csv", na.strings=c("NA",-4,"-4.0",""," "))

TADPOLE_D1_D2 <- read.csv("C:/Users/jtame/Dropbox (Personal)/Documents/FRESACAD/TADPOLE/TADPOLE/TADPOLE_D1_D2.csv", na.strings=c("NA",-4,"-4.0",""," "))

TADPOLE_D3 <- read.csv("C:/Users/jtame/Dropbox (Personal)/Documents/FRESACAD/TADPOLE/TADPOLE/TADPOLE_D3.csv", na.strings=c("NA",-4,"-4.0",""," ","NaN"))

submissionTemplate <- read_excel("TADPOLE_Simple_Submission_TeamName.xlsx")

submissionTemplate$`Forecast Date` <- as.Date(paste(submissionTemplate$`Forecast Date`,"-01",sep=""))

#DataSplit

TrainingSet <- subset(TADPOLE_D1_D2,D1==1)
D2TesingSet <- subset(TADPOLE_D1_D2,D2==1)

#DataProcessing

source('~/GitHub/TADPOLE/dataPreprocessing.R')
source('~/GitHub/TADPOLE/TADPOLE_Train.R')


dataTadpole <- dataTADPOLEPreprocesing(TrainingSet,TADPOLE_D3,TADPOLE_D1_D2_Dict,MinVisit=36,colImputeThreshold=0.25,rowImputeThreshold=0.25)


save(dataTadpole,file="D3DataFrames.RDATA")

dataTadpole <- dataTADPOLEPreprocesing(TrainingSet,D2TesingSet,TADPOLE_D1_D2_Dict,MinVisit=36,colImputeThreshold=0.25,rowImputeThreshold=0.25)


save(dataTadpole,file="D2DataFrames.RDATA")



CognitiveClassModels <- TrainTadpoleClassModels(dataTadpole$AdjustedTrainFrame,
                        predictors=c("AGE","PTGENDER",colnames(dataTadpole$AdjustedTrainFrame)[-c(1:22)]),
                        MLMethod=BSWiMS.model,
                        NumberofRepeats = 5)

save(CognitiveClassModels,file="CognitiveClassModels.RDATA")

load(file="D2DataFrames.RDATA")

dataTadpole$testingFrame$EXAMDATE <- as.Date(dataTadpole$testingFrame$EXAMDATE)

check <- forecastCognitiveStatus(CognitiveClassModels,dataTadpole$testingFrame,submissionTemplate)

table(check$crossprediction$DX)
status <- (check$crossprediction$DX == "NL" | check$crossprediction$DX == "MCI to NL") + 
  2*(check$crossprediction$DX == "Dementia to MCI" | check$crossprediction$DX == "NL to MCI" | check$crossprediction$DX == "MCI") + 
  3*(check$crossprediction$DX == "MCI to Dementia" | check$crossprediction$DX == "Dementia")

status[is.na(status)] <- 4

table(check$crossprediction$pDX,status)

print(nrow(check$crossprediction))

