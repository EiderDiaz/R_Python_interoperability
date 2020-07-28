
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

CognitiveClassModelsSVM <- TrainTadpoleClassModels(dataTadpole$AdjustedTrainFrame,
                                                predictors=c("AGE","PTGENDER",colnames(dataTadpole$AdjustedTrainFrame)[-c(1:22)]),
                                                MLMethod=e1071::svm,
                                                asFactor=TRUE,
                                                )


save(CognitiveClassModels,file="CognitiveClassModels.RDATA")

load(file="CognitiveClassModels.RDATA")

dataTadpole$testingFrame$EXAMDATE <- as.Date(dataTadpole$testingFrame$EXAMDATE)

predictADNI <- forecastCognitiveStatus(CognitiveClassModelsSVM,dataTadpole$testingFrame)

predictADNI$crossprediction

table(predictADNI$crossprediction$DX)
table(predictADNI$lastDX)
status <- (predictADNI$crossprediction$DX == "NL" | predictADNI$crossprediction$DX == "MCI to NL") + 
  2*(predictADNI$crossprediction$DX == "Dementia to MCI" | predictADNI$crossprediction$DX == "NL to MCI" | predictADNI$crossprediction$DX == "MCI") + 
  3*(predictADNI$crossprediction$DX == "MCI to Dementia" | predictADNI$crossprediction$DX == "Dementia")

status[is.na(status)] <- 4

statusLO <- (predictADNI$lastKownDX == "NL" | predictADNI$lastKownDX == "MCI to NL") + 
  2*(predictADNI$lastKownDX == "Dementia to MCI" | predictADNI$lastKownDX == "NL to MCI" | predictADNI$lastKownDX == "MCI") + 
  3*(predictADNI$lastKownDX == "MCI to Dementia" | predictADNI$lastKownDX == "Dementia")

table(status,statusLO)

predTable <- predictADNI$crossprediction

table(predictADNI$crossprediction$pDX,status)

table(predictADNI$crossprediction$pDX,statusLO)

table(status,statusLO)

length(predictADNI$crossprediction$pDX)
length(predictADNI$MCITOADprediction)

table(statusLO,predictADNI$MCITOADprediction > 0.5)
table(predictADNI$crossprediction$pDX,predictADNI$MCITOADprediction > 0.5)

table(statusLO,predictADNI$NCToMCIprediction > 0.5)
table(predictADNI$crossprediction$pDX,predictADNI$NCToMCIprediction > 0.5)

print(nrow(predictADNI$crossprediction))

