
##Load the Datasets

TADPOLE_D1_D2_Dict <- read.csv("C:/Users/jtame/Dropbox (Personal)/Documents/FRESACAD/TADPOLE/TADPOLE/TADPOLE_D1_D2_Dict.csv", na.strings=c("NA",-4,"-4.0",""," "))

TADPOLE_D1_D2 <- read.csv("C:/Users/jtame/Dropbox (Personal)/Documents/FRESACAD/TADPOLE/TADPOLE/TADPOLE_D1_D2.csv", na.strings=c("NA",-4,"-4.0",""," "))

TADPOLE_D3 <- read.csv("C:/Users/jtame/Dropbox (Personal)/Documents/FRESACAD/TADPOLE/TADPOLE/TADPOLE_D3.csv", na.strings=c("NA",-4,"-4.0",""," ","NaN"))


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

