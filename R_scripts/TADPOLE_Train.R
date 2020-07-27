#AdjustedFrame The datatoPredict
#Predictors the predictors
#The months for training
#numberOfRandomSamples the number of samples from the adjustedFrame
#MLMethod the Machine Learning method
#asFactor is the class should be treated as a factor
#... parameters to be passed to the ML method

#It will return models that predict if a subject will convert to MCI to AD
#it will return models that predict the time to conversion

TrainTadpoleClassModels <- function(AdjustedFrame,predictors,months=NULL,numberOfRandomSamples=5,MLMethod=BSWiMS.model,asFactor=FALSE,...)
{
  
  library("FRESA.CAD")
  if (is.null(months))
  {
    months <- names(table(AdjustedFrame$M))
  }
  cpredictors <- predictors
  
  AdjustedFrame <- AdjustedFrame[order(AdjustedFrame$Years_bl),]
  AdjustedFrame <- AdjustedFrame[order(AdjustedFrame$RID),]
  
  pdis <- AdjustedFrame$RID
  lastTimepointSet <- AdjustedFrame[c(pdis[1:(length(pdis)-1)] != pdis[-1],TRUE),]
  rownames(lastTimepointSet) <- lastTimepointSet$RID

  BaseTimepointSet <- AdjustedFrame[c(TRUE,pdis[-1] != pdis[1:(length(pdis)-1)]),]
  rownames(BaseTimepointSet) <- BaseTimepointSet$RID
  deltaFeaturepredictors <- predictors[regexpr('_bl', predictors) < 0][-(c(1:2))]
  
  TimePointsSubset <- list();
  Orderbytimepoint <- NULL
  m <- 0
  i <- 1;
  for (m in months)
  {
    TimePointsSubset[[i]] <- subset(AdjustedFrame,M == m)
    rownames(TimePointsSubset[[i]]) <- TimePointsSubset[[i]]$RID
    TimePointsSubset[[i]]$Year_bl_LastVisit <- lastTimepointSet[TimePointsSubset[[i]]$RID,"Years_bl"]
    TimePointsSubset[[i]]$Last_DX <- lastTimepointSet[TimePointsSubset[[i]]$RID,"DX"]
    TimePointsSubset[[i]]$TimeToLastVisit <- TimePointsSubset[[i]]$Year_bl_LastVisit - TimePointsSubset[[i]]$Years_bl
    deltaObservations <- TimePointsSubset[[i]][,deltaFeaturepredictors] - BaseTimepointSet[rownames(TimePointsSubset[[i]]),deltaFeaturepredictors]
    colnames(deltaObservations) <- paste("Delta",colnames(deltaObservations),sep="_")
    TimePointsSubset[[i]] <- cbind(TimePointsSubset[[i]],deltaObservations)
    TimePointsSubset[[i]] <- TimePointsSubset[[i]][complete.cases(TimePointsSubset[[i]]),]
    Orderbytimepoint <- rbind(Orderbytimepoint,TimePointsSubset[[i]])
    i <- i + 1
  }
  
  AdjustedFrame <- Orderbytimepoint
  Orderbytimepoint <- NULL
  predictors <- c(predictors,colnames(deltaObservations))
  

  ## Get All the MCI subjects that progressed
  
  table(AdjustedFrame$DX)
  MCISubset <- subset(AdjustedFrame,(DX_bl == "LMCI" | DX_bl == "EMCI") & DX == "MCI")
  
  subsetMCIADConversion <-  as.data.frame(subset(AdjustedFrame,DX == "MCI to Dementia"))
  pidss <- subsetMCIADConversion$RID
  tpids <- table(pidss)
  removeTp <- tpids[pidss] == 1
  sum(removeTp == FALSE)
  
  subsetMCIADConversion <- subsetMCIADConversion[removeTp,]
  
  rownames(subsetMCIADConversion) <- subsetMCIADConversion$RID
  
  ### MCI Subset by time points
  
  MCItoADorderbytimepoint <- NULL
  for (m in months)
  {
    TimePointsMCISubset <- subset(MCISubset,M == m)
    rownames(TimePointsMCISubset) <-  TimePointsMCISubset$RID
    TimePointsMCISubset$TimeToEvent <- subsetMCIADConversion[TimePointsMCISubset$RID,"Years_bl"] - TimePointsMCISubset$Years_bl
    MCItoADorderbytimepoint <- rbind(MCItoADorderbytimepoint,TimePointsMCISubset)
  }
  
  controlMCIToADset <- MCItoADorderbytimepoint[is.na(MCItoADorderbytimepoint$TimeToEvent),]
  controlMCIToADset <- subset(controlMCIToADset,Year_bl_LastVisit > 4)
  hist(controlMCIToADset$TimeToLastVisit)
  controlMCIToADset$TimeToEvent <- controlMCIToADset$TimeToLastVisit
  
  caseMCIToADset <- MCItoADorderbytimepoint[!is.na(MCItoADorderbytimepoint$TimeToEvent),]
  caseMCIToADset <- subset(caseMCIToADset,TimeToEvent > 0 )
  hist(caseMCIToADset$TimeToEvent)
  
  ## MCI Modeling Set

    controlMCIToADset$class <- 0
  caseMCIToADset$class <- 1
  MCI_to_AD_set <- rbind(controlMCIToADset,caseMCIToADset)
  MCI_to_AD_set$TimeToLastVisit <- NULL
  
  MCI_to_AD_TrainSet <- MCI_to_AD_set[MCI_to_AD_set$D1==1,]
  
  table(MCI_to_AD_TrainSet$class)
  
  
  ## Modeling MCI conversion
  
  table(MCI_to_AD_TrainSet$VISCODE)
  
  MCI_to_ADSets <- list();
  MCI_TO_AD_Model <- list();
  MCI_TO_AD_TimeModel <- list();
  n=1
  
  for (n in 1:numberOfRandomSamples)
  {
    randomnumber <- sample(1:nrow(MCI_to_AD_TrainSet),nrow(MCI_to_AD_TrainSet))
    MCI_to_AD_RandomSet <- MCI_to_AD_TrainSet[randomnumber,]
    MCI_to_AD_RandomSet <- MCI_to_AD_RandomSet[order(MCI_to_AD_RandomSet$RID),]
    RID <- MCI_to_AD_RandomSet$RID
    set1 <- MCI_to_AD_RandomSet[c(RID[1:length(RID)-1] != RID[-1],TRUE),]
    rownames(set1) <- set1$RID
    set1 <- set1[complete.cases(set1),]
    print(nrow(set1))
    MCI_to_ADSets[[n]] <- set1[,c("class",predictors)]
    MCI_TO_AD_Model[[n]] <- MLMethod(class ~ 1,MCI_to_ADSets[[n]],...)
    set1 <- subset(set1,class==1)
    print(nrow(set1))
    
    MCI_to_ADSets[[n]] <- set1[,c("TimeToEvent",predictors)]
    MCI_to_ADSets[[n]]$TimeToEvent <- log(set1$TimeToEvent)
    MCI_TO_AD_TimeModel[[n]] <- MLMethod(TimeToEvent ~ 1,MCI_to_ADSets[[n]],...)
    
  }
  
  ## Get All the NC subjects that progressed
  
  table(AdjustedFrame$DX,AdjustedFrame$DX_bl)
  NCSubset <- subset(AdjustedFrame,(DX_bl == "CN" | DX_bl == "SMC") & DX == "NL")
  
  
  subsetNCADConversion <-  as.data.frame(subset(AdjustedFrame,DX == "NL to Dementia" | DX == "NL to MCI"))
  pidss <- subsetNCADConversion$RID
  tpids <- table(pidss)
  removeTp <- tpids[pidss] == 1
  sum(removeTp == FALSE)
  
  subsetNCConvConversion <- subsetNCADConversion[removeTp,]
  
  rownames(subsetNCConvConversion) <- subsetNCConvConversion$RID
  
  ### NC Subset by time points
  
  NCConvorderbytimepoint <- NULL
  for (m in months)
  {
    TimePointsNCSubset <- subset(NCSubset,M == m)
    rownames(TimePointsNCSubset) <-  TimePointsNCSubset$RID
    TimePointsNCSubset$TimeToEvent <- subsetNCConvConversion[TimePointsNCSubset$RID,"Years_bl"] - TimePointsNCSubset$Years_bl
    NCConvorderbytimepoint <- rbind(NCConvorderbytimepoint,TimePointsNCSubset)
  }
  
  controlNCConvset <- NCConvorderbytimepoint[is.na(NCConvorderbytimepoint$TimeToEvent),]
  controlNCConvset <- subset(controlNCConvset, Year_bl_LastVisit >= 4)
  hist(controlNCConvset$TimeToLastVisit)
  controlNCConvset$TimeToEvent <- 5*controlNCConvset$TimeToLastVisit
  caseNCConvset <- NCConvorderbytimepoint[!is.na(NCConvorderbytimepoint$TimeToEvent),]
  caseNCConvset <- subset(caseNCConvset,TimeToEvent > 0 )
  hist(caseNCConvset$TimeToEvent)
  
  ## Modeling Nomal congitive Set
  controlNCConvset$class <- 0
  caseNCConvset$class <- 1
  NCConv_set <- rbind(controlNCConvset,caseNCConvset)
  
  NCConv_TrainSet <- NCConv_set[NCConv_set$D1==1,]
  
  table(NCConv_TrainSet$class)
  
  table(NCConv_TrainSet$VISCODE)
  
  NCConvSets <- list();
  NCConv_Model <- list();
  NL_TO_OTHER_TimeModel <- list();
  n=1
  
  
  for (n in 1:numberOfRandomSamples)
  {
    randomnumber <- sample(1:nrow(NCConv_TrainSet),nrow(NCConv_TrainSet))
    NCConv_RandomSet <- NCConv_TrainSet[randomnumber,]
    NCConv_RandomSet <- NCConv_RandomSet[order(NCConv_RandomSet$RID),]
    RID <- NCConv_RandomSet$RID
    set1 <- NCConv_RandomSet[c(RID[1:length(RID)-1] != RID[-1],TRUE),]
    rownames(set1) <- set1$RID
    set1 <- set1[complete.cases(set1),]
    print(nrow(set1))
    
    NCConvSets[[n]] <- set1[,c("class",predictors)]
    NCConv_Model[[n]] <- MLMethod(class ~ 1,NCConvSets[[n]],...)

    set1 <- subset(set1,class==1)
    print(nrow(set1))
    NCConvSets[[n]] <- set1[,c("TimeToEvent",predictors)]
    NCConvSets[[n]]$TimeToEvent <- log(set1$TimeToEvent)
    NL_TO_OTHER_TimeModel[[n]] <- MLMethod(TimeToEvent ~ 1,NCConvSets[[n]],...)
  }


  ## Cross Sectional Modeling
  
  ### Baseline Modeling Set


  class <- 
    2*(AdjustedFrame$DX == "Dementia" | AdjustedFrame$DX == "MCI to Dementia" | AdjustedFrame$DX == "NL to Dementia") +
    1*(AdjustedFrame$DX == "Dementia to MCI" | AdjustedFrame$DX == "MCI" | AdjustedFrame$DX == "NL to MCI")
  
  AdjustedFrame$class <- class
  

  AllADNISets <- list();
  AllADNI_Model <- list();
  n=1
  
  for (n in 1:resamplingNumber)
  {
    randomnumber <- sample(1:nrow(AdjustedFrame),nrow(AdjustedFrame))
    AllADNI_RandomSet <- AdjustedFrame[randomnumber,]
    AllADNI_RandomSet <- AllADNI_RandomSet[order(AllADNI_RandomSet$RID),]
    RID <- AllADNI_RandomSet$RID
    set1 <- AllADNI_RandomSet[c(RID[1:length(RID)-1] != RID[-1],TRUE),]
    rownames(set1) <- set1$RID
    AllADNISets[[n]] <- set1[,c("class",cpredictors)]
    AllADNISets[[n]] <- AllADNISets[[n]][complete.cases(AllADNISets[[n]]),]
    print(nrow(AllADNISets[[n]]))
    print(table(set1$DX_bl,set1$class))
    if (asFactor)
    {
      AllADNISets[[n]]$class <- as.factor(AllADNISets[[n]]$class)
    }
    AllADNI_Model[[n]] <- MLMethod(class ~ .,AllADNISets[[n]],...)
  }
  
  predicitionModels <- list(CrossModels = AllADNI_Model,
                            MCIToADModels=MCI_TO_AD_Model,
                            MCIToADTimeModel = MCI_TO_AD_TimeModel,
                            NCToMCIModel=NCConv_Model,
                            NCToMCITimeModel=NL_TO_OTHER_TimeModel
                            )
  
  return (predicitionModels)
}