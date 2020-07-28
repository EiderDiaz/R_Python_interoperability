forecastCognitiveStatus <- function(Models,TestDataFrame)
{
  predictors <- Models$predictors
  months <- as.numeric(names(table(TestDataFrame$M)))
  print(months)
  
  
  TestDataFrame$class <- numeric(nrow(TestDataFrame))
  TestDataFrame$TimeToEvent <- numeric(nrow(TestDataFrame))
  
  TestDataFrame$RID <- as.character(TestDataFrame$RID)
  library("FRESA.CAD")
  if (is.null(months))
  {
    months <- as.numeric(names(table(TestDataFrame$M)))
    print(months)
  }
  cpredictors <- predictors
  
  TestDataFrame <- TestDataFrame[order(TestDataFrame$EXAMDATE),]
  TestDataFrame <- TestDataFrame[order(as.numeric(TestDataFrame$RID)),]
  
  pdis <- TestDataFrame$RID
  lastTimepointSet <- TestDataFrame[c(pdis[1:(length(pdis)-1)] != pdis[-1],TRUE),]
  rownames(lastTimepointSet) <- lastTimepointSet$RID
  print(nrow(TestDataFrame))
  print(nrow(lastTimepointSet))
  
  BaseTimepointSet <- TestDataFrame[c(TRUE,pdis[-1] != pdis[1:(length(pdis)-1)]),]
  rownames(BaseTimepointSet) <- BaseTimepointSet$RID
  deltaFeaturepredictors <- predictors[regexpr('_bl', predictors) < 0][-(c(1:2))]
  print(nrow(BaseTimepointSet))
  
  TimePointsSubset <- list();
  Orderbytimepoint <- NULL
  m <- 0
  i <- 1;
  lastDate <- BaseTimepointSet$EXAMDATE
  names(lastDate) <- BaseTimepointSet$RID
  lastDX <- BaseTimepointSet$DX
  names(lastDX) <- BaseTimepointSet$RID
  print(sum(is.na(TestDataFrame$M)))
#  totR <- 0
  for (m in months)
  {
    TimePointsSubset[[i]] <- subset(TestDataFrame,M == m)
#    totR <-  totR + nrow(TimePointsSubset[[i]])
#    print(c(m,totR,nrow(TimePointsSubset[[i]])))
    rownames(TimePointsSubset[[i]]) <- TimePointsSubset[[i]]$RID
    TimePointsSubset[[i]]$Year_bl_LastVisit <- lastTimepointSet[TimePointsSubset[[i]]$RID,"Years_bl"]
    TimePointsSubset[[i]]$Last_DX <-  lastTimepointSet[TimePointsSubset[[i]]$RID,"DX"]
    ldx <- TimePointsSubset[[i]]$DX
    names(ldx) <- TimePointsSubset[[i]]$RID
    ldx <- ldx[!is.na(TimePointsSubset[[i]]$DX)]
    lastDX[names(ldx)] <- ldx
    lastDate[names(ldx)] <- TimePointsSubset[[i]][names(ldx),"EXAMDATE"]
    TimePointsSubset[[i]]$TimeToLastVisit <- TimePointsSubset[[i]]$Year_bl_LastVisit - TimePointsSubset[[i]]$Years_bl
    deltaObservations <- TimePointsSubset[[i]][,deltaFeaturepredictors] - BaseTimepointSet[rownames(TimePointsSubset[[i]]),deltaFeaturepredictors]
    colnames(deltaObservations) <- paste("Delta",colnames(deltaObservations),sep="_")
    TimePointsSubset[[i]] <- cbind(TimePointsSubset[[i]],deltaObservations)
    Orderbytimepoint <- rbind(Orderbytimepoint,TimePointsSubset[[i]])
    i <- i + 1
  }
  
  TestDataFrame <- Orderbytimepoint
  print(nrow(TestDataFrame))
  
  Orderbytimepoint <- NULL
  
  
  if (is.null(Models$CrossModels[[1]]$oridinalModels))
  {
    crossprediction <- predict(Models$CrossModels[[1]],lastTimepointSet)
  }
  else
  {
    crossprediction <- predict(Models$CrossModels[[1]]$oridinalModels,lastTimepointSet)
  }
  for (n in 2:length(Models$CrossModels))
  {
    if (is.null(Models$CrossModels[[n]]$oridinalModels))
    {
      crossprediction <- crossprediction + predict(Models$CrossModels[[n]],lastTimepointSet)
    }
    else
    {
      crossprediction <- crossprediction + predict(Models$CrossModels[[n]]$oridinalModels,lastTimepointSet)
    }
  }
  if (is.null(Models$CrossModels[[1]]$oridinalModels))
  {
    crossprediction <- as.integer(crossprediction/length(Models$CrossModels)+0.5)
  }
  else
  {
    crossprediction <- as.data.frame(crossprediction[,(ncol(crossprediction)-2):ncol(crossprediction)]/length(Models$CrossModels))
    rownames(crossprediction) <- lastTimepointSet$RID
  }
  crossprediction <- crossprediction[order(as.numeric(rownames(crossprediction))),]
  crossprediction$pDX <- apply(crossprediction,1,which.max)
  crossprediction$DX <- lastTimepointSet[rownames(crossprediction),"DX"]
  crossprediction$date <- lastTimepointSet[rownames(crossprediction),"EXAMDATE"]

  TestDataFrame <- TestDataFrame[order(TestDataFrame$EXAMDATE),]
  TestDataFrame <- TestDataFrame[order(as.numeric(TestDataFrame$RID)),]
  pdis <- TestDataFrame$RID
  lastTimepointSet <- TestDataFrame[c(pdis[1:(length(pdis)-1)] != pdis[-1],TRUE),]
  rownames(lastTimepointSet) <- lastTimepointSet$RID
  print(nrow(lastTimepointSet))
  
  
  MCITOADprediction <- predict(Models$MCIToADModels[[1]],lastTimepointSet)
  MCITOADTimeprediction <- predict(Models$MCIToADTimeModel[[1]],lastTimepointSet)
  NCToMCIprediction <- predict(Models$NCToMCIModel[[1]],lastTimepointSet)
  NCToMCITimeprediction <- predict(Models$NCToMCITimeModel[[1]],lastTimepointSet)
  for (n in 2:length(Models$MCIToADModels))
  {
    MCITOADprediction <- MCITOADprediction + predict(Models$MCIToADModels[[n]],lastTimepointSet)
    MCITOADTimeprediction <- MCITOADTimeprediction + predict(Models$MCIToADTimeModel[[n]],lastTimepointSet)
    NCToMCIprediction <- NCToMCIprediction + predict(Models$NCToMCIModel[[n]],lastTimepointSet)
    NCToMCITimeprediction <- NCToMCITimeprediction + predict(Models$NCToMCITimeModel[[n]],lastTimepointSet)
  }
  MCITOADprediction <- MCITOADprediction/length(Models$CrossModels)
  MCITOADTimeprediction <- MCITOADTimeprediction/length(Models$CrossModels)
  NCToMCIprediction <- NCToMCIprediction/length(Models$CrossModels)
  NCToMCITimeprediction <- NCToMCITimeprediction/length(Models$CrossModels)
  

  predictions <- list(orderedTestFrame = TestDataFrame,
                      predictedTimePointData = lastTimepointSet,
                      fullPredictors = predictors, 
                      crossprediction = crossprediction,
                      MCITOADprediction = MCITOADprediction,
                      MCITOADTimeprediction = MCITOADTimeprediction,
                      NCToMCIprediction = NCToMCIprediction,
                      NCToMCITimeprediction = NCToMCITimeprediction,
                      lastKownDX = lastDX,
                      lastDateDX = lastDate
                      )
  
  
  return (predictions)
}