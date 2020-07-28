forecastCognitiveStatus <- function(Models,TestDataFrame,futureDataPoints=NULL)
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
  
  TestDataFrame <- TestDataFrame[order(TestDataFrame$Years_bl),]
  TestDataFrame <- TestDataFrame[order(TestDataFrame$RID),]
  
  pdis <- TestDataFrame$RID
  lastTimepointSet <- TestDataFrame[c(pdis[1:(length(pdis)-1)] != pdis[-1],TRUE),]
  rownames(lastTimepointSet) <- lastTimepointSet$RID
  print(nrow(lastTimepointSet))
  
  BaseTimepointSet <- TestDataFrame[c(TRUE,pdis[-1] != pdis[1:(length(pdis)-1)]),]
  rownames(BaseTimepointSet) <- BaseTimepointSet$RID
  deltaFeaturepredictors <- predictors[regexpr('_bl', predictors) < 0][-(c(1:2))]
  print(nrow(BaseTimepointSet))
  
  TimePointsSubset <- list();
  Orderbytimepoint <- NULL
  m <- 0
  i <- 1;
  for (m in months)
  {
    TimePointsSubset[[i]] <- subset(TestDataFrame,M == m)
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
  
  TestDataFrame <- Orderbytimepoint
  Orderbytimepoint <- NULL
  
  
  crossprediction <- predict(Models$CrossModels[[1]]$oridinalModels,lastTimepointSet)
  for (n in 2:length(Models$CrossModels))
  {
    crossprediction <- crossprediction + predict(Models$CrossModels[[n]]$oridinalModels,lastTimepointSet)
  }
  crossprediction <- as.data.frame(crossprediction[,(ncol(crossprediction)-2):ncol(crossprediction)]/length(Models$CrossModels))
  rownames(crossprediction) <- lastTimepointSet$RID
  crossprediction <- crossprediction[order(as.numeric(rownames(crossprediction))),]
  crossprediction$pDX <- apply(crossprediction,1,which.max)
  crossprediction$DX <- lastTimepointSet[rownames(crossprediction),"DX"]
  crossprediction$date <- lastTimepointSet[rownames(crossprediction),"EXAMDATE"]
  
  predictions <- list(orderedTestFrame = TestDataFrame, fullPredictors = predictors, crossprediction=crossprediction)
  
  
  return (predictions)
}