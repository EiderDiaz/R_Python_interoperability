#' FiveYearForeCast
#'
#' @param Classpredictions Subject wise predictions of NC, MCI or AD
#' @param ADAS_Predictions Subject wise predictions of ADAS score
#' @param Ventricle_Predictions Subject wise predicitons of Ventricle Volumes
#' @param Subject_datestoPredict Frame of Subject and date to Predict
#'
#' @return
#  The data frame with all the predictions
#' @export
#'
#' @examples
FiveYearForeCast <- function(Classpredictions=NULL,ADAS_Predictions=NULL,Ventricle_Predictions=NULL,Subject_datestoPredict=NULL)
{
  predictedFrame <-NULL
  classPredictions <- NULL
  RID <- as.character(Subject_datestoPredict$RID)
  sdMIC <- sd(Classpredictions$MCITOADTimeprediction)
  sdNC <- sd(Classpredictions$NCToMCITimeprediction)
  Forecastdates <- submissionTemplate$`Forecast Date`
  
  statusLO <- (Classpredictions$lastKownDX == "NL" | Classpredictions$lastKownDX == "MCI to NL") + 
    2*(Classpredictions$lastKownDX == "Dementia to MCI" | Classpredictions$lastKownDX == "NL to MCI" | Classpredictions$lastKownDX == "MCI") + 
    3*(Classpredictions$lastKownDX == "MCI to Dementia" | Classpredictions$lastKownDX == "Dementia")
  names(statusLO) <- names(Classpredictions$lastKownDX)
  
  thrNCMCI <- max( (1.5-Classpredictions$NCMCIAUC), (1.0-Classpredictions$pNCtoMCIEvent) )
  thrMCINAD <- max( (1.5-Classpredictions$MCIADAUC), (1.0-Classpredictions$pMCItoADEvent) )           
  thrMCINC <- max ( (1.5-Classpredictions$MCINCAUC), (1.0-Classpredictions$pMCItoNCEvent) )
  
  print(c(thrNCMCI,thrMCINAD,thrMCINC))

  for (n in 1:nrow(Subject_datestoPredict))
  {
      id <- RID[n]
      fdate <- Forecastdates[n] 

      BaseCN_prob <- 1*(statusLO[id] == 1)
      BaseMCI_prob <- 1*(statusLO[id] == 2)
      BaseAD_prob <- 1*(statusLO[id] == 3)
      
      TimeToAD <-  Classpredictions$MCITOADTimeprediction[id]
      TimeToMCI <-  Classpredictions$NCToMCITimeprediction[id]
      TimeToNC <-  Classpredictions$MCITONCTimeprediction[id]

      timeInterval <- as.numeric(fdate-Classpredictions$predictedTimePointData[id,"EXAMDATE"])/365.25
      NCMCITimeLine <- exp(-(timeInterval-0.75*TimeToMCI)/(0.25*TimeToMCI))
      NCMCITimeLine <- 1.0/(1.0+NCMCITimeLine)
      MCIADTimeLine <- exp(-(timeInterval-0.75*TimeToAD)/(0.25*TimeToAD))
      MCIADTimeLine <- 1.0/(1.0+MCIADTimeLine)
      MCINCTimeLine <- exp(-(timeInterval-0.75*TimeToNC)/(0.25*TimeToNC))
      MCINCTimeLine <- 1.0/(1.0+MCINCTimeLine)
      
      NCTOMCIprob <- Classpredictions$NCToMCIprediction[id]*NCMCITimeLine
      NCTOMCIprob <- NCTOMCIprob*( NCTOMCIprob > thrNCMCI ) 
      
      MCITOADprob <- Classpredictions$MCITOADprediction[id]*MCIADTimeLine
      MCITOADprob <- MCITOADprob*( MCITOADprob > thrMCINAD )  
      
      MCITONCprob <- Classpredictions$MCITONCprediction[id]*MCINCTimeLine
      MCITONCprob <- MCITONCprob*( MCITONCprob > thrMCINC )
      
      finalNCProb <- BaseCN_prob*(1.0 - NCTOMCIprob) + BaseMCI_prob*MCITONCprob
      finalMCIProb <- BaseMCI_prob*(1.0 - MCITOADprob) + BaseCN_prob*NCTOMCIprob
      finalADProb <- BaseAD_prob + BaseMCI_prob*MCITOADprob
      
      totalProb <- finalNCProb + finalMCIProb + finalADProb
      finalNCProb <- finalNCProb/totalProb
      finalMCIProb <- finalMCIProb/totalProb
      finalADProb <- finalADProb/totalProb
      
      Subject_datestoPredict[n,4] <- as.numeric(finalNCProb)
      Subject_datestoPredict[n,5] <- as.numeric(finalMCIProb)
      Subject_datestoPredict[n,6] <- as.numeric(finalADProb)
  }
    
  return (Subject_datestoPredict)
}