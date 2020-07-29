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
  Forecastdates <- submissionTemplate$`Forecast Date`
  for (n in 1:nrow(Subject_datestoPredict))
  {
      id <- RID[n]
      fdate <- Forecastdates[n] 
      BaseCN_prob <- Classpredictions$crossprediction[id,"0"]
      BaseMCI_prob <- Classpredictions$crossprediction[id,"1"]
      BaseAD_prob <- Classpredictions$crossprediction[id,"2"]

      TimeToAD <-  exp(Classpredictions$MCITOADTimeprediction[id])
      TimeToMCI <-  exp(Classpredictions$NCToMCITimeprediction[id])
      timeInterval <- as.numeric(fdate-Classpredictions$predictedTimePointData[id,"EXAMDATE"])/365.25
      NCMCITimeLine <- exp(-(timeInterval-0.5*TimeToMCI)/TimeToMCI)
      NCMCITimeLine <- 1.0/(1.0+NCMCITimeLine)
      MCIADTimeLine <- exp(-(timeInterval-0.5*TimeToAD)/TimeToAD)
      MCIADTimeLine <- 1.0/(1.0+MCIADTimeLine)
      
      NCTOMCIprob <- Classpredictions$NCToMCIprediction[id]*NCMCITimeLine
      MCITOADprob <- Classpredictions$MCITOADprediction[id]*MCIADTimeLine
      
      finalNCProb <- BaseCN_prob*(1.0 - NCTOMCIprob)
      finalMCIProb <- BaseMCI_prob*(1.0 - MCITOADprob) + BaseCN_prob*NCTOMCIprob
      finalADProb <- BaseAD_prob + BaseMCI_prob*MCITOADprob
      
      totalProb <- finalNCProb + finalMCIProb + finalADProb
      finalNCProb <- finalNCProb/totalProb
      finalMCIProb <- finalMCIProb/totalProb
      finalADProb <- finalADProb/totalProb
      
#     if (Subject_datestoPredict[n,2]==1)
#      {
#        print(c(id,timeInterval,NCMCITimeLine,MCIADTimeLine,BaseCN_prob,BaseMCI_prob,BaseAD_prob,finalNCProb,finalMCIProb,finalADProb))
#      }
#      if (Subject_datestoPredict[n,2]==60)
#      {
#        print(c(id,timeInterval,NCMCITimeLine,MCIADTimeLine,BaseCN_prob,BaseMCI_prob,BaseAD_prob,finalNCProb,finalMCIProb,finalADProb))
#      }
      
      Subject_datestoPredict[n,4] <- as.numeric(finalNCProb)
      Subject_datestoPredict[n,5] <- as.numeric(finalMCIProb)
      Subject_datestoPredict[n,6] <- as.numeric(finalADProb)
  }
    
  return (Subject_datestoPredict)
}