# Correlation analysis between the dependent variables (Figure 2 & Figure SI2)

# Severi Santavirta 16.10.2024

library(corrplot)
library(ggplot2)
library(matrixStats)
library(gridExtra)

input <- "path/dependent_variable_correlations/data"
dset <- c("localizer","kasky","conjuring")
tws <- c(200,500,1000,2000,4000)

for(tw in seq(from=1,to=length(tws))){
  for(d in seq(from=1,to=length(dset))){
    
    # Load dataset
    df <- read.csv(paste(input,"/correlation_dataset_",dset[d],"_tw",tws[tw],".csv",sep = ""))
    colnames(df) <- c("trial_ds",'Pupil size',"ISC","Fixation rate","Blink sychronization")
    
    # Calculate correlations separately for each trial
    for(tr in seq(from=1,to=max(df$trial_ds))){
      df_trial <- df[df$trial_ds==tr,2:5]
      if(tr==1){
        cormat <- cor(df_trial) 
      }else{
        cormat <- cormat+cor(df_trial) 
      }
    }
    
    # Average cormat over trials and save the signs
    cormat_avg_dset <- cormat/max(df$trial_ds)
    cormat_dset_sign <- matrix(0,nrow = 4,ncol = 4)
    cormat_dset_sign[cormat_avg_dset>0] <- 1
    
    # Sum over datasets
    if(d==1){
      cormat_avg <- cormat_avg_dset
      cormat_sign <- cormat_dset_sign
    }else{
      cormat_avg <- cormat_avg + cormat_avg_dset
      cormat_sign <- cormat_sign + cormat_dset_sign
    }
  }
  
  # Average over dsets
  cormat_avg <- cormat_avg/3
  
  # Asterisks to the correlations that were consistent over the datasets
  write.csv(cormat_sign,paste("path/dependent_variable_correlations/plots/tw",tws[tw],"sign.csv",sep = ""))
  
  # Use corplot to output the correlation matrix (add the asterisks manually)
  pdf(paste("path/dependent_variable_correlations/plots/tw",tws[tw],".pdf",sep = ""),height = 10,width = 10)
  corrplot(cormat_avg,type = 'lower',diag = F,method = 'color',tl.col = "black",tl.cex = 3,number.cex = 3,addCoef.col = "grey50",col=colorRampPalette(c("#2166AC","#4393C3","#92C5DE","#D1E5F0","white","#FDDBC7","#F4A582","#D6604D","#B2182B"))(20))
  dev.off()
}

##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Plot pupil time series for Experiment 1

df <- read.csv("path/dependent_variable_correlations/data/pupil_for_timeseries_localizer_tw1000.csv")

# Lineplot with ribbon
data_pupil = as.data.frame(rowMedians(as.matrix(df)))
colnames(data_pupil) = "med"
data_pupil$t <- seq(from = 1, to = nrow(data_pupil)) * 1
q <- rowQuantiles(as.matrix(df),probs = c(0,.1,.2,.3,.7,.8,.9,1)) # Calculate certain quantiles from the data
pdf("path/dependent_variable_correlations/plots/pupil_timeseries_localizer.pdf",height = 3,width = 10)
ggplot(data_pupil,aes(x=t,y=med)) + 
  geom_ribbon(aes(ymin=q[,2],ymax=q[,7]),fill = "#F4A582") + # 80 % within this ribbon
  geom_ribbon(aes(ymin=q[,3],ymax=q[,6]),fill="#D6604D") + # 60% raters within this ribbon
  geom_ribbon(aes(ymin=q[,4],ymax=q[,5]),fill="#B2182B") + # 40% raters within this ribbon
  geom_line(size=0.5) + 
  xlim(0,max(data_pupil$t)) +
  ylim(0.5,1.3) +
  xlab("Time (s)") +
  ylab("Normalized pupil size") + 
  theme_bw() +
  theme(
    axis.title.x = element_text(size = 16),  # Increase x-axis title size
    axis.title.y = element_text(size = 16),  # Increase y-axis title size
    axis.text.x = element_text(size = 14),   # Increase x-axis tick label size
    axis.text.y = element_text(size = 14))    # Increase y-axis tick label size
dev.off()

## Plot ISC time series for Experiment 1
df <- read.csv("path/dependent_variable_correlations/data/isc_for_timeseries_localizer_tw1000.csv")

# Lineplot with ribbon
data_isc = as.data.frame(rowMedians(as.matrix(df[,3:ncol(df)]),na.rm = T))
colnames(data_isc) = "med"
data_isc$t <- seq(from = 1, to = nrow(data_isc)) * 1
q <- rowQuantiles(as.matrix(df),probs = c(0,.1,.2,.3,.7,.8,.9,1)) # Calculate certain quantiles from the data
pdf("path/dependent_variable_correlations/plots/isc_timeseries_localizer.pdf",height = 3,width = 10)
ggplot(data_isc,aes(x=t,y=med)) + 
  geom_ribbon(aes(ymin=q[,2],ymax=q[,7]),fill = "#92C5DE") + # 80 % within this ribbon
  geom_ribbon(aes(ymin=q[,3],ymax=q[,6]),fill="#4393C3") + # 60% raters within this ribbon
  geom_ribbon(aes(ymin=q[,4],ymax=q[,5]),fill="#2166AC") + # 40% raters within this ribbon
  geom_line(size=0.5) + 
  xlim(0,max(data_isc$t)) +
  ylim(0,1) +
  xlab("Time (s)") +
  ylab("ISC") + 
  theme_bw() +
  theme(
    axis.title.x = element_text(size = 16),  # Increase x-axis title size
    axis.title.y = element_text(size = 16),  # Increase y-axis title size
    axis.text.x = element_text(size = 14),   # Increase x-axis tick label size
    axis.text.y = element_text(size = 14))    # Increase y-axis tick label size
dev.off()



