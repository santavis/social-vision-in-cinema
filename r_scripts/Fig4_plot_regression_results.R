# Plot results of the regression analysis using the 500ms window only.

# Severi Santavirta 17.10.2024

library(superheat)
library(ggplot2)
library(corrplot)
library(stringr)
library(lessR)

##------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Pupil size

features <- c('pupil','isc','fixationRate','blinkRate')
shifts <- c('1000','0','0','0')

for(i in seq(from=1,to=4)){
  beta <- read.csv(paste('path/regression/combined_results/',features[i],'_shift',shifts[i],'_betas.csv',sep=''))
  r <-  read.csv(paste('path/regression/combined_results/',features[i],'_shift',shifts[i],'_rs.csv',sep=''))
  cons <-  read.csv(paste('path/regression/combined_results/',features[i],'_shift',shifts[i],'_consistencies.csv',sep=''))
  pval <-  read.csv(paste('path/regression/combined_results/',features[i],'_shift',shifts[i],'_pvals.csv',sep=''))
  
  # Give clean names for the predictors
  names <- c('(A) Intensity / roughness, D','(A) Spectral properties','Plesant situation','Gaze on object','(V) Movement','(V) Luminance / entropy','Scene cut','Talking','Gaze on body','Gaze on face','Gaze on background','(A) Intensity / roughness','Body movement','Unpleasant situation','(A) Spectral properties, D','(V) Luminance / entropy, D')
  beta$Row <- names
  r$Row <- names
  cons$Row <- names
  pval$Row <- names
  
  # Sort by custom order and select only the 500ms time window
  custom_order <- c('Talking','Plesant situation','Body movement','Unpleasant situation','(A) Spectral properties, D','(A) Spectral properties','(V) Movement','(A) Intensity / roughness, D','(V) Luminance / entropy, D','(A) Intensity / roughness','(V) Luminance / entropy','Scene cut','Gaze on object','Gaze on background','Gaze on body','Gaze on face')
  custom_order_factor <- factor(beta$Row, levels = custom_order)
  beta <- beta[order(custom_order_factor),c(1,3)]
  r <- r[order(custom_order_factor),c(1,3)]
  cons <- cons[order(custom_order_factor),c(1,3)]
  pval <- pval[order(custom_order_factor),c(1,3)]
  
  # Collect the results to one dataframe over different dependent variables
  if(i == 1){
    betaAll <- beta 
    rAll <- r
    consAll <- cons
    pvalAll <- pval
  }else{
    betaAll <- cbind(betaAll,beta[,2])
    rAll <- cbind(rAll,r[,2])
    consAll <- cbind(consAll,cons[,2])
    pvalAll <- cbind(pvalAll,pval[,2])
  }
}

# Name the columns
colnames(betaAll) <- c("Features",features)
colnames(rAll) <- c("Features",features)
colnames(consAll) <- c("Features",features)
colnames(pvalAll) <- c("Features",features)

# Plot p-values as asterisks
pvalNum <- as.matrix(pvalAll[,2:5])
pvalNum[pvalNum>0.05] <- 1
pvalStr <- apply(pvalNum, c(1, 2), as.character)
pvalStr[pvalStr != "1"] <- "*"
pvalStr[pvalStr == "1"] <- ""

# Make data matrix for plotting
data <- as.matrix(betaAll[,2:ncol(betaAll)])
rownames(data)<- betaAll$Features

# We do not want to plot the 0 betas as a color (zero was assigned if the effect was inconsistent in the cross-validation)
# Add infinite value to them, to plot them as grey in the heatmap
data[as.matrix(consAll[,2:ncol(pvalAll)])==0] <- NA

# We cannot plot all the results in one heatmap, since the betas have different scales. Superheat also has a bug that a single column cannot be plotted. Hence we plot the full heatmap 4 times scaling the colours for different features every time. The colums are then separated from each other using Inkscape.

# Scale based on the absolute max beta for the i:th column
require(lattice)
for(i in seq(from=1,to=4)){
  data_i <- as.matrix(data[,i])
  scaling <- max(abs(min(data_i,na.rm = TRUE)),abs(max(data_i,na.rm = TRUE)))
  pdf(paste('path/regression/combined_results/heatmaps/Fig4_regression_results/',features[i],'_shift',shifts[i],'_legend.pdf',sep=''),height = 10,width = 8)
  p <- superheat(data,
            left.label.col = "white",
            bottom.label.col = "white",
            heat.pal = rev(c("#B2182B","#D6604D","#F4A582","white","#92C5DE","#4393C3","#2166AC")),
            heat.pal.values = c(0,0.10,0.30,0.50,0.70,0.90,1),
            heat.lim = c(-scaling,scaling),
            X.text = pvalStr,
            X.text.size = 3.5,
            left.label.size = 2,
            bottom.label.size = 2,
            left.label.text.alignment = "right",
            bottom.label.text.alignment = "right",
            bottom.label.text.angle = 90,
            legend = TRUE)
  print(p)
  dev.off()
}
