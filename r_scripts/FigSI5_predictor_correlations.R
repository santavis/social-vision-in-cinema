# Plot predictor correlations over multiple time windows

# Severi Santavirta 17.10.2024

library(corrplot)
library(lessR)

# Load data
tw200 <- read.csv("path/predictor_correlations/predictors_tw200.csv",sep=",")
tw500 <- read.csv("path/predictor_correlations/predictors_tw500.csv",sep=",")
tw1000 <- read.csv("path/predictor_correlations/predictors_tw1000.csv",sep=",")
tw2000 <- read.csv("path/predictor_correlations/predictors_tw2000.csv",sep=",")
tw4000 <- read.csv("path/predictor_correlations/predictors_tw4000.csv",sep=",")

# Calculate correlation matrices
cormat200 <- cor(tw200)
cormat500 <- cor(tw500)
cormat1000 <- cor(tw1000)
cormat2000 <- cor(tw2000)
cormat4000 <- cor(tw4000)

# Calculate average correlation matrix over the analyzed time winwows
cormat_avg = (cormat200 + cormat500 + cormat1000 + cormat2000 + cormat4000) / 5

# Clean predictor names
predictors <- colnames(cormat_avg)
predictors[1] <- 'V Luminance'
predictors[2] <- 'V Entropy'
predictors[3] <- 'V Spatial energy LF'
predictors[4] <- 'V Spatial energy HF'
predictors[5] <- 'V Luminance diff'
predictors[6] <- 'V Entropy diff'
predictors[7] <- 'V Spatial energy LF diff'
predictors[8] <- 'V Spatial energy HF diff'
predictors[9] <- 'V Optic flow'
predictors[10] <- 'V Differential energy'
predictors[11] <- 'A RMS'
predictors[12] <- 'A Waveform sign-change rate' # Zero crossing
predictors[13] <- 'A Spectral geometric mean' # Centroid 
predictors[14] <- 'A Spectral standard deviation' # Spread
predictors[15] <- 'A Entropy'
predictors[16] <- 'A High-frequency energy' # Rolloff85
predictors[17] <- 'A Roughness'
predictors[18] <- 'A RMS diff'
predictors[19] <- 'A Waveform sign-change eate diff' # Zero crossing Diff
predictors[20] <- 'A Spectral geometric mean diff' # Centroid diff
predictors[21] <- 'A Spectral standard deviation diff' # Spread diff
predictors[22] <- 'A Entropy diff'
predictors[23] <- 'A High-frequency energy diff' # Rolloff85 diff
predictors[24] <- 'A Roughness diff'
predictors[25] <- 'Eyes'
predictors[26] <- 'Mouth'
predictors[27] <- 'Other face area'
predictors[28] <- 'Body parts'
predictors[29] <- 'Object'
predictors[30] <- 'Background'
predictors[31] <- 'Scene cut'
predictors[32] <- 'Aggressive'
predictors[33] <- 'Aroused'
predictors[34] <- 'Body movement'
predictors[35] <- 'Pain'
predictors[36] <- 'Playful'
predictors[37] <- 'Pleasant feelings'
predictors[38] <- 'Talking'
predictors[39] <- 'Unpleasant feelings'

colnames(cormat_avg) <- predictors
rownames(cormat_avg) <- predictors

# Cluster the correlation matrix hierarchically and choose logiaclly meaningful number of luster without simplifying the model too much
png(file = "path/predictor_correlations/avg_cormat.png",width = 1300,height=1300)
corrplot(cormat_avg,order = "hclust", hclust.method = "average",tl.col = "black",method = "number", col=colorRampPalette(c("#2166AC","#4393C3","#92C5DE","#D1E5F0","#FDDBC7","#F4A582","#D6604D","#B2182B"))(20),addrect = 16)
dev.off()



