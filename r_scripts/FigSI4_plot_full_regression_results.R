# Plot regression results in all analyzed time windows

# Severi Santavirta 17.10.2024

library(superheat)
library(ggplot2)
library(corrplot)
library(stringr)
library(lessR)

##------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Pupil size

feature <- 'pupil'
shift <- '1000'
beta <- read.csv(paste('path/regression/combined_results/',feature,'_shift',shift,'_betas.csv',sep=''))
r <-  read.csv(paste('path/regression/combined_results/',feature,'_shift',shift,'_rs.csv',sep=''))
cons <-  read.csv(paste('path/regression/combined_results/',feature,'_shift',shift,'_consistencies.csv',sep=''))
pval <-  read.csv(paste('path/regression/combined_results/',feature,'_shift',shift,'_pvals.csv',sep=''))

# Give clean names for the predictors
names <- c('(A) Intensity / roughness, D','(A) Spectral properties','Plesant situation','Gaze on object','(V) Movement','(V) Luminance / entropy','Scene cut','Talking','Gaze on body','Gaze on face','Gaze on background','(A) Intensity / roughness','Body movement','Unpleasant situation','(A) Spectral properties, D','(V) Luminance / entropy, D')
beta$Row <- names
r$Row <- names
cons$Row <- names
pval$Row <- names

# Sort by custom order
custom_order <- c('Talking','Plesant situation','Body movement','Unpleasant situation','(A) Spectral properties, D','(A) Spectral properties','(V) Movement','(A) Intensity / roughness, D','(V) Luminance / entropy, D','(A) Intensity / roughness','(V) Luminance / entropy','Scene cut','Gaze on object','Gaze on background','Gaze on body','Gaze on face')
custom_order_factor <- factor(beta$Row, levels = custom_order)
beta <- beta[order(custom_order_factor),]
r <- r[order(custom_order_factor),]
cons <- cons[order(custom_order_factor),]
pval <- pval[order(custom_order_factor),]

# Plot p-value as text over the heatmap
pval[pval>0.05] <- 1
pvalStr <- apply(round(as.matrix(pval[,2:ncol(pval)]),3),c(1,2),as.character)
pvalStr[pvalStr== "1"] <- ""
pvalStr[pvalStr== "0"] <- "<0.001"

# Make data matix for plotting
data <- as.matrix(beta[,2:ncol(pval)])
rownames(data)<- beta$Row
colnames(data)<- c("200 ms","500 ms","1000 ms","2000 ms","4000 ms")

# We do not want to plot the 0 betas as a color (zero was assigned if the effect was inconsistent in the cross-validation)
# Add infinite value to them, to plot them as grey in the heatmap
data[as.matrix(cons[,2:ncol(pval)])==0] <- NA

#Scale based on the absolute max beta
scaling <- max(abs(min(data,na.rm = TRUE)),abs(max(data,na.rm = TRUE)))
pdf(paste('path/regression/combined_results/heatmaps/',feature,'_shift',shift,'.pdf',sep=''),height = 10,width = 8)
superheat(data,
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
          legend = FALSE)
dev.off()

##------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ISC

feature <- 'isc'
shift <- '0'
beta <- read.csv(paste('path/regression/combined_results/',feature,'_shift',shift,'_betas.csv',sep=''))
r <-  read.csv(paste('path/regression/combined_results/',feature,'_shift',shift,'_rs.csv',sep=''))
cons <-  read.csv(paste('path/regression/combined_results/',feature,'_shift',shift,'_consistencies.csv',sep=''))
pval <-  read.csv(paste('path/regression/combined_results/',feature,'_shift',shift,'_pvals.csv',sep=''))

# Give clean names for the predictors
names <- c('(A) Intensity / roughness, D','(A) Spectral properties','Plesant situation','Gaze on object','(V) Movement','(V) Luminance / entropy','Scene cut','Talking','Gaze on body','Gaze on face','Gaze on background','(A) Intensity / roughness','Body movement','Unpleasant situation','(A) Spectral properties, D','(V) Luminance / entropy, D')
beta$Row <- names
r$Row <- names
cons$Row <- names
pval$Row <- names

# Sort by custom order
custom_order <- c('Talking','Plesant situation','Body movement','Unpleasant situation','(A) Spectral properties, D','(A) Spectral properties','(V) Movement','(A) Intensity / roughness, D','(V) Luminance / entropy, D','(A) Intensity / roughness','(V) Luminance / entropy','Scene cut','Gaze on object','Gaze on background','Gaze on body','Gaze on face')
custom_order_factor <- factor(beta$Row, levels = custom_order)
beta <- beta[order(custom_order_factor),]
r <- r[order(custom_order_factor),]
cons <- cons[order(custom_order_factor),]
pval <- pval[order(custom_order_factor),]

# Plot p-value as text over the heatmap
pval[pval>0.05] <- 1
pvalStr <- apply(round(as.matrix(pval[,2:ncol(pval)]),3),c(1,2),as.character)
pvalStr[pvalStr== "1"] <- ""
pvalStr[pvalStr== "0"] <- "<0.001"

# Make data matix for plotting
data <- as.matrix(beta[,2:ncol(pval)])
rownames(data)<- beta$Row
colnames(data)<- c("200 ms","500 ms","1000 ms","2000 ms","4000 ms")

# We do not want to plot the 0 betas as a color (zero was assigned if the effect was inconsistent in the cross-validation)
# Add infinite value to them, to plot them as grey in the heatmap
data[as.matrix(cons[,2:ncol(pval)])==0] <- NA

#Scale based on the absolute max beta
scaling <- max(abs(min(data,na.rm = TRUE)),abs(max(data,na.rm = TRUE)))
pdf(paste('path/regression/combined_results/heatmaps/',feature,'_shift',shift,'.pdf',sep=''),height = 10,width = 8)
superheat(data,
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
          legend = FALSE)
dev.off()

##------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Fixation rate

feature <- 'fixationRate'
shift <- '0'
beta <- read.csv(paste('path/regression/combined_results/',feature,'_shift',shift,'_betas.csv',sep=''))
r <-  read.csv(paste('path/regression/combined_results/',feature,'_shift',shift,'_rs.csv',sep=''))
cons <-  read.csv(paste('path/regression/combined_results/',feature,'_shift',shift,'_consistencies.csv',sep=''))
pval <-  read.csv(paste('path/regression/combined_results/',feature,'_shift',shift,'_pvals.csv',sep=''))

# Give clean names for the predictors
names <- c('(A) Intensity / roughness, D','(A) Spectral properties','Plesant situation','Gaze on object','(V) Movement','(V) Luminance / entropy','Scene cut','Talking','Gaze on body','Gaze on face','Gaze on background','(A) Intensity / roughness','Body movement','Unpleasant situation','(A) Spectral properties, D','(V) Luminance / entropy, D')
beta$Row <- names
r$Row <- names
cons$Row <- names
pval$Row <- names

# Sort by custom order
custom_order <- c('Talking','Plesant situation','Body movement','Unpleasant situation','(A) Spectral properties, D','(A) Spectral properties','(V) Movement','(A) Intensity / roughness, D','(V) Luminance / entropy, D','(A) Intensity / roughness','(V) Luminance / entropy','Scene cut','Gaze on object','Gaze on background','Gaze on body','Gaze on face')
custom_order_factor <- factor(beta$Row, levels = custom_order)
beta <- beta[order(custom_order_factor),]
r <- r[order(custom_order_factor),]
cons <- cons[order(custom_order_factor),]
pval <- pval[order(custom_order_factor),]

# Plot p-value as text over the heatmap
pval[pval>0.05] <- 1
pvalStr <- apply(round(as.matrix(pval[,2:ncol(pval)]),3),c(1,2),as.character)
pvalStr[pvalStr== "1"] <- ""
pvalStr[pvalStr== "0"] <- "<0.001"

# Make data matix for plotting
data <- as.matrix(beta[,2:ncol(pval)])
rownames(data)<- beta$Row
colnames(data)<- c("200 ms","500 ms","1000 ms","2000 ms","4000 ms")

# We do not want to plot the 0 betas as a color (zero was assigned if the effect was inconsistent in the cross-validation)
# Add infinite value to them, to plot them as grey in the heatmap
data[as.matrix(cons[,2:ncol(pval)])==0] <- NA

#Scale based on the absolute max beta
scaling <- max(abs(min(data,na.rm = TRUE)),abs(max(data,na.rm = TRUE)))
pdf(paste('path/regression/combined_results/heatmaps/',feature,'_shift',shift,'.pdf',sep=''),height = 10,width = 8)
superheat(data,
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
          legend = FALSE)
dev.off()

##------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Blink rate

feature <- 'blinkRate'
shift <- '0'
beta <- read.csv(paste('path/regression/combined_results/',feature,'_shift',shift,'_betas.csv',sep=''))
r <-  read.csv(paste('path/regression/combined_results/',feature,'_shift',shift,'_rs.csv',sep=''))
cons <-  read.csv(paste('path/regression/combined_results/',feature,'_shift',shift,'_consistencies.csv',sep=''))
pval <-  read.csv(paste('path/regression/combined_results/',feature,'_shift',shift,'_pvals.csv',sep=''))

# Give clean names for the predictors
names <- c('(A) Intensity / roughness, D','(A) Spectral properties','Plesant situation','Gaze on object','(V) Movement','(V) Luminance / entropy','Scene cut','Talking','Gaze on body','Gaze on face','Gaze on background','(A) Intensity / roughness','Body movement','Unpleasant situation','(A) Spectral properties, D','(V) Luminance / entropy, D')
beta$Row <- names
r$Row <- names
cons$Row <- names
pval$Row <- names

# Sort by custom order
custom_order <- c('Talking','Plesant situation','Body movement','Unpleasant situation','(A) Spectral properties, D','(A) Spectral properties','(V) Movement','(A) Intensity / roughness, D','(V) Luminance / entropy, D','(A) Intensity / roughness','(V) Luminance / entropy','Scene cut','Gaze on object','Gaze on background','Gaze on body','Gaze on face')
custom_order_factor <- factor(beta$Row, levels = custom_order)
beta <- beta[order(custom_order_factor),]
r <- r[order(custom_order_factor),]
cons <- cons[order(custom_order_factor),]
pval <- pval[order(custom_order_factor),]

# Plot p-value as text over the heatmap
pval[pval>0.05] <- 1
pvalStr <- apply(round(as.matrix(pval[,2:ncol(pval)]),3),c(1,2),as.character)
pvalStr[pvalStr== "1"] <- ""
pvalStr[pvalStr== "0"] <- "<0.001"

# Make data matix for plotting
data <- as.matrix(beta[,2:ncol(pval)])
rownames(data)<- beta$Row
colnames(data)<- c("200 ms","500 ms","1000 ms","2000 ms","4000 ms")

# We do not want to plot the 0 betas as a color (zero was assigned if the effect was inconsistent in the cross-validation)
# Add infinite value to them, to plot them as grey in the heatmap
data[as.matrix(cons[,2:ncol(pval)])==0] <- NA

#Scale based on the absolute max beta
scaling <- max(abs(min(data,na.rm = TRUE)),abs(max(data,na.rm = TRUE)))
pdf(paste('path/regression/combined_results/heatmaps/',feature,'_shift',shift,'.pdf',sep=''),height = 10,width = 8)
superheat(data,
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
          legend = FALSE)
dev.off()
