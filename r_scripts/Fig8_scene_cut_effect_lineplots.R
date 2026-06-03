# Scene change effect lineplots

# Severi Santavirta 17.10.2024

library(ggplot2)
library(MASS)
library(dplyr)

dset <- c('localizer','kasky','conjuring')

# Load data
eisc_conjuring <- read.csv("path/scene_cut_effect/scene_cut_effect_eisc_conjuring.csv",header = FALSE)
eisc_kasky <- read.csv("path/scene_cut_effect/scene_cut_effect_eisc_kasky.csv",header = FALSE)
eisc_localizer <- read.csv("path/scene_cut_effect/scene_cut_effect_eisc_localizer.csv",header = FALSE)
pupil_conjuring <- read.csv("path/scene_cut_effect/scene_cut_effect_pupil_conjuring.csv",header = FALSE)
pupil_kasky <- read.csv("path/scene_cut_effect/scene_cut_effect_pupil_kasky.csv",header = FALSE)
pupil_localizer <- read.csv("path/scene_cut_effect/scene_cut_effect_pupil_localizer.csv",header = FALSE)
blink_conjuring <- read.csv("path/scene_cut_effect/scene_cut_effect_blink_conjuring.csv",header = FALSE)
blink_kasky <- read.csv("path/scene_cut_effect/scene_cut_effect_blink_kasky.csv",header = FALSE)
blink_localizer <- read.csv("path/scene_cut_effect/scene_cut_effect_blink_localizer.csv",header = FALSE)

#Load permuted random data
eisc_conjuring_random <- read.csv("path/scene_cut_effect/scene_cut_effect_eisc_random_conjuring.csv",header = FALSE)
eisc_kasky_random <- read.csv("path/scene_cut_effect/scene_cut_effect_eisc_random_kasky.csv",header = FALSE)
eisc_localizer_random <- read.csv("path/scene_cut_effect/scene_cut_effect_eisc_random_localizer.csv",header = FALSE)
pupil_conjuring_random <- read.csv("path/scene_cut_effect/scene_cut_effect_pupil_random_conjuring.csv",header = FALSE)
pupil_kasky_random <- read.csv("path/scene_cut_effect/scene_cut_effect_pupil_random_kasky.csv",header = FALSE)
pupil_localizer_random <- read.csv("path/scene_cut_effect/scene_cut_effect_pupil_random_localizer.csv",header = FALSE)
blink_conjuring_random <- read.csv("path/scene_cut_effect/scene_cut_effect_blink_random_conjuring.csv",header = FALSE)
blink_kasky_random <- read.csv("path/scene_cut_effect/scene_cut_effect_blink_random_kasky.csv",header = FALSE)
blink_localizer_random <- read.csv("path/scene_cut_effect/scene_cut_effect_blink_random_localizer.csv",header = FALSE)

# Calculate the means from the real data to get the average effect of the cut
eisc_mean <- rowMeans(t(as.matrix(eisc_conjuring)))
eisc_mean <- c(eisc_mean,rowMeans(t(as.matrix(eisc_kasky))))
eisc_mean <- c(eisc_mean,rowMeans(t(as.matrix(eisc_localizer))))
blink_mean <- rowMeans(t(as.matrix(blink_conjuring)))
blink_mean <- c(blink_mean,rowMeans(t(as.matrix(blink_kasky))))
blink_mean <- c(blink_mean,rowMeans(t(as.matrix(blink_localizer))))
pupil_mean <- rowMeans(t(as.matrix(pupil_conjuring)))
pupil_mean <- c(pupil_mean,rowMeans(t(as.matrix(pupil_kasky))))

# In localizer there is an artefact in the pupil size at the cut from 600ms to 601ms due to the way the data is read from the eye tracker reports. This is corrected.
pupil_mean_localizer <- rowMeans(t(as.matrix(pupil_localizer)))
pupil_mean_localizer[601:3600] <- pupil_mean_localizer[601:3600] - (pupil_mean_localizer[601]-pupil_mean_localizer[600])
pupil_mean <- c(pupil_mean,pupil_mean_localizer)

# Calculate the 0.025% and 0.975% intervals for random pupil size to get the two-tailed exact p-value level of p=0.005, fit also normal distributions to the random data to get the parametric p-value threshold of p=0.05
p_thr_pupil_kasky <- matrix(nrow = ncol(pupil_kasky_random),ncol = 4)
p_thr_pupil_conjuring <- matrix(nrow = ncol(pupil_kasky_random),ncol = 4)
p_thr_pupil_localizer <- matrix(nrow = ncol(pupil_kasky_random),ncol = 4)
for(t in seq(from=1,to=ncol(pupil_kasky_random))){
  
  # Fit normal distributions
  fit_kasky <- fitdistr(as.matrix(pupil_kasky_random[,t]),"normal")
  fit_conjuring <- fitdistr(as.matrix(pupil_conjuring_random[,t]),"normal")
  fit_localizer <- fitdistr(as.matrix(pupil_localizer_random[,t]),"normal")
  params_kasky <- qnorm(c(0.025,0.975), mean = fit_kasky$estimate[1], sd = fit_kasky$estimate[2])
  params_conjuring <- qnorm(c(0.025,0.975), mean = fit_conjuring$estimate[1], sd = fit_conjuring$estimate[2])
  params_localizer <- qnorm(c(0.025,0.975), mean = fit_localizer$estimate[1], sd = fit_localizer$estimate[2])

  p_thr_pupil_kasky[t,] <- c(quantile(as.matrix(pupil_kasky_random[,t]),c(.025,.975)),params_kasky) 
  p_thr_pupil_conjuring[t,] <- c(quantile(as.matrix(pupil_conjuring_random[,t]), c(.025,.975)),params_conjuring)
  p_thr_pupil_localizer[t,] <- c(quantile(as.matrix(pupil_localizer_random[,t]), c(.025,.975)),params_localizer)
}

# Calculate the 0.025% and 0.975% intervals for random eisc to get the two-tailed exact p-value level of p=0.05, fit also normal distributions to the random data to get the parametric p-value threshold of p=0.05
p_thr_eisc_kasky <- matrix(nrow = ncol(eisc_kasky_random),ncol = 4)
p_thr_eisc_conjuring <- matrix(nrow = ncol(eisc_kasky_random),ncol = 4)
p_thr_eisc_localizer <- matrix(nrow = ncol(eisc_localizer_random),ncol = 4)
for(t in seq(from=1,to=ncol(eisc_kasky_random))){
  
  # Fit normal distributions
  fit_kasky <- fitdistr(as.matrix(eisc_kasky_random[,t]),"normal")
  fit_conjuring <- fitdistr(as.matrix(eisc_conjuring_random[,t]),"normal")
  fit_localizer <- fitdistr(as.matrix(eisc_localizer_random[,t]),"normal")
  params_kasky <- qnorm(c(0.025,0.975), mean = fit_kasky$estimate[1], sd = fit_kasky$estimate[2])
  params_conjuring <- qnorm(c(0.025,0.975), mean = fit_conjuring$estimate[1], sd = fit_conjuring$estimate[2])
  params_localizer <- qnorm(c(0.025,0.975), mean = fit_localizer$estimate[1], sd = fit_localizer$estimate[2])
  
  p_thr_eisc_kasky[t,] <- c(quantile(as.matrix(eisc_kasky_random[,t]), c(.025,.975)),params_kasky) 
  p_thr_eisc_conjuring[t,] <- c(quantile(as.matrix(eisc_conjuring_random[,t]), c(.025,.975)),params_conjuring)
  p_thr_eisc_localizer[t,] <- c(quantile(as.matrix(eisc_localizer_random[,t]), c(.025,.975)),params_localizer)
}

# Calculate the 0.025% and 0.975% intervals for random blink rate to get the two-tailed exact p-value level of p=0.05, fit also normal distributions to the random data to get the parametric p-value threshold of p=0.05
p_thr_blink_kasky <- matrix(nrow = ncol(blink_kasky_random),ncol = 4)
p_thr_blink_conjuring <- matrix(nrow = ncol(blink_kasky_random),ncol = 4)
p_thr_blink_localizer <- matrix(nrow = ncol(blink_localizer_random),ncol = 4)
for(t in seq(from=1,to=ncol(blink_kasky_random))){
  
  # Fit normal distributions
  fit_kasky <- fitdistr(as.matrix(blink_kasky_random[,t]),"normal")
  fit_conjuring <- fitdistr(as.matrix(blink_conjuring_random[,t]),"normal")
  fit_localizer <- fitdistr(as.matrix(blink_localizer_random[,t]),"normal")
  params_kasky <- qnorm(c(0.025,0.975), mean = fit_kasky$estimate[1], sd = fit_kasky$estimate[2])
  params_conjuring <- qnorm(c(0.025,0.975), mean = fit_conjuring$estimate[1], sd = fit_conjuring$estimate[2])
  params_localizer <- qnorm(c(0.025,0.975), mean = fit_localizer$estimate[1], sd = fit_localizer$estimate[2])

  p_thr_blink_kasky[t,] = c(quantile(as.matrix(blink_kasky_random[,t]), c(.025,.975)),params_kasky) 
  p_thr_blink_conjuring[t,] = c(quantile(as.matrix(blink_conjuring_random[,t]), c(.025,.975)),params_conjuring)
  p_thr_blink_localizer[t,] = c(quantile(as.matrix(blink_localizer_random[,t]), c(.025,.975)),params_localizer) 
}

# Create tables from the real data
data_eisc <- as.data.frame(eisc_mean)
data_eisc$eisc_mean_norm <- data_eisc
data_eisc$dset <- c(rep('conjuring',ncol(blink_conjuring)),rep('kasky',ncol(blink_conjuring)),rep('localizer',ncol(blink_conjuring)))
data_eisc$time <- c(seq(from=0.2,to=3.6,by=0.2),seq(from=0.2,to=3.6,by=0.2),seq(from=0.2,to=3.6,by=0.2))
data_eisc$time_norm <- data_eisc$time - 0.6
data_eisc$pupper <- c(p_thr_eisc_conjuring[,2],p_thr_eisc_kasky[,2],p_thr_eisc_localizer[,2])
data_eisc$plower <- c(p_thr_eisc_conjuring[,1],p_thr_eisc_kasky[,1],p_thr_eisc_localizer[,1])
data_eisc$pupper_param <- c(p_thr_eisc_conjuring[,4],p_thr_eisc_kasky[,4],p_thr_eisc_localizer[,4])
data_eisc$plower_param <- c(p_thr_eisc_conjuring[,3],p_thr_eisc_kasky[,3],p_thr_eisc_localizer[,3])
data_pupil <- as.data.frame(pupil_mean)
data_pupil$dset <- c(rep('conjuring',ncol(pupil_conjuring)),rep('kasky',ncol(pupil_conjuring)),rep('localizer',ncol(pupil_conjuring)))
data_pupil$time <- c(0.001:ncol(pupil_conjuring)/1000,0.001:ncol(pupil_conjuring)/1000,0.001:ncol(pupil_conjuring)/1000)
data_pupil$time_norm <- data_pupil$time - 0.6
data_pupil$pupper <- c(p_thr_pupil_conjuring[,2],p_thr_pupil_kasky[,2],p_thr_pupil_localizer[,2])
data_pupil$plower <- c(p_thr_pupil_conjuring[,1],p_thr_pupil_kasky[,1],p_thr_pupil_localizer[,1])
data_pupil$pupper_param <- c(p_thr_pupil_conjuring[,4],p_thr_pupil_kasky[,4],p_thr_pupil_localizer[,4])
data_pupil$plower_param <- c(p_thr_pupil_conjuring[,3],p_thr_pupil_kasky[,3],p_thr_pupil_localizer[,3])
data_blink <- as.data.frame(blink_mean)
data_blink$dset <- c(rep('conjuring',ncol(blink_conjuring)),rep('kasky',ncol(blink_conjuring)),rep('localizer',ncol(blink_conjuring)))
data_blink$time <- c(seq(from=0.2,to=3.6,by=0.2),seq(from=0.2,to=3.6,by=0.2),seq(from=0.2,to=3.6,by=0.2))
data_blink$time_norm <- data_blink$time - 0.6
data_blink$pupper <- c(p_thr_blink_conjuring[,2],p_thr_blink_kasky[,2],p_thr_blink_localizer[,2])
data_blink$plower <- c(p_thr_blink_conjuring[,1],p_thr_blink_kasky[,1],p_thr_blink_localizer[,1])
data_blink$pupper_param <- c(p_thr_blink_conjuring[,4],p_thr_blink_kasky[,4],p_thr_blink_localizer[,4])
data_blink$plower_param <- c(p_thr_blink_conjuring[,3],p_thr_blink_kasky[,3],p_thr_blink_localizer[,3])

# Transform blink rate propeortion to percentages for better understandability
data_blink$blink_mean <- data_blink$blink_mean*100
data_blink$pupper <- data_blink$pupper*100
data_blink$pupper_param <- data_blink$pupper_param*100
data_blink$plower <- data_blink$plower*100
data_blink$plower_param <- data_blink$plower_param*100

# Add dummy column for significance
data_eisc$significance = data_eisc$time_norm > 0 & ((data_eisc$eisc_mean > data_eisc$pupper) | (data_eisc$eisc_mean < data_eisc$plower))
data_pupil$significance = data_pupil$time_norm > 0 & ((data_pupil$pupil_mean > data_pupil$pupper) | (data_pupil$pupil_mean < data_pupil$plower))
data_blink$significance = data_blink$time_norm > 0 & ((data_blink$blink_mean > data_blink$pupper) | (data_blink$blink_mean < data_blink$plower))

# Correct for the difference in average ISC which is most likely due to the difference in stimulus size
siz_localizer = 414720/786432
siz_kasky = 737280/786432
siz_conjuring = 750000/786432
data_eisc$eisc_mean_norm <- data_eisc$eisc_mean
data_eisc$eisc_mean_norm[data_eisc$dset=='localizer'] <- data_eisc$eisc_mean_norm[data_eisc$dset=='localizer']*siz_localizer
data_eisc$eisc_mean_norm[data_eisc$dset=='kasky'] <- data_eisc$eisc_mean_norm[data_eisc$dset=='kasky']*siz_kasky
data_eisc$eisc_mean_norm[data_eisc$dset=='conjuring'] <- data_eisc$eisc_mean_norm[data_eisc$dset=='conjuring']*siz_conjuring

# Lineplot eisc (significance as colour)
pdf("path/scene_cut_effect/plots/eisc_change.pdf",width = 4,height = 4)
ggplot(data_eisc, aes(x = time_norm, y = eisc_mean_norm, group = dset, color = dset)) +
  geom_line(linewidth = 1.5) +
  geom_vline(xintercept = 0.000) +
  geom_point(data = subset(data_eisc, significance == TRUE),
             aes(color = "bright"),
             size = 3) +  # Adjust size as needed
  theme_minimal() +
  ylab('Normalized ISC') +
  xlab('Time (s)') +
  theme(legend.position = "none",
        axis.title = element_text(size = 16)) +
  scale_color_manual(values = c("conjuring" = "#1f78b4", "kasky" = "#33a02c", "localizer" = "#e31a1c", "bright" = "#FFA500"))
dev.off()

# Lineplot pupil (significance as colour)
pdf("path/scene_cut_effect/plots/pupil_change.pdf",width = 4,height = 4)
ggplot(data_pupil, aes(x = time_norm, y = pupil_mean, group = dset, color = dset)) +
  geom_line(linewidth = 1.5) +
  geom_vline(xintercept = 0.000) +
  theme_minimal() +
  ylab('Normalized pupil size') +
  xlab('Time (s)') +
  theme(legend.position = "none",
        axis.title = element_text(size = 16)) +
  scale_color_manual(values = c("conjuring" = "#1f78b4", "kasky" = "#33a02c", "localizer" = "#e31a1c", "bright" = "#FFA500"))
dev.off()

# Lineplot blinks (significance as colour)
pdf("path/scene_cut_effect/plots/blink_change.pdf",width = 4,height = 4)
ggplot(data_blink, aes(x = time_norm, y = blink_mean, group = dset, color = dset)) +
  geom_line(linewidth = 1.5) +
  geom_vline(xintercept = 0.000) +
  geom_point(data = subset(data_blink, significance == TRUE),
             aes(color = "bright"),
             size = 3) +  # Adjust size as needed
  theme_minimal() +
  ylab('How many subjects blinked (%)') +
  xlab('Time (s)') +
  theme(legend.position = "none",
        axis.title = element_text(size = 16)) +
  scale_color_manual(values = c("conjuring" = "#1f78b4", "kasky" = "#33a02c", "localizer" = "#e31a1c", "bright" = "#FFA500"))
dev.off()

##-----------------------------------------------------------------------------------------------------------------------------------------
# Reporting

# Identify the peaks of maximal pupil contraction after scene cut and their time points for reporting.
data_pupil_kasky <- data_pupil[data_pupil$dset=="kasky",]
data_pupil_conjuring <- data_pupil[data_pupil$dset=="conjuring",]
data_pupil_localizer <- data_pupil[data_pupil$dset=="localizer",]

min_kasky <- min(data_pupil_kasky$pupil_mean)
min_conjuring <- min(data_pupil_conjuring$pupil_mean)
min_localizer <- min(data_pupil_localizer$pupil_mean)
min_time_kasky <- data_pupil_kasky$time_norm[which.min(data_pupil_kasky$pupil_mean)]
min_time_conjuring <- data_pupil_conjuring$time_norm[which.min(data_pupil_conjuring$pupil_mean)]
min_time_localizer <- data_pupil_localizer$time_norm[which.min(data_pupil_localizer$pupil_mean)]

# Identify time intervals of significant deviation from baseline
first_sign_deviation_pupil_kasky <- data_pupil_kasky$time_norm[head(which(data_pupil_kasky$significance), 1)]
first_sign_deviation_pupil_conjuring <- data_pupil_conjuring$time_norm[head(which(data_pupil_conjuring$significance), 1)]
first_sign_deviation_pupil_localizer <- data_pupil_localizer$time_norm[head(which(data_pupil_localizer$significance), 1)]
last_sign_deviation_pupil_kasky <- data_pupil_kasky$time_norm[tail(which(data_pupil_kasky$significance), 1)]
last_sign_deviation_pupil_conjuring <- data_pupil_conjuring$time_norm[tail(which(data_pupil_conjuring$significance), 1)]
last_sign_deviation_pupil_localizer <- data_pupil_localizer$time_norm[tail(which(data_pupil_localizer$significance), 1)]

# Identify significant deviation times for ISC
data_eisc_kasky <- data_eisc[data_eisc$dset=="kasky",]
data_eisc_conjuring <- data_eisc[data_eisc$dset=="conjuring",]
data_eisc_localizer <- data_eisc[data_eisc$dset=="localizer",]
first_sign_deviation_eisc_kasky <- data_eisc_kasky$time_norm[head(which(data_eisc_kasky$significance), 1)]
first_sign_deviation_eisc_conjuring <- data_eisc_conjuring$time_norm[head(which(data_eisc_conjuring$significance), 1)]
first_sign_deviation_eisc_localizer <- data_eisc_localizer$time_norm[head(which(data_eisc_localizer$significance), 1)]
last_sign_deviation_eisc_kasky <- data_eisc_kasky$time_norm[tail(which(data_eisc_kasky$significance), 1)]
last_sign_deviation_eisc_conjuring <- data_eisc_conjuring$time_norm[tail(which(data_eisc_conjuring$significance), 1)]
last_sign_deviation_eisc_localizer <- data_eisc_localizer$time_norm[tail(which(data_eisc_localizer$significance), 1)]

# Identify significant deviation times for blink synchronization
data_blink_kasky <- data_blink[data_blink$dset=="kasky",]
data_blink_conjuring <- data_blink[data_blink$dset=="conjuring",]
data_blink_localizer <- data_blink[data_blink$dset=="localizer",]
first_sign_deviation_blink_kasky <- data_blink_kasky$time_norm[head(which(data_blink_kasky$significance), 1)]
first_sign_deviation_blink_conjuring <- data_blink_conjuring$time_norm[head(which(data_blink_conjuring$significance), 1)]
first_sign_deviation_blink_localizer <- data_blink_localizer$time_norm[head(which(data_blink_localizer$significance), 1)]
last_sign_deviation_blink_kasky <- data_blink_kasky$time_norm[tail(which(data_blink_kasky$significance), 1)]
last_sign_deviation_blink_conjuring <- data_blink_conjuring$time_norm[tail(which(data_blink_conjuring$significance), 1)]
last_sign_deviation_blink_localizer <- data_blink_localizer$time_norm[tail(which(data_blink_localizer$significance), 1)]

# Correlation between pupil size, eisc and blinks after scene cut
cor_eisc_blink_scene_localizer <- cor(data_eisc$eisc_mean_norm[data_eisc$dset=='localizer'],data_blink$blink_mean[data_blink$dset=='localizer'])
cor_eisc_blink_scene_kasky <- cor(data_eisc$eisc_mean_norm[data_eisc$dset=='kasky'],data_blink$blink_mean[data_blink$dset=='kasky'])
cor_eisc_blink_scene_conjuring <- cor(data_eisc$eisc_mean_norm[data_eisc$dset=='conjuring'],data_blink$blink_mean[data_blink$dset=='conjuring'])


