## Plot pupil time series for Experiment  2 & 3 for Supplements (Figure SI3)

# Severi Santavirta 16.10.2024

library(ggplot2)
library(matrixStats)

##--------------------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 2

df <- read.csv("path/dependent_variable_correlations/data/pupil_for_timeseries_kasky_tw1000.csv")

# Lineplot with ribbon
data_pupil = as.data.frame(rowMedians(as.matrix(df[,3:ncol(df)]),na.rm = T)) # Choose only the first 5 columns (the individual raters) ad calculate the average
colnames(data_pupil) = "med"
data_pupil$t <- seq(from = 1, to = nrow(data_pupil)) * 1
q <- rowQuantiles(as.matrix(df),probs = c(0,.1,.2,.3,.7,.8,.9,1)) # Calculate certain quantiles from the data
pdf("path/dependent_variable_correlations/plots/pupil_timeseries_kasky.pdf",height = 3,width = 10)
ggplot(data_pupil,aes(x=t,y=med)) + 
  geom_ribbon(aes(ymin=q[,2],ymax=q[,7]),fill = "#F4A582") + # 80 % within this ribbon
  geom_ribbon(aes(ymin=q[,3],ymax=q[,6]),fill="#D6604D") + # 60% raters within this ribbon
  geom_ribbon(aes(ymin=q[,4],ymax=q[,5]),fill="#B2182B") + # 40% raters within this ribbon
  geom_line(size=0.1) + 
  xlim(0,max(data_pupil$t)) +
  ylim(0.5,1.6) +
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
df <- read.csv("path/dependent_variable_correlations/data/isc_for_timeseries_kasky_tw1000.csv")

# Lineplot with ribbon
data_isc = as.data.frame(rowMedians(as.matrix(df))) # Choose only the first 5 columns (the individual raters) ad calculate the average
colnames(data_isc) = "med"
data_isc$t <- seq(from = 1, to = nrow(data_isc)) * 1
q <- rowQuantiles(as.matrix(df),probs = c(0,.1,.2,.3,.7,.8,.9,1)) # Calculate certain quantiles from the data
pdf("path/dependent_variable_correlations/plots/isc_timeseries_kasky.pdf",height = 3,width = 10)
ggplot(data_isc,aes(x=t,y=med)) + 
  geom_ribbon(aes(ymin=q[,2],ymax=q[,7]),fill = "#92C5DE") + # 80 % within this ribbon
  geom_ribbon(aes(ymin=q[,3],ymax=q[,6]),fill="#4393C3") + # 60% raters within this ribbon
  geom_ribbon(aes(ymin=q[,4],ymax=q[,5]),fill="#2166AC") + # 40% raters within this ribbon
  geom_line(size=0.1) + 
  xlim(0,max(data_isc$t)) +
  ylim(0,0.8) +
  xlab("Time (s)") +
  ylab("ISC") + 
  theme_bw() +
  theme(
    axis.title.x = element_text(size = 16),  # Increase x-axis title size
    axis.title.y = element_text(size = 16),  # Increase y-axis title size
    axis.text.x = element_text(size = 14),   # Increase x-axis tick label size
    axis.text.y = element_text(size = 14))    # Increase y-axis tick label size
dev.off()

##--------------------------------------------------------------------------------------------------------------------------------------------------------
# Experiment 3

df <- read.csv("path/dependent_variable_correlations/data/pupil_for_timeseries_conjuring_tw1000.csv")

# Lineplot with ribbon
data_pupil = as.data.frame(rowMedians(as.matrix(df)))
data_pupil$t <- seq(from = 1, to = nrow(data_pupil)) * 1
q <- rowQuantiles(as.matrix(df),probs = c(0,.1,.2,.3,.7,.8,.9,1)) # Calculate certain quantiles from the data
pdf("path/dependent_variable_correlations/plots/pupil_timeseries_conjuring.pdf",height = 3,width = 10)
ggplot(data_pupil,aes(x=t,y=med)) + 
  geom_ribbon(aes(ymin=q[,2],ymax=q[,7]),fill = "#F4A582") + # 80 % within this ribbon
  geom_ribbon(aes(ymin=q[,3],ymax=q[,6]),fill="#D6604D") + # 60% raters within this ribbon
  geom_ribbon(aes(ymin=q[,4],ymax=q[,5]),fill="#B2182B") + # 40% raters within this ribbon
  geom_line(size=0.25) + 
  xlim(0,max(data_pupil$t)) +
  ylim(0.5,1.6) +
  xlab("Time (s)") +
  ylab("Normalized pupil size") + 
  theme_bw() +
  theme(
    axis.title.x = element_text(size = 16),  # Increase x-axis title size
    axis.title.y = element_text(size = 16),  # Increase y-axis title size
    axis.text.x = element_text(size = 14),   # Increase x-axis tick label size
    axis.text.y = element_text(size = 14))   # Increase y-axis tick label size
dev.off()

## Plot ISC time series for Experiment 1
df <- read.csv("path/dependent_variable_correlations/data/isc_for_timeseries_conjuring_tw1000.csv")

# Lineplot with ribbon
data_isc <- as.data.frame(rowMedians(as.matrix(df[,3:ncol(df)]),na.rm = T))
colnames(data_isc) = "med"
data_isc$t <- seq(from = 1, to = nrow(data_isc)) * 1
q <- rowQuantiles(as.matrix(df),probs = c(0,.1,.2,.3,.7,.8,.9,1)) # Calculate certain quantiles from the data
pdf("path/dependent_variable_correlations/plots/isc_timeseries_conjuring.pdf",height = 3,width = 10)
ggplot(data_isc,aes(x=t,y=med)) + 
  geom_ribbon(aes(ymin=q[,2],ymax=q[,7]),fill = "#92C5DE") + # 80 % within this ribbon
  geom_ribbon(aes(ymin=q[,3],ymax=q[,6]),fill="#4393C3") + # 60% raters within this ribbon
  geom_ribbon(aes(ymin=q[,4],ymax=q[,5]),fill="#2166AC") + # 40% raters within this ribbon
  geom_line(size=0.1) + 
  xlim(0,max(data_isc$t)) +
  ylim(0,0.8) +
  xlab("Time (s)") +
  ylab("ISC") + 
  theme_bw() +
  theme(
    axis.title.x = element_text(size = 16),  # Increase x-axis title size
    axis.title.y = element_text(size = 16),  # Increase y-axis title size
    axis.text.x = element_text(size = 14),   # Increase x-axis tick label size
    axis.text.y = element_text(size = 14))    # Increase y-axis tick label size
dev.off()
