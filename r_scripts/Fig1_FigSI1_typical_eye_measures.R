# Plot typical eye-tracking measure distributions while watching movies

# Severi Santavirta 16.10.2024

library(ggplot2)
library(dplyr)

data_sec <- read.csv('path/typical_eye_measures/second_data.csv')
fix_dur <- read.csv('path/typical_eye_measures/fixation_durations.csv')
sac_dur <- read.csv('path/typical_eye_measures/saccade_durations.csv')
blink_dur <- read.csv('path/typical_eye_measures/blink_durations.csv')

##----------------------------------------------------------------------------------------------------------------------------------------------
# Descriptive statistics for the distributions

# Median
medianFixationDuration <- median(fix_dur$fixation_durations)
medianFixationRate <- median(data_sec$fixation_per_second)
medianFixationTotalTime <- median(data_sec$fixations_total_time)
medianSaccadeDuration <- median(sac_dur$saccade_durations)
medianSaccadeRate <- median(data_sec$saccades_per_second)
medianSaccadeTotalTime <- median(data_sec$saccades_total_time)
medianBlinkDuration <- median(blink_dur$blink_durations)
medianBlinkRate <- median(data_sec$blinks_per_second)
medianBlinkTotalTime <- median(data_sec$blinks_total_time)

# Percentage of durations under some time point
quantileFixationDuration <- quantile(fix_dur$fixation_durations,c(0.05,0.95))
quantileFixationRate <- quantile(data_sec$fixation_per_second,c(0.05,0.95))
quantileFixationTotalTime <- quantile(data_sec$fixations_total_time,c(0.05,0.95))
quantileSaccadeDuration <- quantile(sac_dur$saccade_durations,c(0.05,0.95))
quantileSaccadeRate <- quantile(data_sec$saccades_per_second,c(0.05,0.95))
quantileSaccadeTotalTime <- quantile(data_sec$saccades_total_time,c(0.05,0.95))
quantileBlinkDuration <- quantile(blink_dur$blink_durations,c(0.05,0.95))
quantileBlinkRate <- quantile(data_sec$blinks_per_second,c(0.05,0.95))
quantileBlinkTotalTime <- quantile(data_sec$blinks_total_time,c(0.05,0.95))

##----------------------------------------------------------------------------------------------------------------------------------------------
#Create plots for Figure 1

# Plot fixations per second for each subject
pdf('path/typical_eye_measures/histograms/fixations_per_second.pdf',width = 4,height = 2)
ggplot(data_sec,aes(x=fixation_per_second)) +
  geom_histogram(bins = 30,color="black", fill="#F8766D") +
  theme_minimal() +
  xlab('Fixations per second') +
  ylab('Count (subjects)')
dev.off()

# Plot saccades per second for each subject
pdf('path/typical_eye_measures/histograms/saccades_per_second.pdf',width = 4,height = 2)
ggplot(data_sec,aes(x=saccades_per_second)) +
  geom_histogram(bins = 30,color="black", fill="#619CFF") +
  theme_minimal() +
  xlab('Saccades per second') +
  ylab('Count (subjects)')
dev.off()

# Plot blinks per second for each subject
pdf('path/typical_eye_measures/histograms/blinks_per_second.pdf',width = 4,height = 2)
ggplot(data_sec,aes(x=blinks_per_second)) +
  geom_histogram(bins = 30,color="black", fill="#00BA38") +
  theme_minimal() +
  xlab('Blinks per second') +
  ylab('Count (subjects)')
dev.off()

# Plot total fixation time for each subject
pdf('path/typical_eye_measures/histograms/fixations_total_time.pdf',width = 4,height = 2)
ggplot(data_sec,aes(x=fixations_total_time)) +
  geom_histogram(bins = 30,color="black", fill="#F8766D") +
  theme_minimal() +
  xlab('Fixations, total time (proportion)') +
  ylab('Count (subjects)')
dev.off()

# Plot total saccade time for each subject
pdf('path/typical_eye_measures/histograms/saccades_total_time.pdf',width = 4,height = 2)
ggplot(data_sec,aes(x=saccades_total_time)) +
  geom_histogram(bins =30,color="black", fill="#619CFF") +
  theme_minimal() +
  xlab('Saccades + blinks, total time (proportion)') +
  ylab('Count (subjects)')
dev.off()

# Plot total blink time for each subject
pdf('path/typical_eye_measures/histograms/blinks_total_time.pdf',width = 4,height = 2)
ggplot(data_sec,aes(x=blinks_total_time)) +
  geom_histogram(bins = 30,color="black", fill="#00BA38") +
  theme_minimal() +
  xlab('Blinks, total time (proportion)') +
  ylab('Count (subjects)')
dev.off()

# Plot the fixation durations over all subjects
# Plot only 99% percentile to exclude few outliers
prctile = quantile(fix_dur$fixation_durations,c(0.005,0.995))
fix_dur_prctile <- fix_dur[(fix_dur$fixation_durations>prctile[1] & fix_dur$fixation_durations<prctile[2]),]
pdf('path/typical_eye_measures/histograms/fixation_durations.pdf',width = 4,height = 2)
ggplot(fix_dur_prctile,aes(x=fixation_durations)) +
  geom_histogram(bins = 55,color="black", fill="#F8766D") +
  theme_minimal() +
  xlab('Fixation duration (ms)') +
  ylab('Count')
dev.off()

# Plot the saccade durations over all subjects
# Plot only 99% percentile to exclude few outliers
prctile = quantile(sac_dur$saccade_durations,c(0.005,0.995))
sac_dur_prctile <- sac_dur[(sac_dur$saccade_durations>prctile[1] & sac_dur$saccade_durations<prctile[2]),]
pdf('path/typical_eye_measures/histograms/saccade_durations.pdf',width = 4,height = 2)
ggplot(sac_dur_prctile,aes(x=saccade_durations)) +
  geom_histogram(bins = 55,color="black", fill="#619CFF") +
  theme_minimal() +
  xlab('Saccade duration (ms)') +
  ylab('Count')
dev.off()

# Plot the blink durations over all subjects
# Plot only 99% percentile to exclude few outliers
prctile = quantile(blink_dur$blink_durations,c(0.005,0.995))
blink_dur_prctile <- blink_dur[(blink_dur$blink_durations>prctile[1] & blink_dur$blink_durations<prctile[2]),]
pdf('path/typical_eye_measures/histograms/blink_durations.pdf',width = 4,height = 2)
ggplot(blink_dur_prctile,aes(x=blink_durations)) +
  geom_histogram(bins = 55,color="black", fill="#00BA38") +
  theme_minimal() + 
  xlab('Blink duration (ms)') +
  ylab('Count')
dev.off()

##------------------------------------------------------------------------------------------------------------------------------------------
# Supplementary figure SI1, plot dataset specific plots

# Violins of fixation durations for each dataset separately
fix_dur$dataset <- factor(fix_dur$dataset, levels = c("localizer", "kasky", "conjuring"))
# Exclude the 1% quantile outliers (0.5% from both ends) for each dataset
filtered_data <- fix_dur %>%
  group_by(dataset) %>%
  filter(
    fixation_durations > quantile(fixation_durations, 0.005) &
      fixation_durations < quantile(fixation_durations, 0.995)
  )
# Create the violin plot with ordered datasets and custom x-axis labels
pdf('path/typical_eye_measures/histograms/fixation_durations_dataset.pdf',width = 4,height = 4)
ggplot(filtered_data, aes(x = dataset, y = fixation_durations, fill = dataset)) +
  geom_violin(trim = FALSE, color = "black") +
  scale_fill_manual(values = c("localizer" = "#F8766D", "kasky" = "#00BA38", "conjuring" = "#619CFF")) +
  scale_x_discrete(labels = c("localizer" = "Experiment 1", "kasky" = "Experiment 2", "conjuring" = "Experiment 3")) +
  labs(title = "Fixation durations by dataset",
       x = NULL,
       y = "Fixation durations (ms)") +
  theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5))
dev.off()

# Violins of saccade durations for each dataset separately
sac_dur$dataset <- factor(sac_dur$dataset, levels = c("localizer", "kasky", "conjuring"))
# Exclude the 1% quantile outliers (0.5% from both ends) for each dataset
filtered_data <- sac_dur %>%
  group_by(dataset) %>%
  filter(
    saccade_durations > quantile(saccade_durations, 0.005) &
      saccade_durations < quantile(saccade_durations, 0.995)
  )
# Create the violin plot with ordered datasets and custom x-axis labels
pdf('path/typical_eye_measures/histograms/saccade_durations_dataset.pdf',width = 4,height = 4)
ggplot(filtered_data, aes(x = dataset, y = saccade_durations, fill = dataset)) +
  geom_violin(trim = FALSE, color = "black") +
  scale_fill_manual(values = c("localizer" = "#F8766D", "kasky" = "#00BA38", "conjuring" = "#619CFF")) +
  scale_x_discrete(labels = c("localizer" = "Experiment 1", "kasky" = "Experiment 2", "conjuring" = "Experiment 3")) +
  labs(title = "Saccade durations by dataset",
       x = NULL,
       y = "Saccade durations (ms)") +
  theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5))
dev.off()

# Violins of blink durations for each dataset separately
blink_dur$dataset <- factor(blink_dur$dataset, levels = c("localizer", "kasky", "conjuring"))
# Exclude the 1% quantile outliers (0.5% from both ends) for each dataset
filtered_data <- blink_dur %>%
  group_by(dataset) %>%
  filter(
    blink_durations > quantile(blink_durations, 0.005) &
      blink_durations < quantile(blink_durations, 0.995)
  )
# Create the violin plot with ordered datasets and custom x-axis labels
pdf('path/typical_eye_measures/histograms/blink_durations_dataset.pdf',width = 4,height = 4)
ggplot(filtered_data, aes(x = dataset, y = blink_durations, fill = dataset)) +
  geom_violin(trim = FALSE, color = "black") +
  scale_fill_manual(values = c("localizer" = "#F8766D", "kasky" = "#00BA38", "conjuring" = "#619CFF")) +
  scale_x_discrete(labels = c("localizer" = "Experiment 1", "kasky" = "Experiment 2", "conjuring" = "Experiment 3")) +
  labs(title = "Blink durations by dataset",
       x = NULL,
       y = "Blink durations (ms)") +
  theme_minimal() +
  theme(legend.position = "none",
        plot.title = element_text(hjust = 0.5))
dev.off()



