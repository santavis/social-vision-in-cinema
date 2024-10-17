## Plot gaze prediction model evaluations

# Severi Santavirta 17.10.2024

library(superheat)

# Read evaluation results
cormat <- read.csv('path/gaze_prediction/evaluation/correlation_matrix.csv')
rownames(cormat) <- colnames(cormat)
peak_dist <- read.csv('path/gaze_prediction/evaluation/peak_distance_matrix.csv')
rownames(peak_dist) <- colnames(peak_dist)

# Reverse order for better visualization
reversed_cormat <- cormat[nrow(cormat):1, ncol(cormat):1]
reversed_cormat <- reversed_cormat[, ncol(reversed_cormat):1]

# Plot the heatmaps of the evaluations
cormat_str <- apply(round(as.matrix(reversed_cormat),2),c(1,2),as.character)
pdf('path/gaze_prediction/evaluation/plots/correlation_matrix.pdf',height = 8,width = 8)
superheat(reversed_cormat,
          left.label.col = "white",
          bottom.label.col = "white",
          heat.pal = rev(c("#B2182B","#D6604D","#F4A582","white")),
          heat.pal.values = c(0,0.33,0.66,1),
          heat.lim = c(0.35,0.55),
          X.text = cormat_str,
          X.text.size = 16,
          #left.label.size = 5,
          #bottom.label.size = 5,
          left.label.text.alignment = "right",
          bottom.label.text.alignment = "right",
          bottom.label.text.angle = 90)
dev.off()

# Transform peak distance from pixels to percentages of the image width
width <- 64 # Frames were downsampled to 64x64
peak_dist <- round(peak_dist/width*100)

# Reverse order for better visualization
reversed_peak_dist <- peak_dist[nrow(peak_dist):1, ncol(peak_dist):1]
reversed_peak_dist <- reversed_peak_dist[, ncol(reversed_peak_dist):1]

peak_str <- apply(round(as.matrix(reversed_peak_dist),2),c(1,2),as.character)
pdf('path/gaze_prediction/evaluation/plots/peakdistance_matrix.pdf',height = 8,width = 8)
superheat(reversed_peak_dist,
          left.label.col = "white",
          bottom.label.col = "white",
          heat.pal = rev(c("white","#92C5DE","#4393C3","#2166AC")),
          heat.pal.values = c(0,0.33,0.66,1),
          heat.lim = c(9,18),
          X.text = peak_str,
          X.text.size = 16,
          #left.label.size = 5,
          #bottom.label.size = 5,
          left.label.text.alignment = "right",
          bottom.label.text.alignment = "right",
          bottom.label.text.angle = 90)
dev.off()

