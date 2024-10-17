# Automatic video segmentation quality control in Experiment 1 data

# Severi Santavirta 16.10.2024

library(superheat)

res <- read.csv('path/video_segmentation/segmentation/segmentation_qc/segmentation_qc_results_final.csv')

labels <- c('Eyes','Mouth','Face','Body','Animal','Object','Background','Unknown')

# Plot confusion matrix
instances <- res[,11:18];
rownames(instances) <- labels
colnames(instances) <- labels
ppv <- res$ppv

# Calculate prediction ratios to get better colors for the matrix
total <- res$total_predictions
ratios <- as.matrix(instances)
for(i in seq(from=1,to=nrow(instances))){
  ratios[i,] <- ratios[i,]/total[i]
}
rownames(ratios) <- labels
colnames(ratios) <- labels

# Plot with legend
pdf('path/video_segmentation/segmentation/segmentation_qc/plots/confmat_wlegend.pdf',width = 10,height = 10)
superheat(ratios,bottom.label.text.angle = 45,heat.pal = rev(c("#B2182B","white")),left.label.col = "white",bottom.label.col = "white",X.text = as.matrix(round(instances,2)))
dev.off()

# Plot without legend
pdf('path/video_segmentation/segmentation/segmentation_qc/plots/confmat.pdf',width = 10,height = 10)
superheat(ratios,
          heat.pal = rev(c("#B2182B","white")),
          left.label = "none",
          bottom.label = "none",
          left.label.size = 1.5,
          X.text = as.matrix(round(instances,2)),
          X.text.size = 15,
          legend = F,)
dev.off()






          