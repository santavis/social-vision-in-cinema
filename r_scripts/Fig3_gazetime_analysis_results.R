# Barplot gaze time analysis results

# Severi Santavirta 17.10.2024

library(ggplot2)

dset <- c("localizer","kasky","conjuring")

require(lattice)
for(set in seq(from=1,to=3)){
  data <- read.csv(paste("path/gaze_object_detection/analysis/results_",dset[set],".csv",sep = ""))
  data <- data[-5,]
  features <- data$Row
  features <- c("Eyes","Mouth","Face, excluding eyes and mouth","Body","Object","Background","Outside video area","Unknown")
  data$Row <- features
  
  # Sort table by the difference
  new_order <- c("Eyes","Mouth","Face, excluding eyes and mouth","Object","Unknown","Body","Background","Outside video area")
  data <- data[order(factor(data$Row, levels = new_order)),]
  
  # Reformat
  class <- c()
  condition <- c()
  gazetime <- c()
  for(i in seq(from=1,to=8)){
    class <- c(class,rep(data$Row[i],2))
    condition <- c(condition,"1true_gazetime","2chance_gazetime")
    gazetime <- c(gazetime,data$Average.total.time[i],data$Average.chance.total.time[i])
  }
  data_format <- as.data.frame(class)
  data_format$condition <- condition
  data_format$times <- gazetime
  data_format$class = factor(data_format$class,levels = unique(data_format$class))
  rm("class","condition","gazetime")
  
  pdf(paste("path/gaze_object_detection/analysis/visualization/barchart_times_",dset[set],".pdf",sep = ""),height = 5,width = 12)
  print(ggplot(data_format, aes(x = class, y = times, fill = condition)) +
    geom_bar(stat = "identity", position = "dodge",color = "black") +
    ylim(c(0,0.5)) +
    theme_minimal() +
    scale_fill_manual(values = c("#F8766D","#619CFF")) +
    guides(fill = "none"))
  dev.off()
}




