# Create pupillary light response plots in the Experiment 1 subjects (N=106) with the data from luminance change task.

# Severi Santavirta 16.10.2024

library(ggplot2)
library(matrixStats)

# Read data
tbl1 <- read.csv('path/pupillary_light_reflex/luminance_90.csv')
tbl2 <- read.csv('path/pupillary_light_reflex/luminance_60.csv')
tbl3 <- read.csv('path/pupillary_light_reflex/luminance_30.csv')
tbl4 <- read.csv('path/pupillary_light_reflex/luminance_0.csv')

# Delete time bins before the light change
tbl1 <- tbl1[6:100,]
tbl2 <- tbl2[6:100,]
tbl3 <- tbl3[6:100,]
tbl4 <- tbl4[6:100,]

data1 <- as.matrix(tbl1)
data2 <- as.matrix(tbl2)
data3 <- as.matrix(tbl3)
data4 <- as.matrix(tbl4)

mu1 <- rowMeans(data1)
mu2 <- rowMeans(data2)
mu3 <- rowMeans(data3)
mu4 <- rowMeans(data4)

mu_tbl <- as.data.frame(cbind(mu1,mu2,mu3,mu4))

tbl1 <- cbind(mu1,tbl1)
tbl2 <- cbind(mu2,tbl2)
tbl3 <- cbind(mu3,tbl3)
tbl4 <- cbind(mu4,tbl4)

q1 <- rowQuantiles(data1,probs = c(.25,.375,.625,.75))
q2 <- rowQuantiles(data2,probs = c(.25,.375,.625,.75))
q3 <- rowQuantiles(data3,probs = c(.25,.375,.625,.75))
q4 <- rowQuantiles(data4,probs = c(.25,.375,.625,.75))

t <- seq(0.001,3.761,by=0.040)

pdf('/Volumes/Research/Neurotiede/Severi/gigatrack/pupillary_light_reflex/pupillary_light_reflex.pdf',height = 15,width = 10)
ggplot(mu_tbl,aes(x=t)) + 
  geom_ribbon(aes(ymin=q1[,1],ymax=q1[,4]),fill = "grey95") + geom_ribbon(aes(ymin=q1[,2],ymax=q1[,3]),fill = "grey90") +
  geom_ribbon(aes(ymin=q2[,1],ymax=q2[,4]),fill = "grey65") + geom_ribbon(aes(ymin=q2[,2],ymax=q2[,3]),fill = "grey60") +
  geom_ribbon(aes(ymin=q3[,1],ymax=q3[,4]),fill = "grey45") + geom_ribbon(aes(ymin=q3[,2],ymax=q3[,3]),fill = "grey40") +
  geom_ribbon(aes(ymin=q4[,1],ymax=q4[,4]),fill = "grey25") + geom_ribbon(aes(ymin=q4[,2],ymax=q4[,3]),fill = "grey20") +
  geom_line(aes(y = as.numeric(mu1)), color = "black") + 
  geom_line(aes(y = as.numeric(mu2)), color = "black") + 
  geom_line(aes(y = as.numeric(mu3)), color = "black") + 
  geom_line(aes(y = as.numeric(mu4)), color = "white") +
  xlim(0,3) + 
  ylim(0.4,1.2) +
  theme_bw() +
  theme(axis.text.x = element_text(size = 30),  # Change size of x-axis tick labels
        axis.text.y = element_text(size = 30),
        axis.title = element_blank())  # Change size of y-axis tick labels
dev.off()

# Find the minimum values
min_1 <- t[which.min(as.numeric(mu1))]
min_2 <- t[which.min(as.numeric(mu2))]
min_3 <- t[which.min(as.numeric(mu3))]
min_4 <- t[which.min(as.numeric(mu4))]


