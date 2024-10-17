# Plot gaze prediction results

# Severi Santavirta 17.10.2024

library(ggplot2)
library(viridis)


dset <- c('localizer','kasky','conjuring')
predictors <- c('(A) Intensity / roughness, D','(A) Spectral properties','Pleasant situation','Object','(V) Movement','(V) Luminance / entropy','Scene cut','Talking','Body parts','Eyes','Mouth','Face (other)','Background','(A) Intensity / roughness','Body movement','Unpleasant situation','(A) Spectral properties, D','(V) Luminance / entropy, D')
predictors_clean <- c('A_Intensity_roughness_D','A_Spectral_properties','Pleasant_situation','Object','V_Movement','V_Luminance_entropy','Scene_cut','Talking','Body_parts','Eyes','Mouth','Face_other','Background','A_Intensity_roughness','Body_movement','Unpleasant_situation','A_Spectral_properties_D','V_Luminance_entropy_D')

dummy <- c(4,7,9,10,11,12,13)
input <- 'path/gaze_prediction/interpretation'

##-----------------------------------------------------------------------------------------------------------------------------------------------

# Barplot predictor importance
importance <- read.csv('path/gaze_prediction/interpretation/predictor_importance.csv')

# Transform to long
for(i in seq(from=1,to=ncol(importance))){
  values <- as.numeric(importance[,i])
  values_mean <- rep(mean(values),3)
  names <- rep(predictors[i],3)
  tbl <- as.data.frame(cbind(values,values_mean,names,dset))
  colnames(tbl) <- c('importance','mu_importance','predictor','dataset')
  if(i==1){
    df <- tbl
  }else{
    df <- rbind(df,tbl)
  }
}
df$importance <- as.numeric(df$importance)
df$mu_importance <- as.numeric(df$mu_importance)

# Factorize
df$predictor <- factor(df$predictor, levels = unique(df$predictor))

# Reorder 'predictor' based on 'mu_importance'
df$predictor <- reorder(df$predictor, df$mu_importance, FUN = median)
df$predictor <- factor(df$predictor, levels = rev(levels(df$predictor)))

# Plotting
pdf('path/gaze_prediction/interpretation/plots/importance_bars.pdf',height = 4,width = 12)
ggplot(df, aes(x = predictor, y = importance, fill = dataset)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  scale_fill_manual(values = c("conjuring" = "#2166AC", "kasky" = "#00BA38", "localizer" = "#B2182B")) +
  theme_minimal() +
  ylab("Relative importance") +
  xlab(NULL) + 
  theme(legend.position="none") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1, size = 14),
        axis.title.y = element_text(size = 14))
dev.off()

# Read interpreatation results and plot main effects for each dataset separately
results_localizer <- list()
results_kasky <- list()
results_conjuring <- list()
for(d in seq(from = 1, to = 3)){
  for(i in seq(from = 1, to = length(predictors))){
    if(d==1){
      results_localizer[[i]] <- read.csv(paste(input,'/',dset[d],'_predictor_',i,'_influence.csv',sep = ''))
      data <- results_localizer[[i]]
    }else if(d==2){
      results_kasky[[i]] <- read.csv(paste(input,'/',dset[d],'_predictor_',i,'_influence.csv',sep = ''))
      data <- results_kasky[[i]]
    }else{
      results_conjuring[[i]] <- read.csv(paste(input,'/',dset[d],'_predictor_',i,'_influence.csv',sep = ''))
      data <- results_conjuring[[i]]
    }

    
    # Store plots
    # For dummy predictors create a violin plot
    if (i %in% dummy){ 
      
      viridis_colors <- viridis(100)
      p <- ggplot(data, aes(x = factor(x), y = y, fill = factor(x))) +
        geom_violin(adjust = 2) +
        scale_fill_manual(values = c("0" = "#2166AC", "1" = "#B2182B")) +
        theme_minimal() +
        coord_cartesian(ylim = c(-0.25, 0.25)) +
        ylab('Absolute prediction change') +
        scale_x_discrete(labels = c("0" = "Not present", "1" = "Present")) +
        ggtitle(predictors[i]) +
        theme(axis.text.x = element_text(size = 14),
          plot.title = element_text(hjust = 0.5),  # Center the title
          legend.position = "none",  # Remove legend
          axis.title.x = element_blank()  # Remove x-axis label
        )
    }else{ # For continuous predictors create a density plot
      
      p <- ggplot(data,aes(x=x,y=y)) +
        geom_hex(bins = 500, aes(fill = ..density.., alpha = ..density..),show.legend = FALSE) +
        scale_fill_gradientn(colors = c("#2166AC", "#00BA38", "yellow", "#B2182B")) +
        scale_alpha_continuous(range = c(0.01,50)) + # Adjust the range as needed
        theme_minimal() +
        coord_cartesian(xlim = c(-3, 3), ylim = c(-0.05, 0.05)) +
        xlab('Z-score') +
        ylab('Absolute prediction change') +
        ggtitle(predictors[i]) +
        theme(plot.title = element_text(hjust = 0.5))  # Center the title
    }
    
    # Save the pdf
    pdf(paste('path/gaze_prediction/interpretation/plots/main_effect_',dset[d],'_',predictors_clean[i],'.pdf',sep = ''),height =  5, width = 3)
    print(p)
    dev.off()
  }
}

# Plot also main effects with combined datasets
for(i in seq(from = 1, to = length(predictors))){
    data <- rbind(results_localizer[[i]],results_kasky[[i]],results_conjuring[[i]])
  
    # Store plots
    # For dummy predictors create a violin plot
    if (i %in% dummy){ 
      
      
      viridis_colors <- viridis(100)
      p <- ggplot(data, aes(x = factor(x), y = y, fill = factor(x))) +
        geom_violin(adjust = 2) +
        scale_fill_manual(values = c("0" = "#2166AC", "1" = "#B2182B")) +
        theme_minimal() +
        coord_cartesian(ylim = c(-0.25, 0.25)) +
        ylab('Absolute prediction change') +
        scale_x_discrete(labels = c("0" = "Not present", "1" = "Present")) +
        ggtitle(predictors[i]) +
        theme(axis.text.x = element_text(size = 14),
              axis.title.y = element_text(size = 14),
              plot.title = element_text(hjust = 0.5,size = 14),  # Center the title
              legend.position = "none",  # Remove legend
              axis.title.x = element_blank()  # Remove x-axis label
        )
    }else{ # For continuous predictors create a density plot
      
      p <- ggplot(data,aes(x=x,y=y)) +
        geom_hex(bins = 500, aes(fill = ..density.., alpha = ..density..),show.legend = FALSE) +
        scale_fill_gradientn(colors = c("#2166AC", "#00BA38", "yellow", "#B2182B")) +
        scale_alpha_continuous(range = c(0.01,50)) + # Adjust the range as needed
        theme_minimal() +
        coord_cartesian(xlim = c(-3, 3), ylim = c(-0.05, 0.05)) +
        xlab('Z-score') +
        ylab('Absolute prediction change') +
        ggtitle(predictors[i]) +
        theme(plot.title = element_text(hjust = 0.5,size = 14),
              axis.title.y = element_text(size = 14),
              axis.title.x = element_text(size = 14),)  # Center the title
    }
  
  # Save the pdf
  pdf(paste('path/gaze_prediction/interpretation/plots/main_effect_combined_',predictors_clean[i],'.pdf',sep = ''),height =  5, width = 3)
  print(p)
  dev.off()
}

##-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

library(ggplot2)

files <- dir("path/gaze_prediction/interpretation/social_interactions")
scale <- c(-0.03,0.03)

# Plot the interactions for each dataset separately
require(lattice)
for(i in seq(from=1,to=length(files))){
  tbl <- read.csv(paste("path/gaze_prediction/interpretation/social_interactions/",files[i],sep = ""))
  
  # If there is eyes or mouth (dummy variables) in the interaction, we plot a different plot than for two continuous variables
  if(grepl("Eyes|Mouth", files[i])){
    
    # Plot the effect when gazing eyes/mouth
    tbl1 <- tbl[tbl[,2]==1,]
    name <- gsub("\\.csv$", "", files[i])
    pdf(paste("path/gaze_prediction/interpretation/plots/social_interactions/",name,"_1.pdf",sep = ""),height = 5,width=5)
    p <- ggplot(tbl1,aes(x=tbl1[,1],y=y)) +
      geom_hex(bins = 500, aes(fill = ..density.., alpha = ..density..),show.legend = FALSE) +
      scale_fill_viridis_c() +
      scale_alpha_continuous(range = c(0.01,50)) + # Adjust the range as needed
      theme_minimal() +
      coord_cartesian(xlim = c(-3, 3), ylim = c(-0.3, 0.3)) +
      xlab('Z-score') +
      ylab('Absolute prediction change') +
      ggtitle(paste("Gazing ",colnames(tbl1)[2],sep="")) +
      theme(plot.title = element_text(hjust = 0.5))  # Center the title
    print(p)
    dev.off()
    
    # Plot the effect when NOT gazing eyes/mouth
    tbl1 <- tbl[tbl[,2]==0,]
    pdf(paste("path/gaze_prediction/interpretation/plots/social_interactions/",name,"_0.pdf",sep = ""),height = 5,width=5)
    p <- ggplot(tbl1,aes(x=tbl1[,1],y=y)) +
      geom_hex(bins = 500, aes(fill = ..density.., alpha = ..density..),show.legend = FALSE) +
      scale_fill_viridis_c() +
      scale_alpha_continuous(range = c(0.01,50)) + # Adjust the range as needed
      theme_minimal() +
      coord_cartesian(xlim = c(-3, 3), ylim = c(-0.3, 0.3)) +
      xlab('Z-score') +
      ylab('Absolute prediction change') +
      ggtitle(paste("Not Gazing ",colnames(tbl1)[2],sep="")) +
      theme(plot.title = element_text(hjust = 0.5))  # Center the title
    print(p)
    dev.off()
    
  }else{ # Plot the continuous interaction as tile heatmap
    
    # Some tiles may be out of the chosen scale, correct those to the boundary
    tbl$y[tbl$y<scale[1]] <- scale[1]
    tbl$y[tbl$y>scale[2]] <- scale[2] 
    
    # Create the 2D density plot
    name <- gsub("\\.csv$", "", files[i])
    pdf(paste("path/gaze_prediction/interpretation/plots/social_interactions/",name,".pdf",sep = ""),height = 5,width=5)
    p <- ggplot(tbl, aes(x = tbl[,2], y = tbl[,1])) +
      stat_summary_2d(aes(z = y), bins = 50, fun = mean) +
      scale_fill_gradient2(low = "blue", mid = "white", high = "red",midpoint = 0,limits = c(scale[1],scale[2])) +
      labs(x = colnames(tbl)[2], y = colnames(tbl)[1], fill = "Average y") +
      guides(fill = "none") +  # Remove the legend
      theme_minimal()
    print(p)
    dev.off()
  }
}

# Plot the interactions for combined datasets
files <- list.files("path/gaze_prediction/interpretation/social_interactions/", "localizer", full.names = TRUE)
interaction_names <- gsub("path/gaze_prediction/interpretation/social_interactions//localizer_|\\.csv","",files)
dummy <- c(1,3,5,7,9,11,13,15)
require(lattice)
for(i in seq(from = 1, to = length(interaction_names))){
  
  results_localizer <- read.csv(paste("path/gaze_prediction/interpretation/social_interactions/localizer_",interaction_names[i],".csv",sep = ''))
  results_kasky <- read.csv(paste("path/gaze_prediction/interpretation/social_interactions/kasky_",interaction_names[i],".csv",sep = ''))
  results_conjuring <- read.csv(paste("path/gaze_prediction/interpretation/social_interactions/conjuring_",interaction_names[i],".csv",sep = ''))
  tbl <- rbind(results_localizer,results_kasky,results_conjuring)
  
  # If there is eyes or mouth (dummy variables) in the interaction, we plot a different plot than for two continuous variables
  if (i %in% dummy){ 
    
    # Plot the effect when gazing eyes/mouth
    tbl1 <- tbl[tbl[,2]==1,]
    pdf(paste("path/gaze_prediction/interpretation/plots/social_interactions/combined_",interaction_names[i],"_1.pdf",sep = ""),height = 5,width=5)
    p <- ggplot(tbl1,aes(x=tbl1[,1],y=y)) +
      geom_hex(bins = 500, aes(fill = ..density.., alpha = ..density..),show.legend = FALSE) +
      scale_fill_viridis_c() +
      scale_alpha_continuous(range = c(0.01,50)) + # Adjust the range as needed
      theme_minimal() +
      coord_cartesian(xlim = c(-3, 3), ylim = c(-0.3, 0.3)) +
      xlab('Z-score') +
      ylab('Absolute prediction change') +
      ggtitle(paste("Gazing ",colnames(tbl1)[2],sep="")) +
      theme(plot.title = element_text(hjust = 0.5))  # Center the title
    print(p)
    dev.off()
    
    # Plot the effect when NOT gazing eyes/mouth
    tbl1 <- tbl[tbl[,2]==0,]
    pdf(paste("path/gaze_prediction/interpretation/plots/social_interactions/combined_",interaction_names[i],"_0.pdf",sep = ""),height = 5,width=5)
    p <- ggplot(tbl1,aes(x=tbl1[,1],y=y)) +
      geom_hex(bins = 500, aes(fill = ..density.., alpha = ..density..),show.legend = FALSE) +
      scale_fill_viridis_c() +
      scale_alpha_continuous(range = c(0.01,50)) + # Adjust the range as needed
      theme_minimal() +
      coord_cartesian(xlim = c(-3, 3), ylim = c(-0.3, 0.3)) +
      xlab('Z-score') +
      ylab('Absolute prediction change') +
      ggtitle(paste("Not Gazing ",colnames(tbl1)[2],sep="")) +
      theme(plot.title = element_text(hjust = 0.5))  # Center the title
    print(p)
    dev.off()
    
  }else{ # Plot the continuous interaction as tile heatmap
    
    # Some tiles may be out of the chosen scale, correct those to the boundary
    tbl$y[tbl$y<scale[1]] <- scale[1]
    tbl$y[tbl$y>scale[2]] <- scale[2] 
    
    # Create the 2D density plot
    name <- predictors_clean[i]
    pdf(paste("path/gaze_prediction/interpretation/plots/social_interactions/combined_",interaction_names[i],".pdf",sep = ""),height = 5,width=5)
    p <- ggplot(tbl, aes(x = tbl[,2], y = tbl[,1])) +
      stat_summary_2d(aes(z = y), bins = 50, fun = mean) +
      scale_fill_gradient2(low = "blue", mid = "white", high = "red",midpoint = 0,limits = c(scale[1],scale[2])) +
      labs(x = colnames(tbl)[2], y = colnames(tbl)[1], fill = "Average y") +
      guides(fill = "none") +  # Remove the legend
      theme_minimal()
    print(p)
    dev.off()
  }
}


  

