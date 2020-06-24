# plot by number of dimensions
library(ggplot2)
filename_csv = "results-by-dimension-with-1000-observations.csv"
df = read.csv(filename_csv)



#removing diagonal boomerang
df = df[df$sampler != "Boomerang (diagonal)",]

#setting transparencies
df$alpha1 = ifelse(df$sampler == "Boomerang", 1, 0.1)  
  
df$dimension <- as.factor(df$dimension)
p1 <- ggplot(data = df, aes(x=dimension, y=avg_ess_per_sec, fill=sampler)) +
  geom_boxplot(outlier.alpha =  1, aes(alpha =alpha1)) + labs(x = "number of dimensions" ,y="average ESS per second") + scale_y_log10() +
  #scale_x_discrete(limits = rev(levels(df$observations))) + 
  theme_minimal() +
  #coord_flip() + 
  theme(text=element_text(size=23),
        legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(size = 25)) +
  guides(title="", fill = guide_legend(nrow=1,  override.aes= list(alpha = c(1.0, 0.2, 0.2, 0.2))), alpha=FALSE)+
  
  scale_alpha_continuous(range=c(0.17,1)) +
  ggsave("plot_dimensions_avg_ess.pdf", width=9,height=7)

show(p1)
