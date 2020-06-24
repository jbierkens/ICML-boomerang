# plot by number of observations
library(ggplot2)
filename_csv = "results-by-observations-in-2-dimensions-refresh-0.1.csv"
df = read.csv(filename_csv)


# removing Factorized Boomerang and diagonal boomerang
df = df[df$sampler != "Factorized Boomerang",]
df = df[df$sampler != "Boomerang (diagonal)",]
df = df[df$sampler != "MALA",]

# changing order of the factors
df$sampler<- factor(df$sampler, c("Boomerang", "BPS", "ZigZag", "Boomerang w/subsampling", "BPS w/subsampling",
                                           "ZigZag w/subsampling"))

#setting transparencies of the boxplots
df$alpha1 = ifelse(df$sampler == "Boomerang"  | df$sampler == "Boomerang w/subsampling",
                   1,0)

df$a
df$observations <- as.factor(df$observations)
p1 <- ggplot(data = subset(df, !is.na(avg_ess_per_sec)), aes(x=observations, y=avg_ess_per_sec,fill=sampler)) +
  geom_boxplot(outlier.alpha =  1, aes(alpha =alpha1)) + labs(x = "number of observations" ,y="average ESS per second") + scale_y_log10() +
  #scale_x_discrete(limits = rev(levels(df$observations))) + 
  theme_minimal() +
  #coord_flip() + 
  theme(text=element_text(size=23),
        legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(size = 25)) +
  guides(title="", fill = guide_legend(nrow=3, override.aes= list(alpha = c(rep(c(1., .17, .17), 2)))),alpha=FALSE)+
 scale_alpha_continuous(range=c(0.17,1)) +
  ggsave(paste(format(Sys.time(), "%y%m%d"),"-plot_observations-average-ess.pdf",sep=''), width=9,height=9)
#ggsave("plot.png",width=8,height=12)

show(p1)

