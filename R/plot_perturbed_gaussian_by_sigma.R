# plot by number of dimensions
library(ggplot2)
library(tidyverse)
library(forcats)
library(scales)
# library(ggsci)

filename_csv = "./data-by-sigma.csv"
df = read.csv(filename_csv)
df$dimension <- as.factor(df$dimension)
df$σ_squared <- as.factor(round(df$σ_squared,digits=1))
df$ess_log_density_per_sec <- df$ess_log_density/df$runtime   

#[Not used] filer a specific factor 
dim_1 = df %>%  filter(sampler == "BPS") %>%  filter(dimension == 1) %>% select(ess_log_density_per_sec)
ave_dim_1 = mean(dim_1$ess_log_density_per_sec)

dim_10 = df %>%  filter(sampler == "BPS") %>%  filter(dimension == 10) %>% select(ess_log_density_per_sec)
ave_dim_10 = mean(dim_10$ess_log_density_per_sec)

dim_100 = df %>%  filter(sampler == "BPS") %>%  filter(dimension == 100) %>% select(ess_log_density_per_sec)
ave_dim_100 = mean(dim_100$ess_log_density_per_sec)
                  
p1 <- df  %>% filter(sampler == "Boomerang") %>% ggplot(aes(y=ess_log_density_per_sec, x = σ_squared, fill = dimension)) + 
  geom_boxplot()+ scale_y_log10() +
  theme_minimal() +
  theme(text=element_text(size=23),
        legend.position = "bottom",
        legend.text = element_text(size = 25)) +
  scale_fill_manual(values=c("#009E73", "#E69F00", "#56B4E9")) +
  geom_hline(yintercept=ave_dim_1, linetype="dashed", color="#009E73") +
  geom_hline(yintercept=ave_dim_10, linetype="dashed", color="#E69F00") +
  geom_hline(yintercept=ave_dim_100, linetype="dashed", color="#56B4E9") +

  xlab("sigma^2") + ylab("|x|^2 ESS per second") +
  ggsave("gauss_perturbed_by_sigma.pdf", width=9,height=9) 
show(p1)  
