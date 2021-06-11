---
title: K-Means Clustering of Well Log Data
author: Andy Wohlgenant
date: '2021-06-11'
slug: k-means-clustering-of-well-log-data
categories: []
tags: []
subtitle: ''
summary: ''
authors: []
lastmod: '2021-06-11T10:29:28-07:00'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---

*Photo by Matt Artz on Unsplash*

# Introduction
About a year ago, I posted about using the k-means clustering algorithm in Python for well log data from oil and gas fields. Since then, I’ve been spending some time learning R, so I wanted to post a companion piece showing how to do some of the same things using R. The general process is very similar, and I will use the same data as in the previous post. Along the way, I will demonstrate how little code it takes to do some of the same things in R.

I want to thank **Jared Lander** and his great book *R for Everyone*. The basic workflow for k-means clustering in R that I show here is pretty much straight out of his book, with some minor tweaks around displaying the results in a format that is familiar to folks who’ve worked in the oil patch.

First step, import the packages we will need. For now, all we really need are the `tidyverse` and `useful` packages.  The `useful` package is, well, useful!  It contains some handy tools, such as a function for plotting the results of K-means clustering.


```r
library(tidyverse)
library(useful)
```

# Import the data, and plot the logs
Next, I’ll read in the data using the `read_csv()` function from the tidyverse package, and I'll take look at the format of the variables and the first few values for each.

These data can be downloaded from the SEG Github: https://github.com/seg/tutorials-2016/tree/master/1610_Facies_classification.


```r
logs <- read_csv("kgs_log_data.csv")
glimpse(logs)
```

```
## Rows: 4,149
## Columns: 11
## $ Facies      <dbl> 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2~
## $ Formation   <chr> "A1 SH", "A1 SH", "A1 SH", "A1 SH", "A1 SH", "A1 SH", "A1 ~
## $ `Well Name` <chr> "SHRIMPLIN", "SHRIMPLIN", "SHRIMPLIN", "SHRIMPLIN", "SHRIM~
## $ Depth       <dbl> 2793.0, 2793.5, 2794.0, 2794.5, 2795.0, 2795.5, 2796.0, 27~
## $ GR          <dbl> 77.45, 78.26, 79.05, 86.10, 74.58, 73.97, 73.72, 75.65, 73~
## $ ILD_log10   <dbl> 0.664, 0.661, 0.658, 0.655, 0.647, 0.636, 0.630, 0.625, 0.~
## $ DeltaPHI    <dbl> 9.9, 14.2, 14.8, 13.9, 13.5, 14.0, 15.6, 16.5, 16.2, 16.9,~
## $ PHIND       <dbl> 11.915, 12.565, 13.050, 13.115, 13.300, 13.385, 13.930, 13~
## $ PE          <dbl> 4.6, 4.1, 3.6, 3.5, 3.4, 3.6, 3.7, 3.5, 3.4, 3.5, 3.6, 3.7~
## $ NM_M        <dbl> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1~
## $ RELPOS      <dbl> 1.000, 0.979, 0.957, 0.936, 0.915, 0.894, 0.872, 0.830, 0.~
```

Now I will plot the logs for one of the wells. With `ggplot`, I can use the `geom_path()` geometric object in the `ggplot` call which will connect the data points in order of increasing depth.  I’ll also reverse the y-scale so that depth is increasing downward. Let’s try this first for just one well, *SHRIMPLIN*, and one log curve, *GR*.


```r
shrimplin <- logs %>% filter(`Well Name` == "SHRIMPLIN") %>% select(Depth, GR, ILD_log10, DeltaPHI, PHIND, PE)

ggplot(shrimplin, aes(x=GR, y=Depth)) + geom_path() + theme_bw() + scale_y_reverse()
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-3-1.png" width="672" />

Now I can take advantage of the `facet_wrap()` option in ggplot to plot each curve side-by-side.  For this to work, I will need to have each curve name as a variable rather than as individual variables in separate columns.  This can be accomplished using the `pivot_longer()` function, originally from the `tidyr` package, and included with `tidyverse`.

With `pivot_longer`, I'll be making the dataframe *longer* (more rows) and losing some columns (the individual log curve columns) by combining all curve names into one column (*curve_name*) and all the curve values into one column (*curve_value*).  I'll also *free* the x-scales so that each curve can have its own scale rather than sharing one common scale.


```r
shrimplin_long <- pivot_longer(data=shrimplin, cols=2:6, names_to = "curve_name", values_to = "curve_value")

ggplot(shrimplin_long, aes(x=curve_value, y=Depth)) + 
  geom_path(aes(color=curve_name)) + 
  theme_bw() + 
  scale_y_reverse() + 
  facet_wrap(~ curve_name, nrow=1, scales = "free_x")
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-4-1.png" width="672" />

That's looking pretty good, but let's reorder the curves in the plot by setting the factor levels for the *curve* variable.  In the petroleum world, we usually plot a log on the far left that tells us something about lithology, like the *gamma-ray* (GR), and on the right some kind of *resistivity* curve (ILD) and *porosity* curve (PHIND and DeltaPHI).  We'll put the *photoelectric factor* (PE) curve on the far right.

Then we can replot the logs for the *SHRIMPLIN* well.  I'll also get rid of the legend since it's not really necessary here, and I'll add a title and a better label for the x-axis.


```r
curve_order <- c("GR", "ILD_log10", "PHIND", "DeltaPHI", "PE")
shrimplin_long$curve_name <- factor(shrimplin_long$curve_name, levels=curve_order)

ggplot(shrimplin_long, aes(x=curve_value, y=Depth)) + 
  geom_path(aes(color=curve_name)) + 
  theme_bw() + 
  scale_y_reverse() + 
  theme(legend.position = "none") +
  labs(title="Well: SHRIMPLIN", x="Curve Value") +
  facet_wrap(~ curve_name, nrow=1, scales = "free_x")
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-5-1.png" width="672" />

That looks pretty good.  Now I'll move on to the clustering.

# K-means clustering
First I'll get rid of any null values in the data set, and then I'll select only the numerical variables for clustering.  I will leave out *NM_M* and *RELPOS* for clustering, since these are more interpretive than the downhole geophysical logs.


```r
colSums(is.na(logs))
```

```
##    Facies Formation Well Name     Depth        GR ILD_log10  DeltaPHI     PHIND 
##         0         0         0         0         0         0         0         0 
##        PE      NM_M    RELPOS 
##       917         0         0
```

```r
logs_noNA <- logs %>% filter(!is.na(PE))
logs_train <- logs_noNA %>% select(GR, ILD_log10, PHIND, DeltaPHI, PE)
```

Before performing the k-means clustering, I may want to rescale the variables to avoid having variables with higher magnitude values have larger influence on the algorithm.  From our log plot above, we can see that the GR curve varies from near zero to more than 300, while the ILD curve only varies from less than 0.5 to around 1.5.  This can be accomplished using the `scale()` function from base R, which is essentially performing z-score standardization.


```r
logs_train <- logs_train %>% mutate_all(~(scale(.)))
head(logs_train)
```

```
## # A tibble: 6 x 5
##   GR[,1] ILD_log10[,1] PHIND[,1] DeltaPHI[,1] PE[,1]
##    <dbl>         <dbl>     <dbl>        <dbl>  <dbl>
## 1  0.367        0.0880   -0.204          1.21  0.976
## 2  0.393        0.0756   -0.119          2.03  0.418
## 3  0.419        0.0632   -0.0563         2.15 -0.140
## 4  0.647        0.0508   -0.0478         1.98 -0.251
## 5  0.274        0.0177   -0.0238         1.90 -0.363
## 6  0.254       -0.0278   -0.0128         2.00 -0.140
```

Now I'll use the `kmeans()` function and specify the number of clusters to generate.  I'll generate 9 clusters to compare to the 9 facies groupings that have already been defined in the data set, presumably by geologists and/or petrophysicists at the *Kansas Geological Survey*.


```r
# cluster with 9 centers
set.seed(90210)  #set.seed(13)
logs_9clust <- kmeans(x=logs_train, centers=9)
```

# Visualize the results
Next I can plot the kmeans object with help from the `plot.kmeans` function from the `useful` package.  The data will be projected into two dimensions for visualization.


```r
plot(logs_9clust, data=logs_train)
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-9-1.png" width="672" />

Now I can add the cluster labels back to the dataframe along with well name, facies, etc.


```r
logs_noNA$Cluster <- logs_9clust$cluster
```

I'll reorder the log curves again, then plot the logs again, this time with a track for the clusters.


```r
# add back to log plot with cluster?
curve_order <- c("GR", "ILD_log10", "PHIND", "DeltaPHI", "PE", "Facies", "Cluster")

# plot logs with added track for cluster
logs_noNA %>% filter(`Well Name` == "SHRIMPLIN") %>%
  select(Depth, GR, ILD_log10, PHIND, DeltaPHI, PE, Cluster, Facies) %>%
  pivot_longer(cols=2:8, names_to="curve", values_to="value") %>%
  mutate(curve = factor(curve, levels=curve_order)) %>%
  ggplot(aes(x=value, y=Depth)) + 
  geom_path(aes(color=curve)) + 
  theme_bw() + 
  theme(legend.position = "none") +
  scale_y_reverse() + 
  facet_wrap(~ curve, nrow=1, scales = "free_x") +
  labs(title = "Well: SHRIMPLIN", x = "Curve Value")
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-11-1.png" width="672" />

Now we can compare the clusters broken out by the k-means algorithm to the *Facies*.  There is no reason to expect them to match, since the Facies were most likely defined from cores, not necessarily the input geophysical curves.  But it might be interesting to see if any of the clusters identified match up with one or more facies designations.

We can visualize this with a mosaic plot.


```r
plot(table(logs_noNA$Facies, logs_noNA$Cluster),
     main="Confusion matrix", xlab="Facies", ylab="Cluster")
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-12-1.png" width="672" />

As we can see from this plot, there isn't a great one-to-one match between any of the clusters and facies.  But the results might yield insights into relationships between stratigraphic packages within the same well, as well as between wells. Let's plot just the clusters for four of the wells.


```r
logs_noNA %>%
  filter(`Well Name` %in% c("NEWBY", "NOLAN", "SHANKLE", "SHRIMPLIN")) %>%
  filter(Depth > 2800 & Depth < 3000) %>%
  select(`Well Name`, Depth, Cluster) %>%
  ggplot(aes(x=Cluster, y=Depth)) + 
  geom_path() + 
  theme_bw() + 
  theme(legend.position = "none") +
  scale_y_reverse() + 
  facet_wrap(~ `Well Name`, nrow=1) +
  labs(title = "", x = "Cluster", y="")
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-13-1.png" width="672" />

We could do some more optimizing the number of clusters. I'll save that for a later update.  Thanks for reading!


