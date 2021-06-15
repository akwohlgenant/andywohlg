---
title: FBI Firearm Background Checks
author: Andy Wohlgenant
date: '2021-06-15'
slug: fbi-firearm-background-checks-data
categories: []
tags: []
subtitle: ''
summary: ''
authors: []
lastmod: '2021-06-15T10:01:23-07:00'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---
![my-first-image](seth-schulte-unsplash.jpg)

*Photo by Seth Schulte on Unsplash*

## FBI Background Check Data

These data are from the FBI National Instant Criminal Background Check System (NICS).  You can read more about it on the FBI [website](https://www.fbi.gov/services/cjis/nics).  The data I work with in this post are actually from the *BuzzFeedNews* Github located [here](https://github.com/BuzzFeedNews/nics-firearm-background-checks).  There you can also read about how the original data provided by the FBI in PDF format are parsed into comma separated (CSV) format.


```r
library(tidyverse)
library(anytime)
library(zoo)  # for rollmean funtion
```



```r
nics <- read_csv("fbi_data.csv")
#dim(nics)
#colnames(nics)
#head(nics)
```

## Exploration

Let's see what date range these data cover. It looks like the `month` variable needs to be converted to a proper date format first.


```r
nics$date <- anydate(nics$month)
min(nics$date)
```

```
## [1] "1998-11-01"
```

```r
max(nics$date)
```

```
## [1] "2021-05-01"
```

So these data run from November 1998 to May 2021. I'm interested in looking at background checks on what the FBI calls *long guns*.  Here's how the FBI describes this category of firearms:

> “*a weapon designed or redesigned, made or remade, and intended to be fired from the shoulder, and designed or redesigned and made or remade to use the energy of the explosive in (a) a fixed metallic cartridge to fire a single projectile through a rifled bore for each single pull of the trigger; or (b) a fixed shotgun shell to fire through a smooth bore either a number of ball shot or a single projectile for each single pull of the trigger.*”


```r
nics %>% 
  filter(!is.na(long_gun)) %>%
  group_by(date) %>% 
  summarize(Total = sum(long_gun)) %>%
  ggplot(aes(x=date, y=Total)) + geom_line() +
  theme_minimal() +
  labs(title="Total Long Gun Background Checks by Month",
       x="Date", y="Total Background Checks")
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-4-1.png" width="672" />

Looks like there is a very small number for the first month in the data.  Let's take a look at the actual numbers.


```r
nics %>% 
  filter(!is.na(long_gun)) %>%
  group_by(date) %>% 
  summarize(Total = sum(long_gun)) %>%
  head(n=5)
```

```
## # A tibble: 5 x 2
##   date        Total
##   <date>      <dbl>
## 1 1998-11-01  11909
## 2 1998-12-01 570882
## 3 1999-01-01 309915
## 4 1999-02-01 352411
## 5 1999-03-01 376775
```

The very first month looks to be a bad value since it's more than an order of magnitude smaller than every other month total in the data.  Let's take it out and replot.


```r
nics %>% 
  filter(!is.na(long_gun), date>"1998-11-01") %>%
  group_by(date) %>% 
  summarize(Total = sum(long_gun)) %>%
  ggplot(aes(x=date, y=Total)) + geom_line() +
  theme_minimal() +
  labs(title="Total Long Gun Background Checks by Month",
       x="Date", y="Total Background Checks")
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-6-1.png" width="672" />

Calculate the moving average and plot.


```r
nics_byDate <- nics %>% 
  filter(!is.na(long_gun), date>"1998-12-01") %>%
  group_by(date) %>% 
  summarize(Total = sum(long_gun))

nics_byDate <- nics_byDate %>% mutate(long_gun_ma = rollmean(Total,12, fill=NA))

ggplot(nics_byDate, aes(x=date, y=Total)) + 
  geom_line() +
  geom_line(aes(x=date, y=long_gun_ma, color="12 mo. moving avg"), size=0.75) +
  theme_classic() +
  theme(legend.title = element_blank(),
        legend.position = "bottom") +
  labs(title="Total Long Gun Background Checks by Month",
       x="Date", y="Total Background Checks")
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-7-1.png" width="672" />

Calculate departures from moving average.


```r
nics_byDate <- nics_byDate %>%
  mutate(long_gun_diff = Total - long_gun_ma)

ggplot(nics_byDate, aes(x=date, y=long_gun_diff)) +
  geom_line() +
  geom_hline(yintercept = 0, linetype = "dotted") +
  theme_classic()
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-8-1.png" width="672" />

Let's see which month had the highest departure from the 12-month rolling average.


```r
nics_byDate %>% filter(!is.na(long_gun_diff)) %>%
  filter(long_gun_diff == max(long_gun_diff))
```

```
## # A tibble: 1 x 4
##   date         Total long_gun_ma long_gun_diff
##   <date>       <dbl>       <dbl>         <dbl>
## 1 2012-12-01 1224465     640559.       583906.
```

Looks like it was December 2012, just after Barack Obama won re-election, and there was a huge uptick in people buying guns, presumably because they were afraid they wouldn't be able to buy them in the future.


```r
ggplot(nics_byDate, aes(x=date, y=long_gun_diff)) +
  geom_line() +
  #geom_hline(yintercept = 0, linetype = "dotted") +
  geom_vline(xintercept = as.numeric(as.Date("2012-12-01")), 
             color = "red", linetype = "dashed") +
  theme_classic() +
  labs(title="Departures from 12-month rolling average of long gun checks",
       x = "Date", y = "Difference from rolling average",
       caption = "Surge in gun sales in December 2012 - Obama re-election")
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-10-1.png" width="672" />

## Decomposition of time-series data

Now let's use the `ts` and `stl` functions from `stats` to decompose the time series into *seasonal*, *trend*, and *remainder* components using *loess*.


```r
# Make a time series object of the total long gun checks
nics_ts <- ts(nics_byDate$Total, frequency=12, start=c(1999, 1))

# Decompose using stl
decomposed <- stl(nics_ts, s.window='periodic')

# Make data frame to plot with ggplot
df <- as.data.frame(decomposed$time.series)

# Add date column
df$date <- nics_byDate$date

# Pivot longer for faceting by component
df <- df %>% pivot_longer(cols = 1:3, names_to = "component", values_to = "value")

# Plot the decomposed parts
ggplot(df, aes(x=date, y=value)) + 
  geom_line() + 
  theme_minimal() + 
  labs(title="Decomposition of Total Long Gun Background Checks",
       x = "Date", y="") +
  facet_wrap(~component, nrow=3, ncol=1)
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-11-1.png" width="672" />



