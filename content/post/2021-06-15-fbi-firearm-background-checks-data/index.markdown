---
title: Time Series Analysis of FBI Firearm Background Checks
author: Andy Wohlgenant
date: '2021-06-15'
slug: fbi-firearm-background-checks-data
categories: []
tags: []
subtitle: ''
summary: 'How do background checks for rifles and other so-called *long guns* change over time?'
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

## Why should we care?

The topic of guns in the United States can be a little tricky.  Pretty much everyone has an opinion about them, and arguments can erupt over them in the politest of company.  Wouldn't it be nice to just look at some cold, hard data and see what they say?  That's what I'll try to do here, and the data I found to look at come from the good old *Federal Bureau of Investigation* (FBI), who handles the background checks initiated by the sellers of guns.

I am going to steer clear of handguns in this case and concentrate on what the FBI calls *"long guns"*. Here's how the FBI describes this type of firearm:

> “*a weapon designed or redesigned, made or remade, and intended to be fired from the shoulder, and designed or redesigned and made or remade to use the energy of the explosive in (a) a fixed metallic cartridge to fire a single projectile through a rifled bore for each single pull of the trigger; or (b) a fixed shotgun shell to fire through a smooth bore either a number of ball shot or a single projectile for each single pull of the trigger.*”

That definition sounds like it is describing the kinds of guns that a lot of Americans use for hunting: rifles and shotguns. Do the sales of these kinds of guns vary over time?  And, if so, is there any kind of cyclicity or seasonality to the variation?  If we look past the seasonality (if it exists), is there a larger trend we can see?

These are the kinds of questions I want to look into here.  So let's start with the data.

## FBI Background Check Data

The data I will be working with are originally from the FBI's *National Instant Criminal Background Check System* (NICS).  You can read more about it on the FBI [website](https://www.fbi.gov/services/cjis/nics), but essentially these are just compilations by month and year for each state of the number of background checks for a variety of different categories.  Most of the categories are different types of firearms or different types of transactions. As I mentioned earlier, I will focus on the *long gun* category, but there are many other variables that could be investigated in these data.

The actual data I work with in this post are *not* directly from the FBI.  That is because the FBI releases these data periodically in PDF format, which is not easy to work with.  Luckily, the fine folks at [**BuzzFeed**](https://www.buzzfeed.com/) have done the hard work of parsing the PDF data into an easily-loaded CSV file.  The data can be downloaded from the BuzzFeedNews Github located [here](https://github.com/BuzzFeedNews/nics-firearm-background-checks).  There you can also read more about how they go about parsing the data.


```r
library(tidyverse)
library(anytime)
library(zoo)  # for the rollmean funtion
```



```r
nics <- read_csv("fbi_data.csv")
head(nics)
```

```
## # A tibble: 6 x 27
##   month   state      permit permit_recheck handgun long_gun other multiple admin
##   <chr>   <chr>       <dbl>          <dbl>   <dbl>    <dbl> <dbl>    <dbl> <dbl>
## 1 2021-05 Alabama     28248            317   21664    12423  1334      865     0
## 2 2021-05 Alaska        307              7    3368     2701   323      208     0
## 3 2021-05 Arizona     21767            695   20984     9259  1676     1010     0
## 4 2021-05 Arkansas     7697           1171    8501     5072   422      340     3
## 5 2021-05 California  20742          11514   40160    25824  5576        0     0
## 6 2021-05 Colorado    11105              3   21819    12848  1987     1980     0
## # ... with 18 more variables: prepawn_handgun <dbl>, prepawn_long_gun <dbl>,
## #   prepawn_other <dbl>, redemption_handgun <dbl>, redemption_long_gun <dbl>,
## #   redemption_other <dbl>, returned_handgun <dbl>, returned_long_gun <dbl>,
## #   returned_other <dbl>, rentals_handgun <dbl>, rentals_long_gun <dbl>,
## #   private_sale_handgun <dbl>, private_sale_long_gun <dbl>,
## #   private_sale_other <dbl>, return_to_seller_handgun <dbl>,
## #   return_to_seller_long_gun <dbl>, return_to_seller_other <dbl>, totals <dbl>
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

So these data run from November 1998 to May 2021, a nice chunk of time. Since I'm focussing on background checks for *long guns* and how they vary over time, I will group the data by the `date` variable and summarize the total number of background checks for long guns.


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

There are a couple of interesting things I can see right off the bat.  First, there definitely is some pretty obvious cyclicity to these data, and appears to be annual.  There's a really small value for the very first month, which might be spurious.  There's also a big spike around 2012 that looks interesting. And there seems to be an underlying trend we might be able to tease out.

Let's look at the first five actual values of the summarized data.


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

That looks better.  Let's see if we can see the underlying trend by calculating a 12-month moving average using the `rollmean` function from the `zoo` package and adding it to the plot.


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

Now we can see things a little more clearly.  The overall trend was pretty flat until around 2008, when there was a little upward bump in background checks.  Then there was a big surge around 2012.  The trend settled back down from 2016-2019 before another big surge starting around the beginning of 2020.

These bumps and surges seem to coincide with U.S. presidential election years.  The first bump coincides with Barack Obama's election to his first term, and the second surge coincides with his re-election in 2012.  Then there was a gradual drop during the Trump presidency, followed by another big surge in 2020, when Biden was elected.  So the surges seem to coincide with election of presidents from the Democratic party.

Now let's look at departures from the rolling average.  This should show us that seasonal cyclicity very clearly, as well as any extreme variations smoothed out by the moving average.


```r
nics_byDate <- nics_byDate %>%
  mutate(Departures = Total - long_gun_ma)

ggplot(nics_byDate, aes(x=date, y=Departures)) +
  geom_line() +
  geom_hline(yintercept = 0, linetype = "dotted") +
  theme_classic()
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-8-1.png" width="672" />

Now we can really see that spike around the end of 2012. Let's see which month had the highest departure from the 12-month rolling average.


```r
nics_byDate %>% filter(!is.na(Departures)) %>%
  filter(Departures == max(Departures))
```

```
## # A tibble: 1 x 4
##   date         Total long_gun_ma Departures
##   <date>       <dbl>       <dbl>      <dbl>
## 1 2012-12-01 1224465     640559.    583906.
```

Looks like it was December, 2012, just after Barack Obama won re-election to his second term.  There was a huge uptick in people buying guns that month, maybe because some people were afraid they wouldn't be able to buy them in the future.  I'll mark that month on the plot with a vertical red line.


```r
ggplot(nics_byDate, aes(x=date, y=Departures)) +
  geom_line() +
  #geom_hline(yintercept = 0, linetype = "dotted") +
  geom_vline(xintercept = as.numeric(as.Date("2012-12-01")), 
             color = "red", lwd=1.25) +
  theme_classic() +
  labs(title="Departures from 12-month rolling average of long gun checks",
       x = "Date", y = "Difference from rolling average",
       caption = "Long gun sales surged in December, 2012 after Obama won re-election")
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-10-1.png" width="672" />

## Decomposition of time-series data

A lot of the work I just did can be handled with more formal *time series analysis* statistical methods.  These methods can be easily accessed in R using functions from the `stats` package that comes with R out-of-the-box.

In time series analysis, we typically want to decompose a time series into three components: *trend*, *seasonality*, and a residual process that is **stationary**.  We would then treat the trend and seasonality components as deterministic, and the residual component can be thought of as an error term that is random.

Let's use the `ts` and `stl` functions from `stats` to decompose the time series into *seasonal*, *trend*, and *remainder* components using *loess* or *locally weighted smoothing*.  Then we can plot the three components on separate plot facets.


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

# Factor the component column to the order I want
df$component <- factor(df$component, levels=c("trend", "seasonal", "remainder"))

# Plot the decomposed parts
ggplot(df, aes(x=date, y=value)) + 
  geom_line() + 
  theme_minimal() + 
  labs(title="Decomposition of Total Long Gun Background Checks",
       x = "Date", y="") +
  facet_wrap(~component, nrow=3, ncol=1)
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-11-1.png" width="672" />

Now we can see quite clearly the three components to the data.  There is an overall trend in the top plot, a repeating annual cyclicity component in the middle plot, and the remaining "noise" component in the bottom plot.

There's a lot more we could do with this data set.  We could see if the trends differ by state, for instance.  But I will save that for a later post.  Thanks for reading!  If you have any comments, feel free to email me at akwohlg@gmail.com.

