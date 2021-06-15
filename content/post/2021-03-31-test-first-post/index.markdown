---
title: Test First Post
author: ''
date: '2021-03-31'
slug: test-first-post2
categories: []
tags: []
subtitle: ''
summary: 'This is my first post using the blogdown platform, so I'm just making sure everything works...'
authors: []
lastmod: '2021-03-31T15:49:46-07:00'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---

## My First R Markdown Post

This is my first post, and I will just be plotting some random stuff to see if it works.  I will also be testing a function I copied from Julia Silge's *silgelib* package to make my plots look as nice as hers look in her post on Super Bowl Commercials and Bootstrap Confidence Intervals.





```r
summary(Orange)
```

```
##  Tree       age         circumference  
##  3:7   Min.   : 118.0   Min.   : 30.0  
##  1:7   1st Qu.: 484.0   1st Qu.: 65.5  
##  5:7   Median :1004.0   Median :115.0  
##  2:7   Mean   : 922.1   Mean   :115.9  
##  4:7   3rd Qu.:1372.0   3rd Qu.:161.5  
##        Max.   :1582.0   Max.   :214.0
```


```r
oplot <- ggplot(Orange, aes(x = age, 
                   y = circumference, 
                   colour = Tree)) +
  geom_point() +
  geom_line() +
  guides(colour = FALSE) +
  theme_bw()
oplot
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-2-1.png" width="2400" />

Next I will load the data used by Julia Silge in her post on March 4, 2021, **Bootstrap confidence intervals for #TidyTuesday Super Bowl commercials**.  First I will read in the dataset from the TidyTuesday github repository.


```r
youtube <- read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-03-02/youtube.csv')
```

I noticed that Julia's plots looked really nice - simple but clear and uncluttered.  I also noticed a call in her R setup to a theme called *theme_plex* that I didn't recognize.  I looked it up, and found it is actually from a package Julia built called *silgelib*.  Here's the function separated out from the package.


```r
# This is a plotting function from Julia Silge's package silgelib
theme_plex <- function(base_size = 11,
                       strip_text_size = 12,
                       strip_text_margin = 5,
                       subtitle_size = 13,
                       subtitle_margin = 10,
                       plot_title_size = 16,
                       plot_title_margin = 10,
                       ...) {
  ret <- ggplot2::theme_minimal(base_family = "IBMPlexSans",
                                base_size = base_size, ...)
  ret$strip.text <- ggplot2::element_text(
    hjust = 0, size = strip_text_size,
    margin = ggplot2::margin(b = strip_text_margin),
    family = "IBMPlexSans-Medium"
  )
  ret$plot.subtitle <- ggplot2::element_text(
    hjust = 0, size = subtitle_size,
    margin = ggplot2::margin(b = subtitle_margin),
    family = "IBMPlexSans"
  )
  ret$plot.title <- ggplot2::element_text(
    hjust = 0, size = plot_title_size,
    margin = ggplot2::margin(b = plot_title_margin),
    family = "IBMPlexSans-Bold"
  )
  ret
}
```

And here's an example plot of the diamonds data using the theme_plex() function defined above.


```r
ggplot(diamonds, aes(carat, price, color = clarity)) +
  geom_point(alpha = 0.7) +
  facet_wrap(~cut) +
  labs(title = "Diamonds Data",
       subtitle = "Made prettier with Julia Silge's theme_plex() function") + theme_plex()
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-5-1.png" width="2400" />

Finally, here's a duplicate of the plot in Julia's post.  Isn't that a nice-looking, clear plot?


```r
youtube %>%
  select(year, funny:use_sex) %>%
  pivot_longer(funny:use_sex) %>%
  group_by(year, name) %>%
  summarise(prop = mean(value)) %>%
  ungroup() %>%
  ggplot(aes(year, prop, color = name)) +
  geom_line(size = 1.2, show.legend = FALSE) +
  facet_wrap(vars(name)) +
  scale_y_continuous(labels = scales::percent) +
  labs(x = NULL, y = "% of commercials") + theme_plex()
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-6-1.png" width="2400" />

