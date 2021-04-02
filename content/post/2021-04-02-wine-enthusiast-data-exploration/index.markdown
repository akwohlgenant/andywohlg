---
title: Wine Enthusiast Data Exploration
author: Andy Wohlgenant
date: '2021-04-02'
slug: wine-enthusiast-data-exploration
categories: []
tags: []
subtitle: ''
summary: ''
authors: []
lastmod: '2021-04-02T14:21:51-07:00'
featured: no
image:
  caption: 'Image credit: [**Unsplash**](https://unsplash.com/photos/AZW8IbilGLg)'
  focal_point: ''
  preview_only: no
projects: []
---

### Introduction

I like wine.  Lots of people like wine.  I tend to drink the same wines over and over again, from an embarrassingly small list of varieties and regions.  But there are hundreds of varieties of wine produced in dozens of regions in countries ranging from Armenia to Uruguay, with a dizzying array of naming conventions that can be downright confusing.  And I, like many people, am interested in learning more about wines, maybe in the hope of finding a cheap, underrated bargain at my local liquor store.  Luckily, there are myriad magazines and websites devoted entirely to the discussion, reviewing, and rating of wines.  One of the most popular of such magazines and websites is _Wine Enthusiast_.

![my-first-image](serge-esteve-AZW8IbilGLg-unsplash.jpg)

_Wine Enthusiast_ was founded in 1988 by Adam and Sybil Strum, and now has a circulation of more than 250,000.  One of the most popular features of the Wine Enthusiast magazine and website (www.winemag.com) are the wine reviews, which contain a wealth of information about each wine, both useful and not so useful.  Information found in a given review includes: the country and region of origin, the name of the vintner or winery, the variety (e.g. Pinot Noir), the price (arguably the most important feature), the name of the reviewer, and even the reviewer's _Twitter_ handle.  In addition to descriptive information, each review contains a rating or "points" awarded to the wine based on a variety of categories.  The ratings are purported to be on a 100-point scale, although in reality it is just a 20-point scale from 80 to 100 points.

Wine Enthusiast's 100-point wine-scoring scale:

- Classic: 98–100
- Superb: 94–97
- Excellent: 90–93
- Very Good: 87–89
- Good: 83–86
- Acceptable: 80–82

(https://www.wine-searcher.com/critics-17-wine+enthusiast)

Another feature of each review is a text description or "tasting notes"; these are often full of flowery adjectives and borderline ridiculous imagery to intended to evoke the actual experience of tasting a wine, with comments about the wine's appearance, aroma, flavor, etc.  Here is an example description, chosen at random from the dataset:

_"Fragrant and fresh, this Grillo opens with alluring scents of acacia flower, beeswax and white stone fruit. The succulent palate offers creamy white peach, juicy nectarine, almond and mineral framed in tangy acidity. A note of chopped herb closes the lingering finish."_

How many people know what acacia flowers and beeswax smell like?

### Questions of Interest:

The aim of this analysis was to investigate a dataset consisting of nearly 130,000 wine reviews that were originally scraped (not by me) from the _Wine Enthusiast_ website in 2017 and posted on _Kaggle_ (www.kaggle.com).  A number of questions are investigated, including:

  * Which countries have the most reviewed wines?
  
  * Which varieties are the most reviewed?
  
  * What does the price distribution look like?  Is it a normal distribution, or is it more heavily skewed toward lower priced wines?
  
  * What about the distribution of wine ratings ("points")?
  
  * Is there any relationship between price of wine and its corresponding rating in points?  Are higher priced wines rated higher than cheap wines?
  
  * What are the most commonly used words in the wine descriptions?  How can those be visualized?

These questions will be investigated by using a number of different statistical and graphical techniques using R.

### Data

As mentioned previously, the dataset for this research consists of a comma separated file downloaded from _Kaggle_, and comprises nearly 130,000 wine reviews that were scraped from the _Wine Enthusiast_ website (www.winemag.com) in 2017.  The data contain 14 variables:
 
  * Unique identifier (numerical, column name "X")
  * Country of origin
  * Province (if country is US, this is the state of origin)
  * Region (there are two region variables, *region_1* and *region_2*)
  * Designation (this appears to be some kind of secondary descriptor)
  * Title (generally the name of producer, variety and vintage)
  * Description (text description or "tasting notes" for each wine)
  * Variety (e.g. Pinot Noir or Chardonnay)
  * Winery
  * Price (US Dollars)
  * Points (80-100 points)
  * Taster name
  * Taster Twitter handle
 
Here is a link to download the data from _Kaggle_: 

https://www.kaggle.com/zynicide/wine-reviews

### Methods

The dataset is investigated using a number of different descriptive, statistical, and graphical methods available in R and it's associated packages, including _dplyr_, _ggplot2_, _tm_, and others.  Methods used include:

  * Grouping by country and variety and summarizing counts.
  * Plotting histograms of wine price and points.
  * Binning wines into price range bins (e.g. $10-$20, $20-$30, etc.) and visualizing price distribution
  * Visualizing wine ratings and price bins as a way of addressing whether there is a relationship between price and rating points.
  * Using word frequency counting and word cloud to visualize the frequency of words used in wine descriptions.
  
### Analysis

First, the necessary packages are included for the analysis.


```r
library(dplyr) # for a variety of analysis functions
library(ggplot2) # for plotting
```


```r
# the packages below are included to aid in construction of the word cloud, among other functions
library(tm) 
library(SnowballC)
library(wordcloud)
library(RColorBrewer)
library(tidytext)
library(stringr)
library(tidyverse)
library(tmap)
library(plotly)
```



```r
#install.packages("qdap")
#install.packages("qdap", INSTALL_opts = "--no-multiarch")
library(qdap)
```


Next, the dataset file is loaded into R and the dimensions inspected:


```r
wine <- read.csv("winemag_data_130k_v2.csv", encoding="UTF-8" )
dim(wine)
```

```
## [1] 129971     14
```

The dataset contains 129,971 observations, and 14 variables.  Next the variable names are listed:


```r
colnames(wine)
```

```
##  [1] "X"                     "country"               "description"          
##  [4] "designation"           "points"                "price"                
##  [7] "province"              "region_1"              "region_2"             
## [10] "taster_name"           "taster_twitter_handle" "title"                
## [13] "variety"               "winery"
```

Most of the variable names are pretty self-explanatory. Let's see what a "description" looks like for a randomly chosen wine in the dataset:


```r
wine$description[sample(1:129971, 1)]
```

```
## [1] "Winemaker: Etienne le Riche. This is a beast of a wine right now, begging for time in the cellar to further mature. Currently, assertive oaky tones of sweet smoke, cigar box and char are front and center on the bouquet, with supporting notes of cassis, blackcurrant leaf, brambly berry and plum skin to show the fruity core beneath. The palate is dark and brooding, with concentrated dark-fruit flavors and firmly structured tannins that beg for time to harmonize and resolve. Drink 2020–2025."
```

This description is full of interesting adjectives and imagery meant to elicit in the reader a sense of the appearance, aroma, and flavor of the wine.  Later, the words used in the description variable will be more thoroughly investigated and visualized.

Now the countries of origin will be investigated.  What are the unique countries in this dataset?


```r
countries <- unique(wine$country)
length(countries)
```

```
## [1] 44
```
So 44 unique countries are represented in the dataset.  Let's get a listing of these countries, and rank them by number of reviews and relative proportion of the dataset.  I will also calculate a running total of the proportions.


```r
wine_country <- wine %>%
  filter(!is.na(country)) %>%
  group_by(country) %>%
  summarise(Count=n(), Prop=round(n()/(length(wine$country)), 4)) %>%
  arrange(-Count) %>%
  mutate(Cum_Prop = cumsum(Prop))
wine_country
```

```
## # A tibble: 44 x 4
##    country   Count   Prop Cum_Prop
##    <chr>     <int>  <dbl>    <dbl>
##  1 US        54504 0.419     0.419
##  2 France    22093 0.17      0.589
##  3 Italy     19540 0.150     0.740
##  4 Spain      6645 0.0511    0.791
##  5 Portugal   5691 0.0438    0.835
##  6 Chile      4472 0.0344    0.869
##  7 Argentina  3800 0.0292    0.898
##  8 Austria    3345 0.0257    0.924
##  9 Australia  2329 0.0179    0.942
## 10 Germany    2165 0.0167    0.958
## # ... with 34 more rows
```


As we can see above in the Cum_Prop (cumulative proportion or running total of the proportions) column, the top 10 countries in terms of number of reviews account for 96% of the observations in the dataset, so I am going to filter out the countries with very few reviews before any plotting.


```r
wine_country <- wine_country %>%
  filter(Prop>0.0005)
```

Below is a bar plot of review count by country after filtering.


```r
ggplot(data=wine_country, aes(x=reorder(country, -Count), y=Count)) + geom_bar(stat='identity', fill='dodgerblue') + labs(title="Wine Review Counts by Country",x="Country", y = "Count") + theme(axis.text.x=element_text(angle=90, hjust=1), plot.title = element_text(hjust = 0.5))
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-10-1.png" width="672" />

And here's a line plot of the cumulative sum of country proportions:


```r
ggplot(data=wine_country, aes(x=Cum_Prop, y=(reorder(country, -Cum_Prop)), group=1)) + geom_line(linetype='dashed') + geom_point() + labs(title='Cumulative Sum of Wine Review Proportion by Country', x="Cumulative Proportion", y="Country")
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-11-1.png" width="672" />


As we can see, the top few countries make up the bulk of the wine reviews in the dataset, and the counts tail off rapidly in the rest of the countries represented..

Next, a similar analysis is performed with wine varieties:


```r
varieties <- unique(wine$variety)
length(varieties)
```

```
## [1] 708
```

The dataset contains 708 unique varieties of wine.  In order to visualize the varieties that make up the dataset, it is easiest to filter out wine varieties with the fewest reviews.  In this case, wines with fewer than 500 reviews will be removed before plotting the visualization.


```r
wine_varieties <- wine %>%
  group_by(variety) %>%
  summarise (count=n()) %>%
  filter(count>500) %>%
  arrange(-count)
```



```r
ggplot(data=wine_varieties, aes(x=reorder(variety, -count), y=count)) + geom_bar(stat='identity', fill='dodgerblue') + labs(title="Wine Review Counts by Variety",x="Variety", y = "Count") + theme(axis.text.x=element_text(angle=90, hjust=1), plot.title = element_text(hjust = 0.5))
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-14-1.png" width="672" />

As we can see above, the four most reviewed wines are Pinot Noir, Chardonnay, Cabernet Sauvignon, and Red Blend. This distribution is not nearly as concentrated in a few varieties as was the case with the countries of origin.  

Now let's look into the prices of wines, and the ratings or "points" awarded them.  First, let's filter out the very high priced wines that are out of virtually everyone's price range (over $500 a bottle).


```r
mod_price <- wine %>%
  filter(price < 500)
```



```r
ggplot(data=mod_price, aes(mod_price$price)) + geom_histogram(binwidth = 10, color='black', fill='dodgerblue') + labs(title="Histogram of Wine Prices",x="Price, USD", y = "Count") + theme(plot.title = element_text(hjust = 0.5))
```

```
## Warning: Use of `mod_price$price` is discouraged. Use `price` instead.
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-16-1.png" width="672" />


Here it apparent that the prices are highly concentrated in the low range (less than $50), with a long tail extending out into the higher prices.  We will do more thorough investigation of this observation.  Let's take a look at summary statistics for the wine prices.


```r
summary(wine$price)
```

```
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
##    4.00   17.00   25.00   35.36   42.00 3300.00    8996
```

As we can see from the summary statistics, the median wine price is only 25 US dollars, while the average is more than 35 bucks, dragged higher by those few, very expensive wines, up to a maximum of 3,300 dollars.  There are also almost 9,000 null values in the price variable.

Let's get rid of those observations with null in the price variable before continuing our analysis.


```r
wine <- wine %>%
  filter(!is.na(price))
dim(wine)
```

```
## [1] 120975     14
```

That drops our dataset down to just under 121,000 observations; still plenty of data to work with.

Now lets look at the distribution of the wine ratings represented by the variable _points_:


```r
ggplot(wine, aes(wine$points)) + geom_histogram(binwidth = 1, color='black', fill='dodgerblue') + labs(title="Histogram of Wine Points",x="Points", y = "Count") + theme(plot.title = element_text(hjust = 0.5))
```

```
## Warning: Use of `wine$points` is discouraged. Use `points` instead.
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-19-1.png" width="672" />

```r
# the 'hjust' is added to center the title, otherwise left-justified
```


This distribution of points appears to be very nearly normal, with a peak around 87-88 points.  Below is a statistical summary:


```r
summary(wine$points)
```

```
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##   80.00   86.00   88.00   88.42   91.00  100.00
```

In this near-normal distribution, the mean and median are very close to one another at 88 and 88.42 points, respectively.

As an aside, let's see cheap Australian Rieslings with ratings of at least 90 points:


```r
wine %>%
  filter(country=="Australia", variety=="Riesling", points>90, price<20) %>%
  select(title, price, points) %>%
  arrange(price)
```

```
##                                                                      title
## 1  Thorn Clarke 2012 Mount Crawford Single Vineyard Riesling (Eden Valley)
## 2                                   St Hallett 2014 Riesling (Eden Valley)
## 3       D'Arenberg 2015 The Dry Dam Riesling (McLaren Vale-Adelaide Hills)
## 4                             Robert Oatley 2012 Riesling (Great Southern)
## 5                                   Wakefield 2011 Riesling (Clare Valley)
## 6                      Thorn Clarke 2014 Eden Trail Riesling (Eden Valley)
## 7                Jim Barry 2014 The Lodge Hill Dry Riesling (Clare Valley)
## 8                           Hewitson 2015 Gun Metal Riesling (Eden Valley)
## 9                Jim Barry 2016 The Lodge Hill Dry Riesling (Clare Valley)
## 10              D'Arenberg 2008 The Noble Wrinkled Riesling (McLaren Vale)
##    price points
## 1     15     91
## 2     16     91
## 3     17     92
## 4     17     91
## 5     17     91
## 6     18     92
## 7     18     91
## 8     19     91
## 9     19     91
## 10    19     94
```

And let's see Sauvignon Blancs with points greater than, say, 92 and price less than 30 dollars:


```r
wine %>%
  filter(variety=='Sauvignon Blanc', points>92, price<30) %>%
  select (title, price, points) %>%
  arrange(price)
```

```
##                                                                                   title
## 1  Dutton Estate 2006 Dutton Ranch Kylie's Cuvée Sauvignon Blanc (Russian River Valley)
## 2                    Patianna 2009 Made With Organic Grapes Sauvignon Blanc (Mendocino)
## 3                  Easton 2009 Monarch Mine Vineyard Sauvignon Blanc (Sierra Foothills)
## 4                                    Navarro 2009 Cuvee 128 Sauvignon Blanc (Mendocino)
## 5                     Gainey 2014 Limited Selection Sauvignon Blanc (Santa Ynez Valley)
## 6                                    Guardian 2012 Angel Sauvignon Blanc (Red Mountain)
## 7                                Kenwood 2015 Six Ridges Sauvignon Blanc (Sonoma Coast)
## 8                  Efeste 2012 Boushey Vineyard Sauvage Sauvignon Blanc (Yakima Valley)
## 9                                      Domaine Fouassier 2008 Les Chailloux  (Sancerre)
## 10                                        Michel Vattan 2015 Cuvée Calcaire  (Sancerre)
## 11                  Joseph Jewell 2008 Redwood Ranch Sauvignon Blanc (Alexander Valley)
## 12                         Kenefick Ranch 2015 Estate Grown Sauvignon Blanc (Calistoga)
## 13                                        Clos Henri 2015 Sauvignon Blanc (Marlborough)
## 14                                                  Pierre Morin 2014 Ovide  (Sancerre)
## 15                Woodward Canyon 2009 Estate Sauvignon Blanc (Walla Walla Valley (WA))
## 16                                          Duckhorn 2010 Sauvignon Blanc (Napa Valley)
## 17                                           Conspire 2009 Sauvignon Blanc (Rutherford)
## 18                               MacLaren 2014 Lee's Sauvignon Blanc (Dry Creek Valley)
## 19                                           Conspire 2009 Sauvignon Blanc (Rutherford)
## 20                                           Jean-Max Roger 2014 Cuvée C.M.  (Sancerre)
##    price points
## 1     17     93
## 2     17     93
## 3     18     93
## 4     18     93
## 5     19     93
## 6     20     94
## 7     22     93
## 8     23     93
## 9     23     93
## 10    24     93
## 11    24     93
## 12    24     93
## 13    25     93
## 14    25     93
## 15    26     93
## 16    27     93
## 17    28     93
## 18    28     93
## 19    28     93
## 20    28     93
```

I'll be checking my local wine store for some of these bargains!

We will now do a more thorough investigation of wine prices and compare them with the wine ratings to see if a relationship exists between the two variables.  To simplify this analysis, we will divide the wines into buckets or bins based on their price in $10 increments ($0-$10, $10-$20, etc.)  Then we can see the distribution of wine counts by bin.

First we create a vector of the values to be the "breaks" between bins, and labels or "tags" to attach to each of these breaks.


```r
breaks1 <- c(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 100000)
tags <- c('$0-10', '$10-20', '$20-30', '$30-40', '$40-50', '$50-60', '$60-70', '$70-80', '$80-90', '$90-100', '>$100')
```

Then we can use the _cut_ function to divide the price variable into bins separated by the breaks we just created, and summarize the counts for each bin:


```r
price_bins <- cut(wine$price, breaks=breaks1, include.lowest=TRUE, right=FALSE, labels=tags)
summary(price_bins)
```

```
##   $0-10  $10-20  $20-30  $30-40  $40-50  $50-60  $60-70  $70-80  $80-90 $90-100 
##    2841   36560   29103   17299   12064    7593    5111    3163    1898    1392 
##   >$100 
##    3951
```

Above it is apparent that the largest bin of prices is the $10-$20 bin, which contains 36,560 observations.  We can visualize this distribution better with a bar graph:


```r
ggplot(data=as_tibble(price_bins), mapping=aes(x=value)) + geom_bar(fill="dodgerblue") + labs(title="Binned Wine Prices",x="Wine Price Bins, USD", y = "Count") + theme(plot.title = element_text(hjust = 0.5))
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-25-1.png" width="672" />

Next, we can use these price bins as a way to compare with wine points and investigate the question of whether a relationship exists between price and points. To do this we will first select just those columns, and then create a new column called 'price_bin' that will be assigned one of our previously defined bin tags depending on the price of the wine:


```r
wine_pp <- wine %>% select(price, points)
wb_group <- as_tibble(wine_pp) %>%
  mutate(price_bin = case_when(
    price < 10 ~ tags[1],
    price >= 10 & price < 20 ~ tags[2],
    price >= 20 & price < 30 ~ tags[3],
    price >= 30 & price < 40 ~ tags[4],
    price >= 40 & price < 50 ~ tags[5],
    price >= 50 & price < 60 ~ tags[6],
    price >= 60 & price < 70 ~ tags[7],
    price >= 70 & price < 80 ~ tags[8],
    price >= 80 & price < 90 ~ tags[9],
    price >= 90 & price < 100 ~ tags[10],
    price >= 100 ~ tags[11]
  ))
```

Let's see what that new column looks like:


```r
head(wb_group)
```

```
## # A tibble: 6 x 3
##   price points price_bin
##   <dbl>  <int> <chr>    
## 1    15     87 $10-20   
## 2    14     87 $10-20   
## 3    13     87 $10-20   
## 4    65     87 $60-70   
## 5    15     87 $10-20   
## 6    16     87 $10-20
```

Now let's look at the distribution of wine ratings for each price bin with box plots (often called 'box-and-whisker' plots) for each bin.  I will also post the individual data points and 'jitter' them (shift them a small, random distance laterally) so that we can get a visual feel for the quantity and distribution of price points for each bin.


```r
ggplot(data=wb_group, mapping = aes(x=price_bin, y=points)) + geom_jitter(color='dodgerblue', alpha=0.1) + 
  geom_boxplot(fill='bisque', color='black', alpha=0.3) + theme_minimal()+ labs(title="Boxplots of Wine Price Bins",x="Wine Price Bins, USD", y = "Points") + theme(plot.title = element_text(hjust = 0.5))
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-28-1.png" width="672" />

We can see from this plot not only the spread in the underlying ratings data for each price bin, but also the price-progression of the median ratings for all bins.  It is apparent that overall there is an increase in rating points associated with an increase in price.  This effect is more dramatic in the lower price bins from zero to 50 dollars, and the effect is flattened with higher prices, such that the median rating for a 90-100 dollar wine is not significantly different than that for a 70-80 dollar wine.

This raises an interesting question: are the more expensive wines rated higher because they are better?  Or are the tasters biased by their own knowledge of the wine's price, and therefore more likely to rate the wine higher if it is more expensive?  This question can not be answered from these data, but the relationship could be further investigated using a dataset of blind tastings where the tasters are ignorant of the price when assigning the ratings.

### Word Frequencies in Tasting Notes & Word Cloud

The final question will now be investigated: what words are most commonly used to describe the wines in the description variable?  To do this, word frequencies will be counted, and a word cloud will be generated to visualize the results.

First a _corpus_ is created, which is a collection of natural language text.  In this case, it is created from the _description_ column of the dataset.


```r
wine_corpus = Corpus(VectorSource(wine$description))
```

Next the corpus is edited to remove upper case letter, numbers, punctuation, and common words that don't yield any insights (e.g. "the").


```r
nowords <- c("the", "and", "wine", "with", "this", "flavors", "its", "that", "has", "but", "are", "offers", "more", "from")
```



```r
wine_corpus = tm_map(wine_corpus, content_transformer(tolower))
wine_corpus = tm_map(wine_corpus, removeNumbers)
wine_corpus = tm_map(wine_corpus, removePunctuation)
wine_corpus = tm_map(wine_corpus, removeWords, nowords)
wine_corpus =  tm_map(wine_corpus, stripWhitespace)
```

```r
inspect(wine_corpus[846])
```

```
## <<SimpleCorpus>>
## Metadata:  corpus specific: 1, document level (indexed): 0
## Content:  documents: 1
## 
## [1] dusty tight a whiff of coffee cocoa introducing an elegant interesting streaks of metal graphite young fruit fine tannins in perfect balance a lovely delicacy
```

An inspection above the description for element number 846 in the corpus shows that the unnecessary features were removed.

Next, I create a document-term matrix for the corpus:


```r
wine_dtm <- DocumentTermMatrix(wine_corpus)
wine_dtm = removeSparseTerms(wine_dtm, 0.99)
```

Let's check out the dimensions of the document-term matrix:


```r
dim(wine_dtm)
```

```
## [1] 120975    461
```

Then create and sort a dataframe of the sums for each column of the document-term matrix.


```r
freq = data.frame(freqs=sort(colSums(as.matrix(wine_dtm)), decreasing=TRUE))
head(freq, 10)
```

```
##         freqs
## fruit   41565
## aromas  37479
## palate  36378
## finish  33669
## acidity 31501
## tannins 28104
## drink   27621
## cherry  25974
## ripe    24233
## black   23778
```

And finally, generate the word cloud from the frequency dataframe.


```r
wordcloud(rownames(freq), freq[,1], max.words=50, colors=brewer.pal(1, "Dark2"))
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-36-1.png" width="672" />


The word cloud above is limited to 50 words, but gives a good idea of the kinds of words that are frequently used in the wine descriptions.  While additional fine-tuning might yield more insights (for instance, I could get rid of words like "some" that aren't telling us much), it might be more interesting to divide the wines up by variety and see how the word frequencies differ.  It may even be possible to use the words from the descriptions to predict the variety of wine.  I will save that question for the next project.

### Conclusions

The results of this analysis effectively answered the research questions posed:

  * The countries of origin were highly concentrated in a few countries, with about 74% of the reviews coming from the United States, France and Italy.
  
  * The wine varieties were somewhat more evenly distributed than the countries, with the most common wines being Pinot Noir, Chardonnay, and Cabernet Sauvignon.
  
  * The wine prices were highly concentrated in the $10 to $50 dollar range, with a long tail exending up in price to a maximum price of $3,300.
  
  * The wine ratings or "points" were near normally distributed, with mean and median values of approximately 88 points.
  
  * There was a relationship noted between price and points, such that an increase in price is associated with an increase in awarded points.  This effect was more dramatic in the lower price bins (0-$50) than in the higher price range, where increase in price more than about 70 dollars did not appear to be associated with significantly higher ratings.
  
  * And finally, the most frequently used words in the wine desription column were calculated and visualized using a word cloud.  This provided an interesting way of visualizing the most common words in the descriptions, but to really leverage the description text, we might want to take things a bit further and try to use the language to predict something like the wine variety.
