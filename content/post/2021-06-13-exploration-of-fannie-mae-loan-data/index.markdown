---
title: Exploration of Fannie Mae Loan Data
author: Andy Wohlgenant
date: '2021-06-13'
slug: exploration-of-fannie-mae-loan-data
categories: []
tags: []
subtitle: ''
summary: 'Exploration of Fannie Mae loan data from 2019...'
authors: []
lastmod: '2021-06-13T15:25:41-07:00'
featured: no
image:
  caption: ''
  focal_point: ''
  preview_only: no
projects: []
---
![my-first-image](blake-wheeler-unsplash.jpg)

*Photo by Blake Wheeler on Unsplash*

## Introduction

I became interested in learning more about the mortgage market in the United States after reading the book *Homewreckers* by Aaron Glantz, which documents the monumental change in ownership of single-family homes that took place after the 2008 financial crisis, when many U.S. homes went into foreclosure and were subsequently purchased at a discount by Wall Street firms and wealthy investors. 

One of the major outcomes of the housing crisis was the bailout of *Fannie Mae* and *Freddie Mac*,
government sponsored entities that were originally conceived during the Great Depression to provide liquidity to the beleaguered U.S. mortgage market. I wanted to learn more about these entities and whether there were data available for the mortgage loans on their books. I found a great resource of downloadable data files on the Federal Housing Finance Agency (FHFA) [website](https://www.fhfa.gov/DataTools/Downloads/Pages/Public-Use-Databases.aspx).

The data file of interest contains national level loan data for single-family properties for the year 2019 from Fannie Mae. The downloaded zip file contained the data file, as well as a PDF file with descriptions of each of the variables in the file.

## Read in the data

The data file itself is space-delimited text file, which can be read into R using the `read_table()` function from the `tidyverse` package.  After peeking at the first few rows of the file from the command line, it appears there are no column names in the first row of the file.  The column names are generated in the code chunk below as relatively short versions of the "Field Names" in the PDF that accompanied the data.  The full field names are listed here:

- Enterprise Flag
- Record Number
- Metropolitan Statistical Area (MSA) Code
- 2010 Census Tract - Percent Minority
- Tract Income Ratio
- Borrower Income Ratio
- Loan-to-Value Ratio (LTV)
- Purpose of Loan
- Federal Guarantee
- Borrower Race or National Origin, and Ethnicity
- Co-Borrower Race or National Origin, and Ethnicity
- Borrower Gender
- Co-Borrower Gender
- Number of Units
- Unit - Affordability Category


```r
library(tidyverse)
library(colorspace)
library(wesanderson)
library(vcdExtra)
library(productplots)
library(RColorBrewer)
```


```r
varcols <-
  c(
    "enterprise",
    "record_no",
    "msa_code",
    "tract_pc_minority",
    "tract_inc_ratio",
    "borr_inc_ratio",
    "loan_to_value",
    "loan_purpose",
    "fed_guarantee",
    "borr_race_ethn",
    "coborr_race_eth",
    "borr_gender",
    "coborr_gender",
    "num_units",
    "afford_categ"
  )

loans <- read_table("fnma_sf2019a_loans.txt", col_names = varcols)
head(loans)
```

```
## # A tibble: 6 x 15
##   enterprise record_no msa_code tract_pc_minority tract_inc_ratio borr_inc_ratio
##        <dbl>     <dbl>    <dbl>             <dbl>           <dbl>          <dbl>
## 1          1         1        1                 1               1              3
## 2          1         2        1                 3               1              3
## 3          1         3        1                 2               2              3
## 4          1         4        1                 2               3              3
## 5          1         5        0                 1               3              2
## 6          1         6        1                 1               2              3
## # ... with 9 more variables: loan_to_value <dbl>, loan_purpose <dbl>,
## #   fed_guarantee <dbl>, borr_race_ethn <dbl>, coborr_race_eth <dbl>,
## #   borr_gender <dbl>, coborr_gender <dbl>, num_units <dbl>, afford_categ <dbl>
```

The first few records of the data frame show that every variable in the data is represented by a number.  The accompanying PDF file indicates that for each of these variables, the number in the data file corresponds to a category, so all of these variables are categorical.  Because the numbers in the file are difficult to work with as categories, they will be converted to text fields that are more descriptive.


```r
loans <-
  loans %>% mutate(
    enterprise = case_when(enterprise == 1 ~ "Fannie Mae",
                           enterprise == 2 ~ "Freddie Mac"),
    msa_code = case_when(
      msa_code == 0 ~ "non-metropolitan area",
      msa_code == 1 ~ "metropolitan area"
    ),
    tract_pc_minority = case_when(
      tract_pc_minority == 1 ~ "0-10%",
      tract_pc_minority == 2 ~ "10-30%",
      tract_pc_minority == 3 ~ "30-100%",
      tract_pc_minority == 9 ~ "Missing"
    ),
    tract_inc_ratio = case_when(
      tract_inc_ratio == 1 ~ "0-80%",
      tract_inc_ratio == 2 ~ "80-120%",
      tract_inc_ratio == 3 ~ ">120%",
      tract_inc_ratio == 9 ~ "Missing"
    ),
    borr_inc_ratio = case_when(
      borr_inc_ratio == 1 ~ "0-50%",
      borr_inc_ratio == 2 ~ "50-80%",
      borr_inc_ratio == 3 ~ ">80%",
      borr_inc_ratio == 9 ~ "Not applicable"
    ),
    loan_to_value = case_when(
      loan_to_value == 1 ~ "0-60%",
      loan_to_value == 2 ~ "60-80%",
      loan_to_value == 3 ~ "80-90%",
      loan_to_value == 4 ~ "90-95%",
      loan_to_value == 5 ~ ">95%",
      loan_to_value == 9 ~ "Missing"
    ),
    loan_purpose = case_when(
      loan_purpose == 1 ~ "Purchase",
      loan_purpose == 8 ~ "Other",
      loan_purpose == 9 ~ "Not applicable/not available"
    ),
    fed_guarantee = case_when(
      fed_guarantee == 1 ~ "FHA/VA",
      fed_guarantee == 2 ~ "Rural Housing Service (RHS)",
      fed_guarantee == 3 ~ "Home Equity Conversion Mortgage (HECM",
      fed_guarantee == 4 ~ "No Federal guarantee",
      fed_guarantee == 5 ~ "Title 1 - FHA"
    ),
    borr_race_ethn = case_when(
      borr_race_ethn == 1 ~ "American Indian or Alaska Native",
      borr_race_ethn == 2 ~ "Asian",
      borr_race_ethn == 3 ~ "Black or African American",
      borr_race_ethn == 4 ~ "Native Hawaiian or Other Pacific Islander",
      borr_race_ethn == 5 ~ "White",
      borr_race_ethn == 6 ~ "Two or more races",
      borr_race_ethn == 7 ~ "Hispanic or Latino",
      borr_race_ethn == 9 ~ "Not available/not applicable"
    ),
    coborr_race_eth = case_when(
      coborr_race_eth == 1 ~ "American Indian or Alaska Native",
      coborr_race_eth == 2 ~ "Asian",
      coborr_race_eth == 3 ~ "Black or African American",
      coborr_race_eth == 4 ~ "Native Hawaiian or Other Pacific Islander",
      coborr_race_eth == 5 ~ "White",
      coborr_race_eth == 6 ~ "Two or more races",
      coborr_race_eth == 7 ~ "Hispanic or Latino",
      coborr_race_eth == 9 ~ "Not available/not applicable"
    ),
    borr_gender = case_when(
      borr_gender == 1 ~ "Male",
      borr_gender == 2 ~ "Female",
      borr_gender == 3 ~ "Information not provided",
      borr_gender == 4 ~ "Not applicable",
      borr_gender == 9 ~ "Missing"
    ),
    coborr_gender = case_when(
      coborr_gender == 1 ~ "Male",
      coborr_gender == 2 ~ "Female",
      coborr_gender == 3 ~ "Information not provided",
      coborr_gender == 4 ~ "Not applicable",
      coborr_gender == 5 ~ "No co-borrower",
      coborr_gender == 9 ~ "Missing"
    ),
    afford_categ = case_when(
      afford_categ == 1 ~ "Low-income family in low-income area",
      afford_categ == 2 ~ "Very low-income family in low-income area",
      afford_categ == 3 ~ "Very low-income family not in low-income area",
      afford_categ == 4 ~ "Other",
      afford_categ == 9 ~ "Not available",
      afford_categ == 0 ~ "Missing"
    )
  )
```

Whew, that took a while to get to run correctly (I kept finding missing commas that were holding things up). Now I'll do a quick check of the data for null values.


```r
colSums(is.na(loans))
```

```
##        enterprise         record_no          msa_code tract_pc_minority 
##                 0                 0                 0                 0 
##   tract_inc_ratio    borr_inc_ratio     loan_to_value      loan_purpose 
##                 0                 0                 0                 0 
##     fed_guarantee    borr_race_ethn   coborr_race_eth       borr_gender 
##                 0                 0                 0              1317 
##     coborr_gender         num_units      afford_categ 
##                 0                 0                 0
```

Looks like there are only NA values in the `borr_gender` variable, which is the gender of the primary borrower.  I will decide what to do with these when we get to investigation of that variable.

## Data exploration

It might be good to pause here and summarize what some of my expectations are in the data.  Two of the most interesting variables in the data set to me are the gender and race/national origin/ethnicity variables.  There are also several variables in the data that, while being categorical, are actually representing numerical data that has simply been binned into ranges.  The loan-to-value variable, for instance, is a calculated ratio of the amount of the loan divided by the assessed value of the home.  It is close to one when the borrower has put little money down and has finance nearly the entire value of the home.

Because race in the United States is often related to class and wealth (or lack thereof), I would expect to see that white borrowers would tend to have lower loan-to-value ratios than black or Hispanic borrowers, since they would tend to have more money available for a down-payment.

It might also be interesting to look at gender and race/ethnicity/origin together, and see which race/ethnicity groups have the highest proportion of women as the primary borrower, and which have the lowest.  I don't know quite what I expect to see here, so I look forward to finding out.

### Race and Loan-to-Value Ratio

I'll start first with race or national origin and ethnicity versus the loan-to-value ratio.  Before making a visualization, let's look at a table of proportions for each race/ethnicity group and each loan-to-value category.


```r
round(prop.table(table(loans$borr_race_ethn, loans$loan_to_value)), 3)
```

```
##                                            
##                                              >95% 0-60% 60-80% 80-90% 90-95%
##   American Indian or Alaska Native          0.000 0.000  0.001  0.000  0.000
##   Asian                                     0.003 0.013  0.030  0.008  0.006
##   Black or African American                 0.008 0.005  0.013  0.005  0.007
##   Hispanic or Latino                        0.016 0.016  0.037  0.012  0.016
##   Native Hawaiian or Other Pacific Islander 0.000 0.000  0.001  0.000  0.000
##   Not available/not applicable              0.009 0.028  0.060  0.017  0.015
##   Two or more races                         0.001 0.001  0.003  0.001  0.001
##   White                                     0.055 0.126  0.308  0.088  0.087
```

These proportions are interesting, but they are proportions of the total.  I'm interested in seeing which race/ethnicity groups have the highest proportion of loans *within* the group being high loan-to-value.  The easiest way to see this might just be to plot it using the `fill` option for position in the ggplot call.


```r
ggplot(data = loans) +
  geom_bar(aes(x=borr_race_ethn, fill=loan_to_value), position="fill") + coord_flip()
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-6-1.png" width="672" />

No we can see, for instance, what relative proportion of all loans by Asian borrowers had loan-to-value ratio greater than 95%.  Let's reorder the race/ethnicity groups to show a progression from highest to lowest relative proportion, and while we are at it, let's get rid of the "NA" category.  Also, the legend categories are out of order, so I'll reorder those by refactoring the variable.


```r
race_order_new <- c(
  "Asian",
  "White",
  "Native Hawaiian or Other Pacific Islander",
  "American Indian or Alaska Native",
  "Hispanic or Latino",
  "Black or African American"
)

# Refactor the race variable to make bar graph easier to interpret
loans$borr_race_ethn <-
  factor(loans$borr_race_ethn, levels = race_order_new)

# Refactor the loan-to-value variable to correct order
loans$loan_to_value <- factor(loans$loan_to_value,
                              levels = c("0-60%", "60-80%", "80-90%", "90-95%", ">95%"))

loans %>% filter(borr_race_ethn != "Not available/not applicable") %>%
  ggplot() +
  geom_bar(aes(x = borr_race_ethn, fill = loan_to_value), position = "fill") +
  theme_minimal() +
  labs(title="Race, national origin, or ethnicity and loan-to-value ratio",
       x="", y="Proportion", fill="Loan-to-value ratio") +
  coord_flip()
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-7-1.png" width="672" />

That's pretty interesting: I didn't necessarily expect to see Asian borrowers with the lowest proportion of high loan-to-value mortgages.  It's pretty clear to compare the length of the bars this way, but let's also plot the bars side-by-side instead to see if it's easier to compare the bar lengths that way.


```r
loans %>%
  filter(borr_race_ethn != "Not available/not applicable") %>%
  count(borr_race_ethn, loan_to_value) %>%
  group_by(borr_race_ethn) %>%
  mutate(Sum = sum(n)) %>%
  mutate(Proportion = n / Sum) %>%
  ggplot(aes(x = borr_race_ethn, y = Proportion, fill = loan_to_value)) +
  geom_col(position = "dodge") +
  scale_y_continuous(labels = scales::percent_format(accuracy=1)) +
  scale_fill_manual(name = "Loan to Value Ratio", 
                    values = wes_palette("Moonrise3", n =5)) +
  #scale_fill_discrete_sequential("Purple-Oran", rev=FALSE) +
  theme_minimal() +
  theme(axis.title.y = element_blank()) +
  labs(title = "Loan-to-value ratio by race, national origin and ethnicity",
       x = "", y="",
       caption = "A higher proportion of black and Hispanic 
       borrowers have high loan-to-value mortgages.") +
  coord_flip()
```

<img src="{{< blogdown/postref >}}index_files/figure-html/unnamed-chunk-8-1.png" width="672" />

This tells an interesting story about how black or African American and Hispanic or Latino borrowers tend to carry riskier mortgages, which are often saddled with higher interest rates, as compared to whites and Asian borrowers.  Since loan-to-value ratio is a measure of how much is being borrowed relative to the assessed value of the home, a higher number generally means the borrower had less to put down as a down-payment, and often incurs a higher interest rate and additional private mortgage insurance.  Native Hawaiian and other Pacific Islander borrowers have a similar profile to white and Asian borrowers in this plot, with a tendency toward lower loan-to-value ratios.






