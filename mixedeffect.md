## 'Insight Project: 
### Fitbit Sleep and Mixed Effects Models'
#### Spring 2016

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

### Introduction

We would like to examine if there is a reasonable justification for examining how sleep and activity are related. To do this, we have collected a much data as possible from as many subjects as possible and will use mixed effect modeling to examine how strong this relationship is across the population.

We have collected data from 11 subjects. For each user, we have access to 75 days worth of sleep and activity data (note: if there are missing days for a given user, then there we will have fewer observations; i.e. we do not know what the API contains until after accessing it). Therefore, there will be multiple responses for each user and should not be regarded as independent events. Furthermore, each person should be expected to have different sleep habits which will affect all data points for that subject. Therefore, these different responses will be inter-dependent rather than independent. To address this we add a **random effect** for each subject; that is, we resolve this non-independence by assuming a 'baseline' sleep pattern for each user. This is the basis of the 'mixed-effects model'. Here the 'fixed effects' are meant to be the features of sleep, and the `mixed-effect' is the subject. 

Loosely, the model we consider takes the following form 

$$
\text{activity} \sim \text{sleep} + (1~|~\text{subject}) + \epsilon.
$$

In short, this says there are multiple data points for each subject. That is to day, we are disregarding by-subject variation. It is also reasonable to assume that the amount of sleep is affected by the day of the week. That is, just it being Saturday or Sunday may lead to greater sleep amounts than during the weekdays. These can be thought of as 'by-item' variation. It isn't necessarily the purpose of this model to account for these types of variation (and very likely we will remove weekdays from our analysis) but this is something to consider. In this case, the model would take the form

$$
\text{activity} \sim \text{sleep} + (1~|~\text{subject}) + (1~|~\text{day of the week}) + \epsilon.
$$

In essence, if the previous model allows for different y-intercepts for different subjects, then this refined model allows for different y-intercepts for different days of the week. Note, we have used one-hot encoding to encode this categorical information. We discuss now the various features measured for sleep and activity. 

### Data and Features
For each user we have measured the following variables

#####For sleep:

* minuteStartTime: the time the user went to sleep
* minutesAsleep: the duration of their 'main sleep'
* StartTime2: higher order term given by (minuteStartTime)^2
* minutesAsleep2: higher order term given by (minutesAsleep)^2
* StartTimeMinutesAsleep: higher order term given by (minuteStartTime)(minutesAsleep)
* yestday_StartTime: the time the user went to sleep yesterday
* yestday_minutesAsleep: the duration of yesterday's sleep

#####For activity:

* activityCalories: logged activity calories for the day
* caloriesOut: logged calories out for the day
* steps: user's step count 
* step_goal: 0 or 1, depending on if the user met their step goal

#####Weekdays: one-hot encoded as categorical variables

* X0: Monday
* X1: Tuesday
* X2: Wednesday
* X3: Thursday
* X4: Friday
* X5: Saturday
* X6: Sunday

###Applying mixed models
In this section we apply some linear mixed effects models. Start by installing necessary packages.

```{r}
library(lme4)
library(nlme)
```

...and read in the data....

```{r}
setwd("~/Dropbox/DS/InsightHealthDS/fitbit/")
df = read.csv("total_df_scaled.csv")
head(df)
```

Note that users are identified by the 'id' variable, ranging from 1 to 11. Each of the continuous variables above has been normalized. Also, look at preliminary information about the dataframe. 

```{r}
dim(df)
which(is.na(df))
```

We have already cleaned the data to remove any missing values. 

####Comparing sleep and steps
To begin, let's apply a mixed effect model for steps and the user's start time for sleep and duration, controlling for user. 

```{r}
step.model = lmer(steps ~ minutesAsleep + (1 | id), data = df)
summary(step.model)
```

Looking at the output we see information for fixed effects and random effects. Some quick definitions: a **random effect** is generally somethng that can be expected to have non-systematic, unpredictable or 'random' influence on your data. For us, that is the subject, or 'id'. In short, we want to generalize over these unpredictable aspects of individuals. 

A **fixed effect** is a variable that is expected to have systematic and predictable influence over the data. For use, these are the measurements involving sleep. Another way to think of fixed effects is that they cover all possible 'levels of a factor' (like, 'male/female' cover all possible levels of 'gender'). Here we are thinking sleep bedtime and sleep duration cover all possible levels of the sleep factor. This is worth considering. 

Looking at the *Random effects* section, we see the variance and st. dev of the dependent measure of id. The 'Residual' stands for the variability that's not due to id. This is the $\epsilon$ in the model above.

Looking at the *Fixed effects* section we can see the slope for the sleep duration. It is negative which makes sense??

Let's see how this changes when considering two variables for sleep. 

```{r}
step.model = lmer(steps ~ minuteStartTime + minutesAsleep + (1 | id), data = df)
summary(step.model)
```

One thing to notice is that the Residual term in the Random effects column is only slightly smaller. This likely means that the variation due to sleep duration was not confoucing with the variation due to sleep start time.

Also note, the slope for start time of sleep is positive but still the slope for duration is negative (but small) which could be better justified. The intercept is practically zero, perhaps it is worth accounting for sex as a fixed effect too? This would require more information. I could probably get this from the Fitbit API...

We can enhance this model by adding all variables for sleep. We will discuss more interaction terms later. 

```{r}
full_step.model = lmer(steps ~ minuteStartTime + minutesAsleep + StartTime2
                      + minutesAsleep2 + StartTimeMinutesAsleep + yestday_StartTime
                      + yestday_minutesAsleep + (1 | id), data = df)
summary(full_step.model)
```

This is particularly interesting to me as I would expect the *coefficient for the order two terms to be negative*. This leads me to believe that we could optimize this function across the variables to determine optimal sleep for each user. 

####Measuring statistical significance
In this section, we explore the statistical significance of the effect of sleep on activity; that is, we compute a $p$-value for our models above. Note that here the $p$-values for mixed models are slightly different than regular fixed effects linear models. There are various approaches to doing this. We'll use a Likelihood Ratio Test. 

Simply put, we consider two models: one without the factor I'm interested in (this is the *null model*) and one with the facotr I'm interested in. We'll then compare these two models to determine if there is a statistically signficant difference between them; that is, we'll check if the difference between the likelihood of the two models is significant.

First, we construct the null model which has no sleep data. This is saying we're just assuming there is a difference in the step behavior for different users with no dependence on sleep.
```{r}
step.null = lmer(steps ~ (1 | id), data = df,
                  REML = FALSE)
```
Note here, we include the argument REML = FALSE. This is necessary when using the likelihood ratio test. Next, we consider the full model from above, 
```{r}
full_step.model = lmer(steps ~ minuteStartTime + minutesAsleep + StartTime2
                      + minutesAsleep2 + StartTimeMinutesAsleep + yestday_StartTime
                      + yestday_minutesAsleep + (1 | id), data = df,
                      REML = FALSE)

```
We can now compare these two models using the likelihood ratio test using an ANOVA. 
```{r}
anova(step.null, full_step.model)
```
One quick note about the Chi-square test here. There is a theorem called **Wilk's Theorem** which states that twice the log likelihood ratio of two models approaches the Chi-Square distribution with the number of degrees of freedom equal to the number of parameters that differ between the models. (question: what are the consequences of using more parameters in the models?)

####Interaction terms
We're also somewhat interested in interaction terms, that is we want to consider the possibility that sleep bedtime and sleep duration are interdependent on each other in some way (seems reasonable!) To analyze this a bit more, let's repeat the analysis above but in a more restricted setting. Another thing to consider is the possibility of an interaction between the two variables between sleep duration and bed time. To look into this, consider the null model

```{r}
step.null = lmer(steps ~ (1 | id), data = df,
                  REML = FALSE)
```

Then consider the model with interacting terms...
```{r}
int_step.model = lmer(steps ~ minuteStartTime*minutesAsleep + (1 | id), data = df,
                  REML = FALSE)
```
...and compare the two models using an anova again. 
```{r}
anova(step.null, int_step.model)
```
Note, here the $p$-value is also approaching significance so it seems justifiable to include interaction terms in our model above. 

With this in mind perhaps we should consider compare the  null model against the following:
```{r}
int_full_step.model = lmer(steps ~ minuteStartTime + minutesAsleep + StartTime2
                      + minutesAsleep2 + minutesAsleep*minuteStartTime + yestday_StartTime
                      + yestday_minutesAsleep + (1 | id), data = df,
                      REML = FALSE)

```
Comparing by the ANOVA, it still seems significant. 
```{r}
anova(step.null, int_full_step.model)
```

#### Conclusion
It seems reasonable to look into how sleep affects activity (in this case, measured by steps). There are many other measurements for activity given by Fitbit. It may be worthwhile to look into each of these factors and see which gives the strongest deviation from the null hypothesis. We may do this depending on time. 

Also, we haven't had a chance to examine mixed effects for day of the week either. These would be interesting to examine as well.
