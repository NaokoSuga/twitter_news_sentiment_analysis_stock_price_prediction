# Stock Prediction Based off Public Sentiment

###### Mod 3 Project: Built classification model to predict probability of stock price moving up/down based on market conditions and news sentiment


## Summary
For my Mod 3 Project my partner and I analyzed the impact the public sentiment has on a stock price. We analyzed 3 major companies (Netflix, Facebook and Apple). We experimented with several shallow learning classification models and found that Adaptive Boosting and Gradient Boosting produced the best accuracy.


## Objectives
1. Find out which News sources overall sentiment were the most impactful when predicting if a stock price would increase or decrease

2. See if twitter sentiment has more of an impact than reputable news sources

3. Predict if a stock price will close higher or lower than it opened in 30 minute intervals


### Data Gathering
Overall we used 3 different API's for the following purposes.
  1. NewsAPI to get all articles about a company for the past week https://newsapi.org/sources. (about 35 articles per company)
  2. Twitter API to get tweets about a company for the past week (about 20,000 tweets per company)
  3. Alpha Vantage Stock API to get the stock price of each company at 5 minute and 30 minute intervals for the past week https://www.alphavantage.co/documentation/.


### Feature Engineering

Once we gathered all tweets and articles we created multiple features for our model. Each was based on the vader sentiment (from NLTK) calculated on the scraped text  These included:

#### Tweets

  1. Time Windows = past 30 minutes, 60 minutes, 120 minutes, period from after stock market closes to midnight and 12PM to 9AM (market opens)
  2. Metrics per time window = count of negative tweets, count of positive tweets, count of neutral tweets, average sentiment.

#### News Articles

  1. Time Windows = past day and past two days for all 8 sources (ABC News, Business Insider, Reuters, NBC News, The New York Times, Techcrunch, Wired)
  2. Metrics per time window = average vader score of negative tweets, average vader score of positive tweets, average vader score of neutral tweets, average compounded vader score.

Below is a heat map of the pearson correlation between a stock going up in a 30 minute interval and each feature. This shows:
  1. For tweets the highest positive correlation between a stock going up is the average sentiment the night before and the morning of.
  2. For news articles the highest positive correlation between a stock going up is compounded score over the past day.
    + On a source by source the one day Wired, two day Techcrunch and one day New York times sentiment had the highest correlation

![alt text](https://github.com/NaokoSuga/twitter_news_sentiment_analysis_stock_price_prediction/blob/master/Screenshots/heatmap.png?raw=True)

### Classification Models

NOTE: Our data was very limited due the fact that you can only pull tweets for the past week if using the free membership of twitter's API. We identify that our models are overfit even when using cross validation as we had very high dimensionality post feature engineering but only a weeks worth of data points.

Steps to fix this moving forward:
1. Add more data by adding tweets to a database as they come in
2. Use PCA to reduce dimensionality (we had not covered this concept at this point in the course)

Adaptive boosting and Gradient Boosting were our most accurate models and had the best area under the ROC curve (AUC). Both of these metrics can be seen visualized below:

![alt text](https://github.com/NaokoSuga/twitter_news_sentiment_analysis_stock_price_prediction/blob/master/Screenshots/models.png?raw=True)

![alt text](https://github.com/NaokoSuga/twitter_news_sentiment_analysis_stock_price_prediction/blob/master/Screenshots/auc.png?raw=True)

As you can see in the discrepancy between the train accuracy and test accuracy our models were overfit. We need to add more data to work through this and reduce the bias.

### Feature Importances

Although the models were overfit we did gain some interesting insights from the feature importances of our Adaptive Boosting Model

#### 1. All Features Included

Below shows the feature importances of all features included. Clearly the standard stock market metrics (black) are still the most important regardless of the public sentiment.

![alt text](https://github.com/NaokoSuga/twitter_news_sentiment_analysis_stock_price_prediction/blob/master/Screenshots/allfeat.png?raw=True)

#### 2. Only Twitter Features Included

We then fit an Adaptive Boosting model only including the twitter features. Overall the metrics in the 30 minute time window had the highest importances

![alt text](https://github.com/NaokoSuga/twitter_news_sentiment_analysis_stock_price_prediction/blob/master/Screenshots/twitter.png?raw=True)


#### 3. Only News Features Included

We also fit an AdaBoost model only including the news features. Overall the metrics in the one day time window had the highest importances

![alt text](https://github.com/NaokoSuga/twitter_news_sentiment_analysis_stock_price_prediction/blob/master/Screenshots/news.png?raw=True)

### ARIMA Models

Finally, we fit 4 different SARIMAX models using different combinations of P (number of auto regressive terms), D (number of differences) and Q (number of moving average) by using a grid search. TO BE CONTINUED

![alt text](https://github.com/NaokoSuga/twitter_news_sentiment_analysis_stock_price_prediction/blob/master/Screenshots/ARIMA.png?raw=True)
