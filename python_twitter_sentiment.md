# Twitter and sentiment analysis

<!-- TOC -->

- [Twitter and sentiment analysis](#twitter-and-sentiment-analysis)
    - [Twitter API wrapper](#twitter-api-wrapper)
        - [Setup](#setup)
        - [Retrieve home feeds](#retrieve-home-feeds)
        - [Retrieve tweets of a specific user](#retrieve-tweets-of-a-specific-user)
        - [Get user account information of a specific user](#get-user-account-information-of-a-specific-user)
        - [Search for tweets from query](#search-for-tweets-from-query)
        - [Tweeting out](#tweeting-out)
        - [Create friendship and send direct message](#create-friendship-and-send-direct-message)
        - [Retweet](#retweet)
        - [Delete tweet](#delete-tweet)
        - [Simple pagination](#simple-pagination)
        - [Search with real person filter and advanced pagination](#search-with-real-person-filter-and-advanced-pagination)
        - [Retrieve tweet timestamps](#retrieve-tweet-timestamps)
        - [Multi tweets with timer](#multi-tweets-with-timer)
    - [VADER-Sentiment-Analysis](#vader-sentiment-analysis)

<!-- /TOC -->

## Twitter API wrapper

[**documentation**](http://docs.tweepy.org/en/v3.5.0/api.html)

### Setup

```python 
# Dependencies
import json
import tweepy 

# Import Twitter API Keys
from config import consumer_key, consumer_secret, access_token, access_token_secret

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

```

### Retrieve home feeds

```python
# Get all tweets from home feed
public_tweets = api.home_timeline()

# Loop through all tweets
for tweet in public_tweets:
    # Utilize JSON dumps to generate a pretty-printed json
    print(json.dumps(tweet, sort_keys=True, indent=4))
```

### Retrieve tweets of a specific user

```python
# Target User Account
target_user = "@TheStressGuys"

# Counter
counter = 1

# Get all tweets from home feed
public_tweets = api.user_timeline(target_user)

for tweet in public_tweets:

    # Print Tweets
    print(f'Tip {counter}: {tweet["text"]}')

    # Add to Counter
    counter = counter + 1
```

### Get user account information of a specific user

```python
target_user = "Taylorswift13"
user_account = api.get_user(target_user)
user_real_name = user_account["name"]

# Get the specific data
user_tweets = user_account["statuses_count"]
user_followers = user_account["followers_count"]
user_following = user_account["friends_count"]
user_favorites = user_account["favourites_count"]
```

### Search for tweets from query

```python
# Target Term
target_term = "@DanTurcza"

# Retrieve 100 most recent tweets
public_tweets = api.search(target_term, count=100, result_type="recent")

# Loop through all tweets
for tweet in public_tweets["statuses"]:

    # Get ID and Author of most recent tweet
    tweet_id = tweet["id"]
    tweet_author = tweet["user"]["screen_name"]
    tweet_text = tweet["text"]
```

### Tweeting out

```python
# Create a status update
api.update_status("Hey! I'm tweeting programmatically!")

# Create a status update
api.update_with_media("../Resources/too-much-big-data.jpg",
                      "And now... I just tweeted an image programmatically!")
```

### Create friendship and send direct message

```python
# Create a friendship with another user
api.create_friendship(screen_name="@PlotBot5", follow=True)

# Send a direct message to another user
# (Hint: You will need them to follow your account)
api.send_direct_message(user="plotbot5", text="hiiiiiii!!!!!!")
```

### Retweet

```python
# Retweet any tweet from someone else's account 
# (You will need to locate a tweet's id)
target_user = "@ddjournalism"
public_tweets = api.user_timeline(target_user)
tweet_id = public_tweets[0]["id"]
api.retweet(tweet_id)
```

### Delete tweet

```python
my_tweets = api.user_timeline()
tweet_id = my_tweets[0]["id"]
api.destroy_status(tweet_id)
```

### Simple pagination

```python
# Target User
target_user = "GuardianData"

# Tweet Texts
tweet_texts = []

# Create a loop to iteratively run API requests
for x in range(1, 11):

    # Get all tweets from home feed (for each page specified)
    public_tweets = api.user_timeline(target_user, page=x)

    # Loop through all tweets
    for tweet in public_tweets:

        # Print Tweet
        print(tweet["text"])

        # Store Tweet in Array
        tweet_texts.append(tweet["text"])
```

### Search with real person filter and advanced pagination

```python
# "Real Person" Filters
min_tweets = 5
max_tweets = 10000
max_followers = 2500
max_following = 2500
lang = "en"

# Search for People Tweeting about Mark Hamill
search_term = "Mark Hamill"

# Create variable for holding the oldest tweet
oldest_tweet = None

# List to hold unique IDs
unique_ids = []

# Counter to keep track of the number of tweets retrieved
counter = 0

# Loop through 5 times (total of 500 tweets)
for x in range(5):

    # Retrieve 100 most recent tweets -- specifying a max_id
    public_tweets = api.search(search_term, 
                               count=100, 
                               result_type="recent", 
                               max_id=oldest_tweet)

    # Print Tweets
    for tweet in public_tweets["statuses"]:

        tweet_id = tweet["id"]

        # Use filters to check if user meets conditions
        if (tweet["user"]["followers_count"] < max_followers and
            tweet["user"]["statuses_count"] > min_tweets and
            tweet["user"]["statuses_count"] < max_tweets and
            tweet["user"]["friends_count"] < max_following and
                tweet["user"]["lang"] == lang):

            # Print the username
            print(tweet["user"]["screen_name"])

            # Print the tweet id
            print(tweet["id_str"])

            # Print the tweet text
            print(tweet["text"])
            print()

            # Append tweet_id to ids list if it doesn't already exist
            # This allows checking for duplicate tweets
            if tweet_id not in unique_ids:
                unique_ids.append(tweet_id)

            # Increase counter by 1
            counter += 1

        # Reassign the the oldest tweet (i.e. the max_id)
        # Subtract 1 so the previous oldest isn't included
        # in the new search
        oldest_tweet = tweet_id - 1

```

### Retrieve tweet timestamps

```python
# Target User Account
target_user = "latimes"

# Get all tweets from home feed
public_tweets = api.user_timeline(target_user)
tweet_times = []

for tweet in public_tweets:

    # timestamps
    raw_time = tweet["created_at"]
    tweet_times.append(raw_time)

    # Add to Counter
    counter = counter + 1

# Convert tweet timestamps to datetime objects that can be manipulated by Python
converted_timestamps = []
for raw_time in tweet_times:
    converted_time = datetime.strptime(raw_time, "%a %b %d %H:%M:%S %z %Y")
    converted_timestamps.append(converted_time)

# Time difference between tweets
time_diffs = []

for x in range(converted_length - 1):
    time_diff = converted_timestamps[x] - converted_timestamps[x + 1]

    # convert time_diff to hours
    time_diff = time_diff.seconds / 3600
    time_diffs.append(time_diff)

print(f"Avg. Hours Between Tweets: {np.mean(time_diffs)}")
```

### Multi tweets with timer

```python
# Dependencies
import tweepy
import json
import time
from config import consumer_key, consumer_secret, access_token, access_token_secret

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

# Create a function that tweets
def TweetOut(tweet_number):

    api.update_status("This is tweet #%s, but with a timer man!" % tweet_number)

# Create a loop that calls the TweetOut function every minute
counter = 0

# Infinite loop
while(True):

    # Call the TweetQuotes function and specify the tweet number
    TweetOut(counter)

    # Once tweeted, wait 60 seconds before doing anything else
    time.sleep(60)

    # Add 1 to the counter prior to re-running the loop
    counter = counter + 1
```



## VADER-Sentiment-Analysis

https://github.com/cjhutto/vaderSentiment

```python
# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Run analysis
results = analyzer.polarity_scores(target_string)

# Fetch scores
compound = results["compound"]
pos = results["pos"]
neu = results["neu"]
neg = results["neg"]
```
