import csv
import tweepy
import ssl
import random

# Oauth keys
consumer_key = "***************"
consumer_secret = "***************"
access_token = "***************"
access_token_secret = "***************"


# Authentication with Tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
ssl._create_default_https_context = ssl._create_unverified_context
api = tweepy.API(auth)
api = tweepy.API(auth, wait_on_rate_limit=True)

user = api.me()
print (user.name)

# Candidates information 
tweet_id_Lasso = '1333803090852528138'
username_Lasso = 'LassoGuillermo'
tweet_id_Arauz = '1334120637212860419'
username_Arauz = 'ecuarauz'

# Opening a csv file
csvFile = open('tweets_lasso.csv', 'a')
csvWriter = csv.writer(csvFile)

# Obtaining the ids of the last 50 tweets ofcandidates
def retrive_random50_tweet_ids(name, first_tweet):
    tweet_ids = []
    for tweet in tweepy.Cursor(api.user_timeline,
                           screen_name=name,
                           since_id=first_tweet,
                           exclude_replies=True).items():
            tweet_ids.append(str(tweet.id))
    tweet_ids = random.sample(tweet_ids, 50)
    return tweet_ids

arauz_tweet_ids=retrive_random50_tweet_ids(username_Arauz, tweet_id_Arauz)
lasso_tweet_ids=retrive_random50_tweet_ids(username_Lasso, tweet_id_Lasso)
print(lasso_tweet_ids)

# Retriving the replies to the last 50 tweets of a candidate and save it in a csv file
for tweet in lasso_tweet_ids:
    print(tweet)
    for reply in tweepy.Cursor(api.search,q='to:'+username_Lasso, tweet_mode = 'extended', timeout=6000, retry=True).items(1000):
        if hasattr(reply, 'in_reply_to_status_id_str'):
            if (reply.in_reply_to_status_id_str==tweet):
                csvWriter.writerow([reply.id, reply.created_at, reply.full_text])