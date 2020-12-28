# Imports
library(rtweet)
library(dplyr)
library(data.table)

# Oauth keys and authentication
create_token(
  app = "**********",
  consumer_key = "**************",
  consumer_secret = "*****************",
  access_token = "****************",
  access_secret = "*****************",
  set_renv = TRUE)

# Fetching all the replies to candidates
replies_arauz = search_tweets('to:ecuarauz', sinceID = "1334120637212860419", n=10000, include_rts = FALSE, lang = "es") 

replies_lasso = search_tweets('to:LassoGuillermo', sinceID = "1333803090852528138", n=10000, include_rts = FALSE, lang = "es") 


# Obtaining random sample of 1000 replies
replies_arauz_sample <- sample_n(replies_arauz, 1000) 

replies_lasso_sample <- sample_n(replies_lasso, 1000) 

# Separating the columns of datasets that have list as data type in order to export the file 

replies_arauz_sample$listed_count <- vapply(replies_arauz_sample$listed_count, paste, collapse = ", ", character(1L))
sapply(replies_arauz_sample, class)

replies_lasso_sample$mentions_user_id <- vapply(replies_lasso_sample$mentions_user_id, paste, collapse = ", ", character(1L))
sapply(replies_lasso_sample, class)

# Exporting the data in csv file
write.csv(replies_arauz_sample,'replies_random1000_arauz_Diciembre.csv')

write.csv(replies_lasso_sample,'replies_random1000_lasso_Diciembre.csv')

