
# coding: utf-8

# 1. Problem definition
# ==
# 
# <span style="background-color: #F9EBEA; color:##C0392B">AirBnB</span> is a marketplace for short term rentals that allows you to list part or all of your living space for others to rent. You can rent everything from a room in an apartment to your entire house on AirBnB. Because most of the listings are on a short-term basis, AirBnB has grown to become a popular alternative to hotels. The company itself has grown from it's founding in 2008 to a 30 billion dollar [valuation in 2016](http://www.bloomberg.com/news/articles/2016-08-05/airbnb-files-to-raise-850-million-at-30-billion-valuation) and is currently worth more than any hotel chain in the world.
# 
# One challenge that hosts looking to rent their living space face is determining the optimal nightly rent price. In many areas, renters are presented with a good selection of listings and can filter on criteria like price, number of bedrooms, room type and more. Since AirBnB is a marketplace, the amount a host can charge on a nightly basis is closely linked to the dynamics of the marketplace. Here's a screenshot of the search experience on AirBnB:
# 
# <img width="600" alt="creating a repo" src="https://drive.google.com/uc?export=view&id=1-XClY8MpJhG52q9ovWgqU74wKlJA7ZOS">
# 
# 
# As a host, if we try to charge above market price for a living space we'd like to rent, then renters will select more affordable alternatives which are similar to ours.. If we set our nightly rent price too low, we'll miss out on potential revenue.
# 
# One strategy we could use is to:
# 
# - find a few listings that are similar to ours,
# - average the listed price for the ones most similar to ours,
# - set our listing price to this calculated average price.
# 
# The process of discovering patterns in existing data to make a prediction is called **machine learning**. In our case, we want to use data on local listings to predict the optimal price for us to set. In this mission, we'll explore a specific machine learning technique called **k-nearest neighbors**, which mirrors the strategy we just described. Before we dive further into machine learning and k-nearest neighbors, let's get familiar with the dataset we'll be working with.

# 2. Introduction to the data
# ==
# 
# While AirBnB doesn't release any data on the listings in their marketplace, a separate group named [Inside AirBnB](http://insideairbnb.com/get-the-data.html) has extracted data on a sample of the listings for many of the major cities on the website. In this post, we'll be working with their dataset from October 3, 2015 on the listings from Washington, D.C., the capital of the United States. Here's a [direct link to that dataset](http://data.insideairbnb.com/united-states/dc/washington-dc/2015-10-03/data/listings.csv.gz). Each row in the dataset is a specific listing that's available for renting on AirBnB in the Washington, D.C. area
# 
# To make the dataset less cumbersome to work with, we've removed many of the columns in the original dataset and renamed the file to dc_airbnb.csv. Here are the columns we kept:
# 
# - <span style="background-color: #F9EBEA; color:##C0392B">host_response_rate</span>: the response rate of the host
# - <span style="background-color: #F9EBEA; color:##C0392B">host_acceptance_rate</span>: number of requests to the host that convert to rentals
# - <span style="background-color: #F9EBEA; color:##C0392B">host_listings_count</span>: number of other listings the host has
# - <span style="background-color: #F9EBEA; color:##C0392B">latitude</span>: latitude dimension of the geographic coordinates
# - <span style="background-color: #F9EBEA; color:##C0392B">longitude</span>: longitude part of the coordinates
# - <span style="background-color: #F9EBEA; color:##C0392B">city</span>: the city the living space resides
# - <span style="background-color: #F9EBEA; color:##C0392B">zipcode</span>: the zip code the living space resides
# - <span style="background-color: #F9EBEA; color:##C0392B">state</span>: the state the living space resides
# - <span style="background-color: #F9EBEA; color:##C0392B">accommodates</span>: the number of guests the rental can accommodate
# - <span style="background-color: #F9EBEA; color:##C0392B">room_type</span>: the type of living space (Private room, Shared room or Entire home/apt
# - <span style="background-color: #F9EBEA; color:##C0392B">bedrooms</span>: number of bedrooms included in the rental
# - <span style="background-color: #F9EBEA; color:##C0392B">bathrooms</span>: number of bathrooms included in the rental
# - <span style="background-color: #F9EBEA; color:##C0392B">beds</span>: number of beds included in the rental
# - <span style="background-color: #F9EBEA; color:##C0392B">price</span>: nightly price for the rental
# - <span style="background-color: #F9EBEA; color:##C0392B">cleaning_fee</span>: additional fee used for cleaning the living space after the guest leaves
# - <span style="background-color: #F9EBEA; color:##C0392B">security_deposit</span>: refundable security deposit, in case of damages
# - <span style="background-color: #F9EBEA; color:##C0392B">minimum_nights</span>: minimum number of nights a guest can stay for the rental
# - <span style="background-color: #F9EBEA; color:##C0392B">maximum_nightss</span>: maximum number of nights a guest can stay for the rental
# - <span style="background-color: #F9EBEA; color:##C0392B">number_of_reviews</span>: number of reviews that previous guests have left
# 
# Let's read the dataset into Pandas and become more familiar with it.
# 
# 
# <br>
# <div class="alert alert-info">
# <b>Exercise Start.</b>
# </div>
# 
# **Description**: 
# 
# 1. Read <span style="background-color: #F9EBEA; color:##C0392B">dc_airbnb.csv</span> into a Dataframe named <span style="background-color: #F9EBEA; color:##C0392B">dc_listings</span>.
# 2. Use the print function to display the first row in <span style="background-color: #F9EBEA; color:##C0392B">dc_listings</span>.
# 

# In[10]:

import pandas as pd
import numpy as np

dc_listings = pd.read_csv('dc_airbnb.csv')

dc_listings.head()


# 3. K-nearest neighbors
# ==
# 
# Here's the strategy we wanted to use:
# 
# - Find a few similar listings.
# - Calculate the average nightly rental price of these listings.
# - Set the average price as the price for our listing.
# 
# The k-nearest neighbors algorithm is similar to this strategy. Here's an overview:
# 
# <img width="900" alt="creating a repo" src="https://drive.google.com/uc?export=view&id=1b3uN7WtvbamsIVxYYML1cXbWTXvs6UGb">
# 
# 
# There are 2 things we need to unpack in more detail:
# 
# - the similarity metric
# - how to choose the <span style="background-color: #F9EBEA; color:##C0392B">k</span> value
# 
# In this mission, we'll define what similarity metric we're going to use. Then, we'll implement the k-nearest neighbors algorithm and use it to suggest a price for a new, unpriced listing. We'll use a <span style="background-color: #F9EBEA; color:##C0392B">k</span> value of <span style="background-color: #F9EBEA; color:##C0392B">5</span> in this mission. In later missions, we'll learn how to evaluate how good the suggested prices are, how to choose the optimal <span style="background-color: #F9EBEA; color:##C0392B">k</span> value, and more.
# 

# 4. Euclidean distance
# ==
# 
# The similarity metric works by comparing a fixed set of numerical **features**, another word for attributes, between 2 **observations**, or living spaces in our case. When trying to predict a continuous value, like price, the main similarity metric that's used is **Euclidean distance**. Here's the general formula for Euclidean distance:
# 
# $\displaystyle d = \sqrt{(q_1 - p_1)^2 + (q_2 - p_2)^2 + \ldots + (q_n - p_n)^2}$
# 
# where $q_1$ to $q_n$ represent the feature values for one observation and $p_1$ to $p_n$ represent the feature values for the other observation. Here's a diagram that breaks down the Euclidean distance between the first 2 observations in the dataset using only the <span style="background-color: #F9EBEA; color:##C0392B">host_listings_count</span>, <span style="background-color: #F9EBEA; color:##C0392B">accommodates</span>, <span style="background-color: #F9EBEA; color:##C0392B">bedrooms</span>, <span style="background-color: #F9EBEA; color:##C0392B">bathrooms</span>, and <span style="background-color: #F9EBEA; color:##C0392B">beds</span> columns:
# 
# 
# <img width="900" alt="creating a repo" src="https://drive.google.com/uc?export=view&id=15wH6nSdX74TEKIFeBqNrMceiwoCk5j3s">
# 
# In this mission, we'll use just one feature in this mission to keep things simple as you become familiar with the machine learning workflow. Since we're only using one feature, this is known as the **univariate case**. Here's how the formula looks like for the univariate case:
# 
# $\displaystyle d = \sqrt{(q_1 - p_1)^2}$
# 
# The square root and the squared power cancel and the formula simplifies to:
# 
# $ \displaystyle d = \left | q_1 - p_1 \right |$
# 
# The living space that we want to rent can accommodate 3 people. Let's first calculate the distance, using just the <span style="background-color: #F9EBEA; color:##C0392B">accommodates</span> feature, between the first living space in the dataset and our own.
# 
# 
# <br>
# <div class="alert alert-info">
# <b>Exercise Start.</b>
# </div>
# 
# **Description**: 
# 
# 1. Calculate the Euclidean distance between our living space, which can accommodate 3 people, and the first living space in the <span style="background-color: #F9EBEA; color:##C0392B">dc_listings</span> Dataframe.
# 2. Assign the result to <span style="background-color: #F9EBEA; color:##C0392B">first_distance</span> and display the value using the <span style="background-color: #F9EBEA; color:##C0392B">print</span> function.

# In[11]:

first_distance = abs(dc_listings['accommodates'][0] - 3)

print(first_distance)


# 5. Calculate distance for all observations
# ==
# 
# The Euclidean distance between the first row in the <span style="background-color: #F9EBEA; color:##C0392B">dc_listings</span> Dataframe and our own living space is <span style="background-color: #F9EBEA; color:##C0392B">1</span>. How do we know if this is high or low? If you look at the Euclidean distance equation itself, the lowest value you can achieve is <span style="background-color: #F9EBEA; color:##C0392B">0</span>. This happens when the value for the feature is exactly the same for both observations you're comparing. If p1=q1, then $ \displaystyle d = \left | q_1 - p_1 \right |$ which results in $d=0$. The closer to <span style="background-color: #F9EBEA; color:##C0392B">0</span> the distance the more similar the living spaces are.
# 
# If we wanted to calculate the Euclidean distance between each living space in the dataset and a living space that accommodates <span style="background-color: #F9EBEA; color:##C0392B">8</span> people, here's a preview of what that would look like.
# 
# <img width="600" alt="creating a repo" src="https://drive.google.com/uc?export=view&id=1res4uO-8wP8_g7scMbr1kP594ZnOmk-y">
# 
# Then, we can rank the existing living spaces by ascending distance values, the proxy for similarity.
# 
# 
# <br>
# <div class="alert alert-info">
# <b>Exercise Start.</b>
# </div>
# 
# **Description**: 
# 
# 1. Calculate the distance between each value in the <span style="background-color: #F9EBEA; color:##C0392B">accommodates</span> column from <span style="background-color: #F9EBEA; color:##C0392B">dc_listings</span> and the value <span style="background-color: #F9EBEA; color:##C0392B">3</span>, which is the number of people our listing accommodates:
#     - Use the <span style="background-color: #F9EBEA; color:##C0392B">apply</span> method to calculate the absolute value between each value in <span style="background-color: #F9EBEA; color:##C0392B">accommodates</span> and <span style="background-color: #F9EBEA; color:##C0392B">3</span> and return a new Series containing the distance values.
# 2. Assign the distance values to the <span style="background-color: #F9EBEA; color:##C0392B">distance</span> column.
# 3. Use the Series method <span style="background-color: #F9EBEA; color:##C0392B">value_counts</span> and the <span style="background-color: #F9EBEA; color:##C0392B">print</span> function to display the unique value counts for the <span style="background-color: #F9EBEA; color:##C0392B">distance</span> column.

# In[12]:

allDistance = abs(dc_listings['accommodates'] - 3)

dc_listings = dc_listings.assign(Distance = allDistance)
print(dc_listings['Distance'].value_counts)


# 6. Randomizing, and sorting
# ==
# 
# It looks like there are quite a few, 461 to be precise, living spaces that can accommodate 3 people just like ours. This means the 5 "nearest neighbors" we select after sorting all will have a distance value of 0. If we sort by the <span style="background-color: #F9EBEA; color:##C0392B">distance</span> column and then just select the first 5 living spaces, we would be **biasing** the result to the ordering of the dataset.
# 
# >```python
# dc_listings[dc_listings["distance"] == 0]["accommodates"]
# 26      3
# 34      3
# 36      3
# 40      3
# 44      3
# 45      3
# 48      3
# 65      3
# 66      3
# 71      3
# 75      3
# 86      3
# ...
# ```
# 
# Let's instead randomize the ordering of the dataset and then sort the Dataframe by the <span style="background-color: #F9EBEA; color:##C0392B">distance</span> column. This way, all of the living spaces with the same number of bedrooms will still be at the top of the Dataframe but will be in random order across the first 461 rows. We've already done the first step of setting the random seed, so we can perform answer checking on our end.
# 
# <br>
# <div class="alert alert-info">
# <b>Exercise Start.</b>
# </div>
# 
# **Description**: 
# 
# 1. Randomize the order of the rows in <span style="background-color: #F9EBEA; color:##C0392B">dc_listings</span>:
#     - Use the <span style="background-color: #F9EBEA; color:##C0392B">np.random.permutation()</span> function to return a NumPy array of shuffled index values.
#     - Use the Dataframe method <span style="background-color: #F9EBEA; color:##C0392B">loc[]</span> to return a new Dataframe containing the shuffled order.
#     - Assign the new Dataframe back to <span style="background-color: #F9EBEA; color:##C0392B">dc_listings</span>.
# 2. After randomization, sort <span style="background-color: #F9EBEA; color:##C0392B">dc_listings</span> by the <span style="background-color: #F9EBEA; color:##C0392B">distance</span> column.
# 3. Display the first 10 values in the <span style="background-color: #F9EBEA; color:##C0392B">price</span> column using the <span style="background-color: #F9EBEA; color:##C0392B">print</span> function.

# In[21]:

# put your code here
np.random.seed(1)

dc_listings = dc_listings.loc[np.random.permutation(dc_listings.index)[::]]
dc_listings = dc_listings.sort_values('Distance')

print(dc_listings['price'][:10])


# 7. Average price
# ==
# 
# Before we can select the 5 most similar living spaces and compute the average price, we need to clean the <span style="background-color: #F9EBEA; color:##C0392B">price</span> column. Right now, the <span style="background-color: #F9EBEA; color:##C0392B">price</span> column contains comma characters (<span style="background-color: #F9EBEA; color:##C0392B">,</span>) and dollar sign characters and is formatted as a text column instead of a numeric one. We need to remove these values and convert the entire column to the float datatype. Then, we can calculate the average price.
# 
# <br>
# <div class="alert alert-info">
# <b>Exercise Start.</b>
# </div>
# 
# **Description**: 
# 
# 1. Remove the commas (<span style="background-color: #F9EBEA; color:##C0392B">,</span>) and dollar sign characters (<span style="background-color: #F9EBEA; color:##C0392B">$</span>) from the <span style="background-color: #F9EBEA; color:##C0392B">price</span> column:
#     - Use the <span style="background-color: #F9EBEA; color:##C0392B">str</span> accessor so we can apply string methods to each value in the column followed by the string method replace to replace all comma characters with the empty character: <span style="background-color: #F9EBEA; color:##C0392B">stripped_commas = dc_listings['price'].str.replace(',', '')</span>
#     - Repeat to remove the dollar sign characters as well.
# 2. Convert the new Series object containing the cleaned values to the <span style="background-color: #F9EBEA; color:##C0392B">float</span> datatype and assign back to the <span style="background-color: #F9EBEA; color:##C0392B">price</span> column in <span style="background-color: #F9EBEA; color:##C0392B">dc_listings</span>.
# 3. Calculate the mean of the first 5 values in the <span style="background-color: #F9EBEA; color:##C0392B">price</span> column and assign to <span style="background-color: #F9EBEA; color:##C0392B">mean_price</span>.
# 4. Use the <span style="background-color: #F9EBEA; color:##C0392B">print</span> function or the variable inspector below to display <span style="background-color: #F9EBEA; color:##C0392B">mean_price</span>

# In[20]:

#dc_listings['price'] = dc_listings['price'].str.replace(',', '').str.replace('$', '').astype(np.float64)
#stripped_simbol = dc_listings['price'].str.replace('$', '').astype(np.float64)

mean_price = dc_listings['price'].head(5).mean()
print(mean_price)


# 8. Function to make predictions
# ==
# 
# Congrats! You've just made your first prediction! Based on the average price of other listings that accommdate 3 people, we should charge **156.6** dollars per night for a guest to stay at our living space. In the next mission, we'll dive into evaluating how good of a prediction this is.
# 
# Let's write a more general function that can suggest the optimal price for other values of the <span style="background-color: #F9EBEA; color:##C0392B">accommodates</span> column. The <span style="background-color: #F9EBEA; color:##C0392B">dc_listings</span> Dataframe has information specific to our living space, e.g. the <span style="background-color: #F9EBEA; color:##C0392B">distance</span> column. To save you time, we've reset the <span style="background-color: #F9EBEA; color:##C0392B">dc_listings</span> Dataframe to a clean slate and only kept the data cleaning and randomization we did since those weren't unique to the prediction we were making for our living space.
# 
# 
# <br>
# <div class="alert alert-info">
# <b>Exercise Start.</b>
# </div>
# 
# **Description**: 
# 
# 1. Write a function named <span style="background-color: #F9EBEA; color:##C0392B">predict_price</span> that can use the **k-nearest neighbors machine learning** technique to calculate the suggested price for any value for <span style="background-color: #F9EBEA; color:##C0392B">accommodates</span>. This function should:
#     - Take in a single parameter, <span style="background-color: #F9EBEA; color:##C0392B">new_listing</span>, that describes the number of bedrooms.
#     - Assign <span style="background-color: #F9EBEA; color:##C0392B">dc_listings</span> to a new Dataframe named <span style="background-color: #F9EBEA; color:##C0392B">temp_df</span> so we aren't constantly modifying the original dataset each time we call the function.
#     - Calculate the distance between each value in the <span style="background-color: #F9EBEA; color:##C0392B">accommodates</span> column and the <span style="background-color: #F9EBEA; color:##C0392B">new_listing</span> value that was passed in. Assign the resulting Series object to the <span style="background-color: #F9EBEA; color:##C0392B">distance</span> column in <span style="background-color: #F9EBEA; color:##C0392B">temp_df</span>.
#     - Sort <span style="background-color: #F9EBEA; color:##C0392B">temp_df</span> by the <span style="background-color: #F9EBEA; color:##C0392B">distance</span> column and select the first 5 values in the <span style="background-color: #F9EBEA; color:##C0392B">price</span> column. Don't randomize the ordering of <span style="background-color: #F9EBEA; color:##C0392B">temp_df</span>.
#     - Calculate the mean of these 5 values and use that as the return value for the entire <span style="background-color: #F9EBEA; color:##C0392B">predict_price</span> function.
# 2. Use the predict_price function to suggest a price for a living space that:
#     - accommodates 1 person, assign the suggested price to **acc_one**.
#     - accommodates 2 people, assign the suggested price to **acc_two**.
#     - accommodates 4 people, assign the suggested price to **acc_four**.

# In[39]:

# Brought along the changes we made to the `dc_listings` Dataframe.
dc_listings = pd.read_csv('dc_airbnb.csv')
stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
dc_listings['price'] = stripped_dollars.astype('float')
dc_listings = dc_listings.loc[np.random.permutation(len(dc_listings))]

def predict_price(new_listing):
    temp_df = dc_listings
    temp_df['distance'] = temp_df['accommodates'].apply(lambda x: np.abs(x - new_listing))
    temp_df = temp_df.sort_values('distance')
    nearest_neighbor_prices = temp_df.iloc[0:5]['price']
    predicted_price = nearest_neighbor_prices.mean()
    return(predicted_price)


# In[40]:

acc_one = predict_price(1)
print(acc_one)


# In[41]:

acc_two = predict_price(2)
print(acc_two)


# In[42]:

acc_four = predict_price(4)
print(acc_four)


# 9. Next steps
# ==
# 
# In this mission, we explored the problem of predicting the optimal price to list an AirBnB rental for based on the price of similar listings on the site. We stepped through the entire machine learning workflow, from selecting a feature to testing the model. To explore the basics of machine learning, we limited ourselves to only using one feature (the univariate case) and a fixed **k** value of **5**.
# 
# In the next mission, we'll learn how to evaluate a model's performance.
