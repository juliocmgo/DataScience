
# coding: utf-8

# 1. Testing quality of predictions
# ==
# 
# We now have a function that can predict the price for any living space we want to list as long as we know the number of people it can accommodate. The function we wrote represents a **machine learning model**, which means that it outputs a prediction based on the input to the model.
# 
# A simple way to test the quality of your model is to:
# 
# - split the dataset into 2 partitions:
#     - the training set: contains the majority of the rows (75%)
#     - the test set: contains the remaining minority of the rows (25%)
# - use the rows in the training set to predict the <span style="background-color: #F9EBEA; color:##C0392B">price</span> value for the rows in the test set
#     - add new column named <span style="background-color: #F9EBEA; color:##C0392B">predicted_price</span> to the test set
# - compare the <span style="background-color: #F9EBEA; color:##C0392B">predicted_price</span> values with the actual  <span style="background-color: #F9EBEA; color:##C0392B">price</span> values in the test set to see how accurate the predicted values were.
# 
# This validation process, where we use the training set to make predictions and the test set to predict values for, is known as **train/test validation**. Whenever you're performing machine learning, you want to perform validation of some kind to ensure that your machine learning model can make good predictions on new data. While train/test validation isn't perfect, we'll use it to understand the validation process, to select an error metric, and then we'll dive into a more robust validation process later in this course.
# 
# Let's modify the <span style="background-color: #F9EBEA; color:##C0392B">predicted_price</span> function to use only the rows in the training set, instead of the full dataset, to find the nearest neighbors, average the <span style="background-color: #F9EBEA; color:##C0392B">price</span> values for those rows, and return the predicted price value. Then, we'll use this function to predict the price for just the rows in the test set. Once we have the predicted price values, we can compare with the true price values and start to understand the model's effectiveness in the next screen.
# 
# To start, we've gone ahead and assigned the first 75% of the rows in <span style="background-color: #F9EBEA; color:##C0392B">dc_listings</span> to <span style="background-color: #F9EBEA; color:##C0392B">train_df</span> and the last 25% of the rows to <span style="background-color: #F9EBEA; color:##C0392B">test_df</span>. Here's a diagram explaining the split:
# 
# <img width="600" alt="creating a repo" src="https://drive.google.com/uc?export=view&id=11IctHIyFi18HxRsg9LpsOf4tVKqfqvRz">
# 
# <br>
# <div class="alert alert-info">
# <b>Exercise Start.</b>
# </div>
# 
# **Description**: 
# 
# 1. Within the <span style="background-color: #F9EBEA; color:##C0392B">predict_price</span> function, change the Dataframe that <span style="background-color: #F9EBEA; color:##C0392B">temp_df</span> is assigned to. Change it from <span style="background-color: #F9EBEA; color:##C0392B">dc_listings</span> to <span style="background-color: #F9EBEA; color:##C0392B">train_df</span>, so only the training set is used.
# 2. Use the Series method <span style="background-color: #F9EBEA; color:##C0392B">apply</span> to pass all of the values in the <span style="background-color: #F9EBEA; color:##C0392B">accommodates</span> column from <span style="background-color: #F9EBEA; color:##C0392B">test_df</span> through the <span style="background-color: #F9EBEA; color:##C0392B">predict_price</span> function.
# 3. Assign the resulting Series object to the <span style="background-color: #F9EBEA; color:##C0392B">predict_price</span> column in <span style="background-color: #F9EBEA; color:##C0392B">test_df</span>.

# In[2]:

# importing packages
import pandas as pd
import numpy as np

# import dataset
dc_listings = pd.read_csv("dc_airbnb.csv")

# cleaning & preparing
stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
dc_listings['price'] = stripped_dollars.astype('float')

# separte data into train and test (75%/25%)
train_df = dc_listings.iloc[0:2792]
test_df = dc_listings.iloc[2792:]

def predict_price(new_listing):
    temp_df = train_df
    temp_df['distance'] = temp_df['accommodates'].apply(lambda x: np.abs(x - new_listing))
    temp_df = temp_df.sort_values('distance')
    nearest_neighbor_prices = temp_df.iloc[0:5]['price']
    predicted_price = nearest_neighbor_prices.mean()
    return(predicted_price)


# In[5]:




# 2. Error Metrics
# ==
# 
# We now need a metric that quantifies how good the predictions were on the test set. This class of metrics is called an **error metric**. As the name suggests, an error metric quantifies how inaccurate our predictions were from the actual values. In our case, the error metric tells us how off our predicted price values were from the actual price values for the living spaces in the test dataset.
# 
# We could start by calculating the difference between each predicted and actual value and then averaging these differences. This is referred to as **mean error** but isn't an effective error metric for most cases. Mean error treats a positive difference differently than a negative difference, but we're really interested in how far off the prediction is in either the positive or negative direction. If the true price was 200 dollars and the model predicted 210 or 190 it's off by 10 dollars either way.
# 
# We can instead use the **mean absolute error**, where we compute the absolute value of each error before we average all the errors.
# 
# $\displaystyle MAE = \frac{\left | actual_1 - predicted_1 \right | + \left | actual_2 - predicted_2 \right | + \
# \ldots + \left | actual_n - predicted_n \right | }{n}$
# 
# <br>
# <div class="alert alert-info">
# <b>Exercise Start.</b>
# </div>
# 
# **Description**: 
# 
# 1. Use <span style="background-color: #F9EBEA; color:##C0392B">numpy.absolute()</span> to calculate the mean absolute error between <span style="background-color: #F9EBEA; color:##C0392B">predicted_price</span> and <span style="background-color: #F9EBEA; color:##C0392B">price</span>.
# 2. Assign the MAE to <span style="background-color: #F9EBEA; color:##C0392B">mae</span>.

# In[2]:

MAE = np.Abs(dc_listings['price'] - predict_price(2))


# 3. Mean Squared Error
# ==
# 
# For many prediction tasks, we want to penalize predicted values that are further away from the actual value much more than those that are closer to the actual value.
# 
# We can instead take the mean of the squared error values, which is called the **mean squared error** or MSE for short. The MSE makes the gap between the predicted and actual values more clear. A prediction that's off by 100 dollars will have an error (of 10,000) that's 100 times more than a prediction that's off by only 10 dollars (which will have an error of 100).
# 
# Here's the formula for MSE:
# 
# $\displaystyle MSE = \frac{(actual_1 - predicted_1)^2 + (actual_2 - predicted_2)^2 + \
# \ldots + (actual_n - predicted_n)^2 }{n}$
# 
# where **n** represents the number of rows in the test set. Let's calculate the MSE value for the predictions we made on the test set.
# 
# <br>
# <div class="alert alert-info">
# <b>Exercise Start.</b>
# </div>
# 
# **Description**: 
# 
# 1. Calculate the MSE value between the <span style="background-color: #F9EBEA; color:##C0392B">predicted_price</span> and <span style="background-color: #F9EBEA; color:##C0392B">price</span> columns and assign to <span style="background-color: #F9EBEA; color:##C0392B">mse</span>.

# In[ ]:

MSE = (predict_price(2) - dc_listings['price'])^2


# 4. Training another model
# ==
# 
# The model we trained achieved a mean squared error of around **18646.5**. Is this a high or a low mean squared error value? What does this tell us about the quality of the predictions and the model? By itself, the mean squared error value for a single model isn't all that useful.
# 
# The units of mean squared error in our case is dollars squared (not dollars), which makes it hard to reason about intuitively as well. We can, however, train another model and then compare the mean squared error values to see which model performs better on a relative basis. Recall that a low error metric means that the gap between the predicted list price and actual list price values is low while a high error metric means the gap is high.
# 
# Let's train another model, this time using the <span style="background-color: #F9EBEA; color:##C0392B">bathrooms</span> column, and compare MSE values.
# 
# <br>
# <div class="alert alert-info">
# <b>Exercise Start.</b>
# </div>
# 
# **Description**: 
# 
# 1. Modify the <span style="background-color: #F9EBEA; color:##C0392B">predict_price</span> function below to use the <span style="background-color: #F9EBEA; color:##C0392B">bathrooms</span> column instead of the <span style="background-color: #F9EBEA; color:##C0392B">accommodates</span> column to make predictions.
# 2. Apply the function to <span style="background-color: #F9EBEA; color:##C0392B">test_df</span> and assign the resulting Series object containing the predicted price values to the <span style="background-color: #F9EBEA; color:##C0392B">predicted_price</span> column in <span style="background-color: #F9EBEA; color:##C0392B">test_df</span>.
# 3. Calculate the squared error between the price and <span style="background-color: #F9EBEA; color:##C0392B">predicted_price</span> columns in <span style="background-color: #F9EBEA; color:##C0392B">test_df</span> and assign the resulting Series object to the <span style="background-color: #F9EBEA; color:##C0392B">squared_error</span> column in <span style="background-color: #F9EBEA; color:##C0392B">test_df</span>.
# 4. Calculate the mean of the <span style="background-color: #F9EBEA; color:##C0392B">squared_error</span> column in <span style="background-color: #F9EBEA; color:##C0392B">test_df</span> and assign to <span style="background-color: #F9EBEA; color:##C0392B">mse</span>.
# 5. Use the <span style="background-color: #F9EBEA; color:##C0392B">print</span> function or the variables inspector to display the MSE value.
# 

# In[8]:

train_df = dc_listings.iloc[0:2792]
test_df = dc_listings.iloc[2792:]

def predict_price(new_listing):
    temp_df = train_df
    temp_df['distance'] = temp_df['accommodates'].apply(lambda x: np.abs(x - new_listing))
    temp_df = temp_df.sort_values('distance')
    nearest_neighbors_prices = temp_df.iloc[0:5]['price']
    predicted_price = nearest_neighbors_prices.mean()
    return(predicted_price)


# 5. Root Mean Squared Error
# ==
# 
# While comparing MSE values helps us identify which model performs better on a relative basis, it doesn't help us understand if the performance is good enough in general. This is because the units of the MSE metric are squared (in this case, dollars squared). An MSE value of 16377.5 dollars squared doesn't give us an intuitive sense of how far off the model's predictions are systematically off from the true price value in dollars.
# 
# **Root mean squared error** is an error metric whose units are the base unit (in our case, dollars). RMSE for short, this error metric is calculated by taking the square root of the MSE value:
# 
# $\displaystyle RMSE=\sqrt{MSE}$
# 
# Since the RMSE value uses the same units as the target column, we can understand how far off in real dollars we can expect the model to perform. For example, if a model achieves an RMSE value of greater than 100, we can expect the predicted price value to be off by 100 dollars on average.
# 
# Let's calculate the RMSE value of the model we trained using the <span style="background-color: #F9EBEA; color:##C0392B">bathrooms</span> column.
# 
# 
# <br>
# <div class="alert alert-info">
# <b>Exercise Start.</b>
# </div>
# 
# **Description**: 
# 
# 1. Calculate the RMSE value of the model we trained using the <span style="background-color: #F9EBEA; color:##C0392B">bathrooms</span> column and assign it to **rmse**.
# 

# In[ ]:




# 6. Comparing MAE and RMSE
# ==
# 
# The model achieved an RMSE value of approximately **135.6**, which implies that we should expect for the model to be off by **135.6** dollars on average for the predicted price values. Given that most of the living spaces are listed at just a few hundred dollars, we need to reduce this error as much as possible to improve the model's usefulness.
# 
# We discussed a few different error metrics we can use to understand a model's performance. As we mentioned earlier, these individual error metrics are helpeful for comparing models. To better understand a specific model, we can compare multiple error metrics for the same model. This requires a better understanding of the mathematical properties of the error metrics.
# 
# If you look at the equation for MAE:
# 
# $\displaystyle MAE = \frac{\left | actual_1 - predicted_1 \right | + \left | actual_2 - predicted_2 \right | + \
# \ldots + \left | actual_n - predicted_n \right | }{n}$
# 
# you'll notice that a prediction that the individual errors (or differences between predicted and actual values) grow linearly. A prediction that's off by 10 dollars has a 10 times higher error than a prediction that's off by 1 dollar. If you look at the equation for RMSE, however:
# 
# $\displaystyle RMSE = \sqrt{\frac{(actual_1 - predicted_1)^2 + (actual_2 - predicted_2)^2 + \
# \ldots + (actual_n - predicted_n)^2 }{n}}$
# 
# you'll notice that each error is squared before the square root of the sum of all the errors is taken. This means that the individual errors grows quadratically and has a different effect on the final RMSE value.
# 
# Let's look at an example using different data entirely. We've created 2 Series objects containing 2 sets of errors and assigned to <span style="background-color: #F9EBEA; color:##C0392B">errors_one</span> and <span style="background-color: #F9EBEA; color:##C0392B">errors_two</span>.
# 
# <br>
# <div class="alert alert-info">
# <b>Exercise Start.</b>
# </div>
# 
# **Description**: 
# 
# 1. Calculate the MAE for <span style="background-color: #F9EBEA; color:##C0392B">errors_one</span> and assign to <span style="background-color: #F9EBEA; color:##C0392B">mae_one</span>.
# 2. Calculate the RMSE for <span style="background-color: #F9EBEA; color:##C0392B">errors_one</span> and assign to <span style="background-color: #F9EBEA; color:##C0392B">rmse_one</span>.
# 3. Calculate the MAE for <span style="background-color: #F9EBEA; color:##C0392B">errors_two</span> and assign to <span style="background-color: #F9EBEA; color:##C0392B">mae_two</span>.
# 4. Calculate the RMSE for <span style="background-color: #F9EBEA; color:##C0392B">errors_two</span> and assign to <span style="background-color: #F9EBEA; color:##C0392B">rmse_two</span>.

# In[ ]:




# 7. Next steps
# ==
# 
# While the MAE (7.5) to RMSE (7.9056941504209481) ratio was about 1:1 for the first list of errors, the MAE (62.5) to RMSE (235.82302686548658) ratio was closer to 1:4 for the second list of errors. The only difference between the 2 sets of errors is the extreme 1000 value in errors_two instead of 10. When we're working with larger data sets, we can't inspect each value to understand if there's one or some outliers or if all of the errors are systematically higher. Looking at the ratio of MAE to RMSE can help us understand if there are large but infrequent errors. You can read more about comparing MAE and RMSE in [this wonderful post](https://medium.com/human-in-a-machine-world/mae-and-rmse-which-metric-is-better-e60ac3bde13d#.lyc8od1ix).
# 
# In this mission, we learned how to test our machine learning models using basic cross validation and different metrics. In the next 2 missions, we'll explore how adding more features to the machine learning model and selecting a more optimal k value can help improve the model's performance.
