#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ### (i) Perform data pre-processing steps on the dataset. Handle missing values (if any), explore the correlation between variables, and identify any potential outliers.

# In[33]:


data = pd.read_csv('ENB2012_data.csv')
df = pd.DataFrame(data)

print(df.info())
print(df.describe())
print(df.head())

# Check for missing values
print("Missing Values:\n", df.isnull().sum())


# In[34]:


df1 = df.dropna()
print(df1)


# In[35]:


# Visualize correlations
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# ### (ii) Split the dataset into an 80:20 ratio for training and testing using the sklearn library.

# In[36]:


from sklearn.model_selection import train_test_split
#Ignoring the features having weak/no correlation.
features_col = ['X1','X2','X4','X5','X7']
target_col = ['Y1']
x = df2[features_col]
y = df2[target_col]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=64, test_size=0.2)


# ### Multivariate linear regression

# In[37]:


x_b = np.c_[np.ones((x.shape[0], 1)), x] 
# Step 2: Compute the model parameters (beta) using the Normal Equation
# beta = (X^T * X)^-1 * X^T * y
beta = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)

# Step 3: Make predictions on the test set

x_test_b = np.c_[np.ones((x_test.shape[0], 1)), x_test]  # Add bias term to the test set
y_pred = x_test_b.dot(beta)

# Evaluate
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Print the performance metrics
print(f"R²: {r2}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")


# ### Linear Regression using sklearn Library

# In[38]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# In[39]:


lr_model = LinearRegression()
#training data
lr_model.fit(x_train, y_train)
# Make predictions on the test set
y_pred_lr = lr_model.predict(x_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred_lr)
rmse = np.sqrt(mse)
print(f"R² Score: {r2}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")


# ### Ridge and Lasso regression using  sklearn library

# In[40]:


from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error


# In[41]:


# Initialize Ridge and Lasso regression models
ridge_model = Ridge(alpha=1.0)  
lasso_model = Lasso(alpha=0.1)  # alpha is the regularization strength


ridge_model.fit(x_train, y_train)
lasso_model.fit(x_train, y_train)

# Predictions 
y_pred_ridge = ridge_model.predict(x_test)
y_pred_lasso = lasso_model.predict(x_test)


# In[42]:


# Evaluate the models
def evaluate_model(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
   
    return r2, mse,rmse

# Evaluate Ridge and Lasso models
r2_ridge, mse_ridge,rmse_ridge = evaluate_model(y_test, y_pred_ridge)
r2_lasso, mse_lasso,rmse_lasso = evaluate_model(y_test, y_pred_lasso)

# Print results
print("Ridge Regression Results:")
print(f"R²: {r2_ridge}")
print(f"MSE: {mse_ridge}")
print(f"RMSE: {rmse_ridge}")



print("\nLasso Regression Results:")
print(f"R²: {r2_lasso}")
print(f"MSE: {mse_lasso}")
print(f"RMSE: {rmse_lasso}")




# ### Compare the performance of the manual model and the sklearn model and plot the predicted vs actual values for both models.

# In[49]:


plt.figure(figsize=(12, 6))

# Manual Model
plt.subplot(1,2,1)
plt.scatter(y_test,y_pred , alpha = 0.6, marker = 'o')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=0.8)
plt.title('Manual Model:Predicted vs Actual')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)



# sklearn Model
plt.subplot(1,2,2)
plt.scatter(y_test, y_pred_lr, alpha = 0.6, marker = 'o', c='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=0.8)
plt.title('sklearn Model:Predicted vs Actual')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.grid(True)



# In[ ]:





# In[ ]:




