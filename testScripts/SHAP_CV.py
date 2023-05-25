# The standard SHAP procedure 

import pandas as pd

# Load data
url = 'https://raw.githubusercontent.com/Sketchjar/MachineLearningHD/main/boston_data.csv'
df = pd.read_csv(url); df.drop('Unnamed: 0',axis=1,inplace=True)
X, y = df.drop('Target', axis=1), df.Target

# Libraries for this section 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import shap

# Split data, establish model, fit model, make prediction, score model, print result
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 10)
model = RandomForestRegressor(random_state=10) # Random state for reproducibility (same results every time)
fit = model.fit(X_train, y_train)
yhat = fit.predict(X_test)
result = mean_squared_error(y_test, yhat)
print('RMSE:',round(np.sqrt(result),4))

# Use SHAP to explain predictions
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, features = X.columns)