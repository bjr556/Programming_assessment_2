import pandas as pd
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
from sqlalchemy import label
import geopandas as gpd
import folium
import Moran
from esda.moran import Moran
from geopy.geocoders import Nominatim
from matplotlib.ticker import NullFormatter
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt



"""
MAIN CODE
"""

# reading dataset file into a dataframe called 'data'
data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Sub_Division_IMD_2017.csv')
display(data)    #  Displaying the DataFrame that we just read

# Data Exploration:
# After inspecting the data, I see that the data is of 117 years of only some selected states in India, with their rainfall data for each month.
# The data ranges from 1901-2017. With a data of 100+ years, we can analyze various aspects.

# Analytical Goals:
# 1. Long-term trend
# 2. Seasonal Pattern
# 3. Months with max-min rainfall and variability
# 4. Extreme events in our centurial data
# 5. State-wise analysis
# 6. Trend-line (direction and pattern of my data)
# 7. Cyclic patterns
# 8. Comparing two different months of the same and different years
# 9. Time-series of the state with max-min rainfall and understanding the pattern for the same
# 10. Spatial Analysis
# 11. Forecast the future rainfall patterns
# 12. Grouping the data into groups of 15-20 years and analyzing the variability

"""

"""
# omitting last 4 columns (JF,MAM,JJA, OND)
data1 = data.iloc[:, :-4]

Uttarakhand = (data1 == 'Uttarakhand').any(axis=1)
Uttarakhand = data1[Uttarakhand]

mean_value = np.mean(Uttarakhand['ANNUAL'])
print(mean_value)
median_value = np.median(Uttarakhand['ANNUAL'])
print(median_value)

std_dev = np.std(Uttarakhand['ANNUAL'])
print(std_dev)
variance = np.var(Uttarakhand['ANNUAL'])
print(variance)

skewness = Uttarakhand['ANNUAL'].skew()
print(skewness)
kurtosis = Uttarakhand['ANNUAL'].kurt()
print(kurtosis)

minimum = np.min(Uttarakhand['ANNUAL'])
print(minimum)
maximum = np.max(Uttarakhand['ANNUAL'])
print(maximum)

q1 = np.percentile(Uttarakhand['ANNUAL'] , 25)
print(q1)
q3 = np.percentile(Uttarakhand['ANNUAL'] , 75)
print(q3)

iqr = q3 - q1
print(iqr)

# Create a box plot for all numeric columns
data1.boxplot(rot=45, figsize=(12, 8))
plt.title('Box Plot for Each Column')
plt.show()

# threshold for Z-scores
threshold = 3

# Creating an empty DataFrame to store outliers
outliers_df = pd.DataFrame()

# Loop through each numeric column
for column_name in data1.select_dtypes(include=np.number).columns:
    # Calculate Z-scores
    z_scores = np.abs((data1[column_name] - data1[column_name].mean()) / df[column_name].std())

    # Identify outliers
    column_outliers = data1[z_scores > threshold]

    # Append outliers to the outliers_df
    outliers_data1 = pd.concat([outliers_df, column_outliers])

# Remove duplicates from outliers_df
outliers_data1 = outliers_df.drop_duplicates()

# Remove outliers from the original DataFrame
data1_cleaned = data1.drop(outliers_df.index)
# data1_cleaned contains the data without outliers in columns

# Display information about removed outliers
print(f'Number of outliers removed: {len(outliers_df)}')
print('Outliers:')
print(outliers_data1)

# Create a box plot for all columns
data1_cleaned.boxplot(rot=45, figsize=(12, 8))
plt.title('Box Plot for Each Column')
plt.show()

Uttarakhand_cleaned = (data1_cleaned == 'Uttarakhand').any(axis=1)
Uttarakhand_cleaned = data1_cleaned[Uttarakhand]

mean_value_cleaned = np.mean(Uttarakhand_cleaned['ANNUAL'])
print(mean_value)
median_value_cleaned = np.median(Uttarakhand_cleaned['ANNUAL'])
print(median_value)

std_dev_cleaned = np.std(Uttarakhand_cleaned['ANNUAL'])
print(std_dev)
variance_cleaned = np.var(Uttarakhand_cleaned['ANNUAL'])
print(variance)

skewness_cleaned = Uttarakhand_cleaned['ANNUAL'].skew()
print(skewness)
kurtosis_cleaned = Uttarakhand_cleaned['ANNUAL'].kurt()
print(kurtosis)

minimum_cleaned = np.min(Uttarakhand_cleaned['ANNUAL'])
print(minimum)
maximum_cleaned = np.max(Uttarakhand_cleaned['ANNUAL'])
print(maximum)

q1_cleaned = np.percentile(Uttarakhand_cleaned['ANNUAL'] , 25)
print(q1)
q3_cleaned = np.percentile(Uttarakhand_cleaned['ANNUAL'] , 75)
print(q3)

iqr_cleaned = q3 - q1
print(iqr_cleaned)

data1_cleaned 

# Seasonal patterns
"""
To find seasonal patterns in a rainfall dataset spanning 100 years, I am using time series analysis
seasonal decomposition, which separates the data into its underlying components:
trend, seasonal, and residual.
Trend will help us look for long-term patterns or trends in the data, if it is increasing, decreasing, or relatively stable over time?
Seasonal helps us Identify repeating patterns that occur at regular intervals, seasonality can be noted with the peaks and trough
Residual Checks for any remaining patterns or irregularities in the data.
"""
decomposition = seasonal_decompose(Uttarakhand_cleaned['ANNUAL'], model = 'multiplicative', period = 12)
plt.figure(figsize = (20,8))
plt.subplot(4,1,1)
plt.plot(Uttarakhand_cleaned.index, decomposition.trend, label='Trend', color = 'blue')
plt.legend()
plt.subplot(4,1,2)
plt.plot(Uttarakhand_cleaned.index, decomposition.seasonal, label='seasonal', color = 'green')
plt.legend()
plt.subplot(4,1,3)
plt.plot(Uttarakhand_cleaned.index, decomposition.resid, label='residual', color = 'red')
plt.legend()
plt.subplot(4,1,4)
plt.plot(Uttarakhand_cleaned.index, Uttarakhand['ANNUAL'], label='original', color = 'black')
plt.legend()
plt.show()