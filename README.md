Programming assessment 2
The data is of 117 years of some selected states in India, with their rainfall data for each month.
he data ranges from 1901-2017. With a data of 100+ years, we can analyze various aspects.

Analytical Goals:
1. Long-term trend
2. Seasonal Pattern
3. Months with max-min rainfall and variability
4. Extreme events in our centurial data
5. State-wise analysis
6. Trend-line (direction and pattern of my data)
7. Comparing two different months of the same and different years
8. Time-series of the state with max-min rainfall and understanding the pattern for the same
9. Spatial Analysis
10. Forecast the future rainfall patterns
11. Grouping the data into groups of 15-20 years and analyzing the variability

I've tried to achieve most of the goals

**pip commands**
pip install pandas 
pip install scipy
pip install matplotlib
pip install ipython
pip install pyflux
pip install folium
pip install statsmodels

**Libraries used:**
from scipy.signal import periodogram     
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import numpy as np
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
import math
from statsmodels.tsa.seasonal import seasonal_decompose

periodogram  - isualize the frequency content, used to identify periodic components or patterns in time-series dataset
ARIMA - The ARIMA (AutoRegressive Integrated Moving Average) model is a statistical technique used for time series forecasting.


1. read the data
2. computed the periodgram
3. resolving the warning
4. Statistical analysis for a state
5. box-plotting the whole dataset for outlier rejection
6. outlier rejection
7. box-plotting after removing outliers
8. Seasonal analysis - with the use of grouped months in the dataset, we can analyze the rainy months in india
9. Seasonal patterns - using from seasonal_decompose - is used for time series analysis to decompose a time series into its constituent components: trend, seasonality, and residuals. This decomposition helps in understanding the underlying patterns and variations present in the time series data.
   Running averages and moving averages
11.Trend line -  Trend line signifies the general direction of the pattern of our data.
12.  converting the dataframe into long format and mapping it into 2 columns - month and rainfall in mm
13.  temporal analysis
14.  seasonal pattern
15.  monthly distribution
16.  grouping the data in 15 years - max rainfall and min rainfall
17.  spatial analysis
18.  merging the main-data with a new dataset which has lat and long of the indian states
19.  spatial distribution of annual rainfall - using geopandas
20.  geolocator - location specific - testing
21.  geolocator - location specific - testing 2
22.  folium mapping, using geocode and geolocator
23.  States with maximum rainfall and minimum rainfall
24.  Forecasting using arima model

Columns are as follows:
CITY	YEAR	**JAN	FEB	MAR	APR	MAY	JUN	JUL	AUG	SEP	OCT	NOV	DEC**	ANNUAL	JF	MAM	JJAS	OND
   
   
    
