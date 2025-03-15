import pandas as pd
df1 = pd.read_csv('SET 1_DailyDelhiClimateTrain.csv')#reading dataframes as csv files
df2 = pd.read_csv('SET 1_DailyDelhiClimateTest.csv')
df1['date'] = pd.to_datetime(df1['date'])#dates being converted from float to date-time format
df2['date'] = pd.to_datetime(df2['date'])  
df1.describe()["meanpressure"]#describing column properties of dataframes
df1.describe()["meantemp"]
df1.describe()["wind_speed"]
df1.describe()["humidity"]
df2.describe()["meanpressure"]
df2.describe()["meantemp"]
df2.describe()["wind_speed"]
df2.describe()["humidity"]
df1.corr()["humidity"]#displaying correlation between parameters
df1.corr()["meantemp"]
df1.corr()["wind_speed"]
df1.corr()["meanpressure"]
df2.corr()["humidity"]
df2.corr()["meantemp"]
df2.corr()["wind_speed"]
df2.corr()["meanpressure"]
train = df1.copy()
test = df2.copy() 
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])
train.shape #diplays (1462, 5)
test.shape #diplays (114, 5)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
train['date_ordinal'] = train['date'].apply(lambda date: date.toordinal()) #dates being converted to ordinal form otherwise linear regression would not be observed
test['date_ordinal'] = test['date'].apply(lambda date: date.toordinal())
predictors = ['date_ordinal', 'meantemp', 'humidity', 'wind_speed']  
target = 'meanpressure' #value of mean pressure would be predicted so it is selected as target
reg.fit(train[predictors], train[target])
LinearRegression()
predictions = reg.predict(test[predictors])
predictions#displaying predicted mean pressure values
predictions.round()#displaying rounded values of the predicted mean pressures
from sklearn.metrics import mean_absolute_error
error = mean_absolute_error(test[target], predictions)
error #displays 19.636164583042596
test['mean pressure predictions'] = predictions
test
error_abs = (test["mean pressure predictions"] - test["meanpressure"]).abs()#displaying absolute errors between predicted mean pressure values and actual test values
error_abs
error_by_date = error_abs.groupby(test['date']).mean()
error_by_date#displaying mean errors by date
mp_predictions = test['meanpressure'].groupby(test['date']).mean()
error_ratio = error_by_date/mp_predictions
error_ratio
error_ratio.plot.hist()#displaying histogram for errors





