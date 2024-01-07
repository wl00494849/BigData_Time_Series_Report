import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import os

def createTimeSeries(df):
     df["hour"] = df.index.hour
     df["dayofweek"] = df.index.dayofweek
     df["quarter"] = df.index.quarter
     df["month"] = df.index.month
     df["year"] = df.index.year
     df["dayofyear"] = df.index.dayofyear
     return df

current_path_data = os.path.join(os.path.dirname(__file__), "pretrain_data.csv")
df = pd.read_csv(current_path_data)

df["date_time"] = pd.to_datetime(df['date'] + ' ' + df['time'].astype(str) + ':00:00')

color_pal = sns.color_palette()

ndf = pd.DataFrame()
ndf["date_time"] = df['date_time']
ndf["pm_val"] = df['pm_val']
print(ndf)

ndf = ndf.set_index('date_time')
ndf.index = pd.to_datetime(ndf.index)
ndf.plot(style='.',figsize=(15,5),color = color_pal[0],title="PM2.5")
plt.show()

print("======================================")

train_set = ndf.loc[ndf.index < pd.to_datetime('2023-12-06')]
test_set = ndf.loc[ndf.index >= pd.to_datetime('2023-12-06')] 

fig ,ax = plt.subplots(figsize=(15,5))
train_set.plot(ax=ax)
test_set.plot(ax=ax)
ax.axvline('2023-12-25',color='black',ls='--')
ax.legend(["train_set","test_set"])

plt.show()

print("============================")

df = createTimeSeries(ndf)
print(df)
fig ,ax = plt.subplots(figsize=(10,8))
sns.boxplot(x="hour",y="pm_val",data=df)
plt.show()

print("=========================")

train = createTimeSeries(train_set)
test = createTimeSeries(test_set)

FEATURES = ["hour","dayofweek"]
TARGET = ["pm_val"]

X_train = train[FEATURES]
Y_train = train[TARGET]

X_test = test[FEATURES]
Y_test = test[TARGET]

reg = xgb.XGBRegressor(n_estimators=100,
                       learning_rating = 0.0001)
reg.fit(X_train,Y_train,
        eval_set=[(X_train,Y_train),(X_test,Y_test)]
        ,verbose = 100)

print(reg.predict(X_test))
test = test.assign(prediction=reg.predict(X_test))
print(test)
df = df.merge(test[['prediction']],how='left',left_index=True,right_index=True)
print(df)

ax = df[['pm_val']].plot(figsize=(15,5))
df['prediction'].plot(ax=ax)
plt.show()