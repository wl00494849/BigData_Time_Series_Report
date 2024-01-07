import pandas as pd
import os

current_path_data = os.path.join(os.path.dirname(__file__), "clear_data.csv")
df = pd.read_csv(current_path_data)

date = []
val = []
time = []
 
for i in range(len(df)):
    for j in range(2,df.shape[1]):
       date.append(pd.to_datetime(df.iat[i,1]))
       time.append(j-2)
       val.append(df.iat[i,j])


ndata = {
    "date":date,
    "time":time,
    "pm_val":val
}

df = pd.DataFrame(ndata)

print(df)

df.to_csv("pretrain_data.csv")