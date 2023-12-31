import pandas as pd
import os

current_path_data = os.path.join(os.path.dirname(__file__), "clear_data.csv")
data = pd.read_csv(current_path_data)
print(data)
date = []
val = []
time = []

for i in range(len(data)):
    for j in range(2,data.shape[1]):
       date.append(data.iat[i,1])
       time.append(j-2)
       val.append(data.iat[i,j])

ndata = {
    "date":date,
    "time":time,
    "pm_val":val
}

df = pd.DataFrame(ndata)

print(df)

df.to_csv("pretraindata.csv")
