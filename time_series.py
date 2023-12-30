import pandas as pd
import numpy as np
import sklearn.model_selection._split as sk
import matplotlib.pyplot as plt
import os

current_path_data = os.path.join(os.path.dirname(__file__), "train.csv")
data = pd.read_csv(current_path_data)

print(data[["日期","00","01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23"]])
dataC = data[["日期","00","01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23"]]

prev = 0
cur = 0

for i in range(len(dataC)):
    for i in range(len(dataC[i])):
            


