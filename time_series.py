import pandas as pd
import numpy as np
import sklearn.model_selection._split as sk
import matplotlib.pyplot as plt
import seaborn as sns
import os

current_path_data = os.path.join(os.path.dirname(__file__), "clear_data.csv")
data = pd.read_csv(current_path_data)