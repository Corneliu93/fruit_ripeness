import pandas as pd

df = pd.read_csv("results/data_split.csv")
print(df.groupby(["class_name", "split"]).size())