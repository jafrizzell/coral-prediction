import matplotlib.pyplot as plt
import pandas as pd

path = 'D:/TAMU Work/TAMU 2022 SPRING/OCEN 460/combined_data.csv'

raw = pd.read_csv(path)
print(raw.describe())

raw = raw.dropna(how='any')
print(raw.describe())

raw.to_csv('D:/TAMU Work/TAMU 2022 SPRING/OCEN 460/combined_data_truncated.csv')

# plt.scatter(raw['longitude'], raw['latitude'], s=0.2, c=raw['depth'])