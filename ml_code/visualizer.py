import matplotlib.pyplot as plt
import pandas as pd

path = 'C:\Users\jafri\Documents\GitHub\coral-prediction\processed_data\combined_data_truncated.csv'

raw = pd.read_csv(path)

plt.scatter(raw['longitude'], raw['latitude'], c=raw['coral_present'], s=0.02)
plt.colorbar()
plt.show()
