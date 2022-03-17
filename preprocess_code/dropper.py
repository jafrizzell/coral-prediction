import pandas as pd

path = 'D:/TAMU Work/TAMU 2022 SPRING/OCEN 460/depthtempsaloxy_short.csv'

raw = pd.read_csv(path)
print(len(raw))
raw = raw.dropna(how='any')
print(len(raw))
raw.to_csv('D:/TAMU Work/TAMU 2022 SPRING/OCEN 460/depthtempsaloxy_short2.csv', index=False)