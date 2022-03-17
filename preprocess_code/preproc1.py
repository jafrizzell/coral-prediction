import pandas as pd

path = 'C:/Users/jafri/Documents/GitHub/coral-prediction/processed_data/deep_sea_corals_rounded.csv'

raw = pd.read_csv(path)


# def round_depth(x, base):
#     return int(base * round(float(x)/base))
#
#
# raw['depth'] = raw['depth'].apply(lambda x: round_depth(x, base=5))

raw = raw[raw.depth >= 0]
print(len(raw))
raw.to_csv('C:/Users/jafri/Documents/GitHub/coral-prediction/processed_data/deep_sea_corals_rounded_depthcorr.csv')