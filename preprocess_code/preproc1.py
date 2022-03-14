import pandas as pd

path = "C:/Users/jafrizzell.22/Desktop/460Design/deep_sea_corals.csv"

raw = pd.read_csv(path)


def round_depth(x, base):
    return int(base * round(float(x)/base))


raw['depth'] = raw['depth'].apply(lambda x: round_depth(x, base=5))


raw.to_csv('C:/Users/jafrizzell.22/Desktop/460Design/deep_sea_corals_rounded.csv')