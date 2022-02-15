import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("./data/data.csv", index_col=0)

# Remove lines with too many missing values, where it is unreasonnable to put
# a default value.
df = df.drop(["Build Year", "Floor of appartment", "Number of floors"], axis=1)
print(df["Subtype of property"].value_counts())
# Remove features that are too corelated to others -- see corr_data.py
df = df.drop(["Entry phone"], axis=1)

# REplaces locality by region
def extract_zip(loc):
    return int(loc.split()[0])


df["Locality"] = df["Locality"].map(extract_zip)


def in_flanders(zip):
    if ((1500 <= zip) & (zip <= 3999)) | ((8000 <= zip) & (zip <= 9999)):
        return True
    else:
        return False


def in_brussels(zip):
    if 1000 <= zip & zip <= 1299:
        return True
    else:
        return False


def in_wallonia(zip):
    if in_flanders(zip) | in_brussels(zip):
        return False
    else:
        return True


def zip_to_region(zip_):
    if in_flanders(zip_):
        return "Flanders"
    elif in_brussels(zip_):
        return "Brussels"
    else:
        return "Wallonia"


df["Locality"] = df["Locality"].map(zip_to_region)

df = df.dropna()

# Remove outliers according to the analysis in outliers.py
print("Shape before:", df.shape)
categorical = [
    "Locality",
    "Subtype of property",
    "Type of property",
    "Kitchen equipment",
    "State of the property",
    "Orientation of the front facade",
    "Postal code",
    "Furnished",
    "Balcony",
    "Cellar",
    "Elevator",
    "Terrace",
    "Garage",
    "Garden",
    "Security door",
    "Access for disabled",
    "Sewer Connection",
    "Surface garden",
]
non_cat = [label for label in df.columns if label not in categorical]
z = np.abs(stats.zscore(df[non_cat], nan_policy="omit"))
df = df[(z < 2).all(axis=1)]
print("Shape of df;", df.shape)
# OneHotEncoding of non-ordinal categ cols
onehot_labels = [
    "Kitchen equipment",
    "State of the property",
    "Type of property",
    "Subtype of property",
]
for label in onehot_labels:
    df_onehot = pd.get_dummies(df[label], prefix=label)
    df = df.drop([label], axis=1)
    df = df.join(df_onehot)

df.to_csv("./data/data_cleaned.csv")
