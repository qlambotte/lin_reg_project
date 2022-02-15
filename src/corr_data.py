from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("./data/data.csv", index_col=0)

corr = df.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

with PdfPages("corr_all.pdf") as pdf:
    plt.figure()
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=0.3,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )
    pdf.savefig()
    plt.close()

# Now we do a heat map for the categorical data alone with the price using
# Cramer-V corelation
df2 = pd.read_csv("./data/data.csv", index_col=0)

df2 = df2[
    [
        "Type of property",
        "Subtype of property",
        "Kitchen equipment",
        "State of the property",
        "Balcony",
        "Furnished",
        "Entry phone",
        "Elevator",
        "Terrace",
        "Security door",
        "Access for disabled",
        "Sewer Connection",
        "Garden",
        "Garage",
        "Price",
    ]
]
for label in [
    "Type of property",
    "Subtype of property",
    "Kitchen equipment",
    "State of the property",
]:
    df2[label] = pd.factorize(df2[label])[0]
print(df2)
with PdfPages("corr_cat.pdf") as pdf:
    fig, axs = plt.subplots(figsize=(12, 9))
    fig.suptitle("Correlation between all categorical columns")
    sns.heatmap(df2.corr(), ax=axs, annot=True)
    pdf.savefig()
    plt.close()
