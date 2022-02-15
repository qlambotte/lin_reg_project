from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./data/data_.csv', index_col=0)
df = df.drop('Locality', axis=1)

with PdfPages('analysis.pdf') as pdf:
    for label in df.columns:
          plt.figure(figsize=(20,5))

          plt.subplot(1,3,1)
          sns.boxplot(x=df[label],color='#005030')
          plt.title(f'Box Plot of {label}')

          plt.subplot(1,3,2)
          sns.histplot(data=df[label], color='#500050', kde=True)
          plt.title(f'Distribution Plot of {label}')

          plt.subplot(1,3,3)
          sns.scatterplot(x=df[label],y=df.Price)
          plt.title(f'Scatter Plot of {label} against Price')
          pdf.savefig()
          plt.close()
    for label in df.columns:
          plt.figure(figsize=(20,5))

          sns.scatterplot(x=df[label],y=df['Livable surface'])
          plt.title(f'Scatter Plot of {label} against Price')
          pdf.savefig()
          plt.close()
