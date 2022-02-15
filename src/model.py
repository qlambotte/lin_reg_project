import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn.feature_selection as featsel
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.ticker import FormatStrFormatter

df = pd.read_csv('./data/data_cleaned.csv', index_col=0)
df2 = pd.read_csv('./data/data.csv')
print(df2['Type of property'].value_counts())

def model(df, region='Belgium', type='All', threshold=500, test_size=0.33):
    if region != 'Belgium':
        df = df[df["Locality"]==region]
    if type != 'All':
        df = df[df[f'Type of property_{type}']==1]
    df = df.drop(["Locality"], axis=1)
    y = df['Price'].to_numpy()
    if len(y) > threshold:
        X = df.drop(['Price'], axis=1).to_numpy()
        linear = linear_model.LinearRegression()
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 123,test_size=test_size)
        linear.fit(X_train,y_train)
        y_pred = linear.predict(X_test)
        return (
            [linear.score(X_train,y_train), r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred), len(y), len(y_train), len(y_test)],
            (y_pred,y_test)
        )
    else:
        return ['Not enough data', 'Not enough data', len(y), 'Not enough data', 'Not enough data'], ([],[])

regions = ['Belgium', 'Wallonia', 'Brussels', 'Flanders']

types = [label.split('_')[-1] for label in df.columns if 'Type of property' in label]
print(types)
types.append('All')

data_ ={}
with PdfPages(f'model.pdf') as pdf:
    for type in types:
        for region in regions:
            data, data_to_plot = model(df, region, type)
            if not('Not enough data' in data):
                data_[(type, region)] = data
            if len(data_to_plot[1])>0:
                plt.figure()
                fig, ax = plt.subplots()
                ax.set_aspect('equal', 'box')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                plt.xticks(rotation = 30)
                x, y = data_to_plot
                sns.scatterplot(x=x, y=y)
                low_x, high_x = ax.get_xlim()
                low_y, high_y = ax.get_ylim()
                low = max(low_x, low_y)
                high = min(high_x, high_y)
                ax.plot([low, high], [low, high], ls="--", c=".3", alpha=0.5)
                plt.title(f'Predictions against reality for {type} in {region}')
                pdf.savefig( transparent=True)
                plt.close("all")
data_ = pd.DataFrame.from_dict(data_)
data_.index = ['R² for training data', 'R² for testing data', 'MSE for testing data', 'Size of data', 'Size of training data', 'Size of test data']
data_.to_latex(buf='data.tex', float_format="%.2f")
data_.to_csv(f'results.csv')
