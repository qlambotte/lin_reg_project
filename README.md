# Linear regression project
## Description
This project aims at modeling the price of the real estate market in Belgium, based on data scraped on ImmoVlan. The model chosen here is an ordinary linear regression.
## Installation

The `scr` directory contains four files, each of which having a specific purpose (see below). Each file can be executed in the standard way:
```python
python3 src/<NAME_OF_FILE>.py
```
Here is a description of each file:
1. the file `corr_data.py` produces two heatmaps representing the corelations among the variables (one with the categorical data and another with the non-categorical data);
2. the file `outliers.py` produces a file containing boxplots, distribution, and scatter plot against the price of each feature (and the price);
3. the file `cleaning.py` formats and cleans the data according to the analysis made using the results of `corr_data.py` and `outliers.py`;
4. the file `model.py` implements the linear model for the data. It produces three files. One contains the scatter plots of the predictions according to different subsets of the data (according to the region and the type of property). Another contains the metrics of the models, in the form of a csv file. Finally, a tex file is produced containing a translation of the csv file.

## Requirements

This repository requires `python 3.9.10`

## Contributor

Quentin Lambotte