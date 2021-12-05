#!/usr/bin/env python3

# importing libraries
from datetime import datetime
from flask import Flask, escape
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import  RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas as pd
import pandas.api.types as ptypes
import seaborn as sns
import sys


app = Flask(__name__)

def load_dataset():
    # Loads the CSV Dataset to feed the model
    df = pd.read_csv('dataset_v2.csv')
    # Filter by Interquartile range
    Q1 = df.revenue.quantile(.25)
    Q3 = df.revenue.quantile(.95)
    IQR = 1.5*(Q3-Q1)
    df = df[df.revenue<IQR]
    return df



def create_train_split(data):
    # Separate the independent and target variable 
    train_X = data.drop(columns=['user_id','device_family', 'revenue'])
    train_Y = data['revenue']

    # Split the data
    return train_X, train_Y

class ToLowerTransformer():
    # Custom transformer to lowercase columns
    def __init__(self, columns=None):
	    self.columns = columns
    def transform(self, X, y=None, **fit_params):
	    for column in self.columns:
		    X[column] = X[column].str.lower()
	    return X
    def fit(self, X, y=None, **fit_params):
	    return self


def train_model(data):
    # Each type of feature has a different processing.
    numerical_features = ['event_1', 'event_2']
    categorical_features = ['country', 'source', 'platform']

    # Numerical values should be rescaled
    numeric_pipeline = Pipeline(steps=[
	    ('scale', StandardScaler())
    ])

    # Categorical variables needs to be transformed and encoded properly
    categorical_pipeline = Pipeline(steps=[
	    ('to_lower', ToLowerTransformer(['platform'])),
	    ('impute', SimpleImputer(strategy='most_frequent')),
	    ('one-hot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    pre_processor = ColumnTransformer(transformers=[
	    ('number', numeric_pipeline, numerical_features),
	    ('category', categorical_pipeline, categorical_features)
    ])

    # Instanciate a RFR model
    rfr_model = RandomForestRegressor(max_depth=20)

    # Split training data
    train_x, train_y = create_train_split(data)
    rfr_pipeline = Pipeline(steps=[
            ('preprocess', pre_processor),
                ('model', rfr_model)
                ])

    # Fit the model and return
    _ = rfr_pipeline.fit(train_x, train_y)
    return rfr_pipeline


df = load_dataset()
model = train_model(df)


@app.route('/')
def index():
    return 'Usage: GET /prediction/<event_1>/<event_2>/<country>/<source>/<platform>\n'

@app.route('/prediction/<event_1>/<event_2>/<country>/<source>/<platform>', methods=['GET'])
def predict(event_1, event_2, country, source, platform):
    print(f'Predicting value: \n')
    point_to_predict = pd.DataFrame({
        'event_1': event_1,
        'event_2': event_2,
        'country': country,
        'source': source,
        'platform': platform
        }, index=['event_1'])
    predicted_value = model.predict(point_to_predict)[0]

    return {'predicted_revenue': predicted_value}

if __name__ == '__main__':
    app.run(debug=True)


