import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, OrthogonalMatchingPursuit, ElasticNet
from sklearn.tree import DecisionTreeRegressor
import pickle
from datetime import datetime
import json
from utils import *

def load_data(path):
    # Data Cleaning
    df = pd.read_csv(path)
    df.drop(['area_type','society','balcony','availability'], axis='columns', inplace=True)
    df.dropna(inplace=True)
    df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))
    df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
    return df

def rm_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        mean = np.mean(subdf['price_per_sqft'])
        std = np.std(subdf['price_per_sqft'])
        reduced_df = subdf[(subdf['price_per_sqft']>(mean-std)) & (subdf['price_per_sqft']<=(mean+std))]
        df_out = pd.concat([df_out, reduced_df],ignore_index=True)
    return df_out

def rm_bhk_outlisers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df['price_per_sqft']),
                'std': np.std(bhk_df['price_per_sqft']),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices,bhk_df[bhk_df['price_per_sqft']<(stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')
  
def feature_engg(df):
    df['price_per_sqft'] = df['price']*100000/df['total_sqft']
    
    # To Reduce Dimensions
    df['location'] = df['location'].apply(lambda x: x.strip())
    location_stats = df.groupby('location')['location'].agg('count').sort_values(ascending=False)
    df['location'] = df['location'].apply(lambda x: 'other' if x in location_stats[location_stats<=10] else x)
    
    # Outlier Removal
    df = df[~(df['total_sqft']/df['bhk']<300)] # Assumed 300sqft is min size of bedroom 
    df = rm_pps_outliers(df)
    #plot_scatter_plot(df, "Hebbal")
    df = rm_bhk_outlisers(df)
    df = df[df['bath']<df['bhk']+2]            # Assumed that a house will not have more than (no. of bedrooms+2) bathrooms
    #plot_hist(df['price_per_sqft'],'Price Per Square Feet Frequency','Price Per Square Feet')
    df.drop(['size','price_per_sqft'],axis='columns', inplace=True)
    return df

def preprocessing(df):
    dummies = pd.get_dummies(df['location'])
    df = pd.concat([df,dummies.drop('other',axis='columns')], axis='columns')
    df.drop('location', axis='columns', inplace=True)
    return df

def best_model_search(X, y):
    print("Best Model Search Started, Please Wait.....")
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'positive': [True, False]
            }
        },
        'orthogonal_matching_pursuit':{
            'model': OrthogonalMatchingPursuit(),
            'params': { 
                'fit_intercept': [True, False],
            }
        },
        'elastic_net': {
            'model': ElasticNet(),
            'params': {
                'l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                'fit_intercept': [True, False]
            }
        },
        'lasso':{
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['friedman_mse', 'poisson', 'absolute_error', 'squared_error'],
                'splitter': ['best', 'random']
            }
        } 
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=23)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X, y)
        scores.append({'model': algo_name, 'best_score': gs.best_score_, 'best_params': gs.best_params_})
    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(ind_variables.columns==location)[0][0]
    x = np.zeros(len(ind_variables.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index>=0:
        x[loc_index] = 1
    return lr_model.predict([x])[0]

if __name__ == "__main__":
    dataset = load_data('bengaluru_house_data.csv')
    print("Data Loaded!")
    dataset = feature_engg(dataset)
    dataset = preprocessing(dataset)
    print("Data Preprocessed")

    ind_variables = dataset.drop('price', axis='columns')
    dep_variables = dataset['price'] 
    # scores_df = best_model_search(ind_variables, dep_variables)
    # print(scores_df)
    print("Training Model, Please Wait....")
    X_train, X_test, y_train, y_test = train_test_split(ind_variables.values, dep_variables.values, test_size=0.2, random_state=23)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    score = lr_model.score(X_test, y_test)
    print(f"Model Score = {score}")

    date_time = datetime.now()
    date = date_time.date()
    with open(f'saved_model/bengaluru_home_price_model{date}.pkl', 'wb') as file:
        pickle.dump(lr_model, file)
    print("Model Saved!")

    columns = {'data_columns': ind_variables.columns.tolist()}
    with open('saved_model/columns.json', 'w') as file:
        file.write(json.dumps(columns))
        file.close()
    #estimate = predict_price("Indira Nagar", 1250, 3, 2)
    #print(estimate)
