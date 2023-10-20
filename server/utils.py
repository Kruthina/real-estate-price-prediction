import json
import pickle
import numpy as np

__LOCATIONS = None
__DATA_COLUMNS = None
__MODEL = None

def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __DATA_COLUMNS.index(location)
    except:
        loc_index = -1
    x = np.zeros(len(__DATA_COLUMNS))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index>=0:
        x[loc_index] = 1
    
    return round(__MODEL.predict([x])[0], 2)

def get_location_names():
    return __LOCATIONS

def load_saved_artifacts():
    global __DATA_COLUMNS
    global __LOCATIONS
    global __MODEL
    with open('./artifacts/columns.json', 'r') as file:
        __DATA_COLUMNS = json.load(file)['data_columns']
        __LOCATIONS = __DATA_COLUMNS[3:]
    
    with open('./artifacts/bengaluru_home_price_model2022-12-14.pkl', 'rb') as f:
        __MODEL = pickle.load(f)   