import pathlib
import sys

import pandas as pd
import numpy as np


def load_data(data_path):
    # Load your dataset from a given path
    df = pd.read_csv(data_path)
    return df


def build_feature(df):
    # Feature engineering tasks
    current_date = 2020
    
    # Car age
    df['car_age'] = current_date - df['year']
    
    # Luxury flag
    luxury_brand = ['bmw', 'audi', 'mercedes', 'jaguar', 'mini']

    df['is_luxury'] = df['company'].str.lower().isin(luxury_brand).astype(int)
    
    # Usage intensity
    df['kms_per_year'] = df['kms_driven'] / (df['car_age'] + 1)
    
    # Log transform
    df['Price'] = np.log1p(df['Price'])
    
    return df


def save_data(df, output_path):
    # Save the processed dataset
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path + '/cleaned_car_v1.csv', index=False)  



def main():
    
    curr_path = pathlib.Path(__file__)
    home_dir = curr_path.parent.parent.parent
    
    
    input_file = sys.argv[1]
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/data/processed'
    
    data = load_data(data_path)
    updated_data = build_feature(data)
    save_data(updated_data, output_path)
    
if __name__=="__main__":
    main()
    