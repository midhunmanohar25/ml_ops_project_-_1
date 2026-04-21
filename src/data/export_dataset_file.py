import joblib
import pathlib
import sys

import pandas as pd

def load_data(data_path):
    # Load your dataset from a given path
    df = pd.read_csv(data_path)
    return df

def save_dataset(data, output_path):
    # Save the dataset to the specified output path
    joblib.dump(data, output_path + '/dataset.joblib')
    
def main():
    
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    
    input_file = sys.argv[1]
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/models'
    
    data = load_data(data_path)
    save_dataset(data, output_path)
    
if __name__== "__main__":
    main()