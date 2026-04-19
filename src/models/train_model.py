import pathlib
import sys
import yaml
import joblib

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

def split_train_data(data_path, target):
    # Splitting the train data into target_features and target for feeding model
    train_df = pd.read_csv(data_path)
    input_features = train_df.drop(columns=[target], axis=1)
    target_feature = train_df[target]
    return input_features, target_feature


def train_model(input_features, target_feature, learning_rate, max_depth, min_samples_leaf, min_samples_split, n_estimators):
    # Train your machine learning model
    model = GradientBoostingClassifier(learning_rate=learning_rate, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, n_estimators=n_estimators)
    model.fit(input_features, target_feature)
    return model
    
    
def save_model(model, output_path):
    # Save the trained model to the specified output path
    joblib.dump(model, output_path + '/model.joblib')
    

def main():
    
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))["train_model"]
    
    input_file = sys.argv[1]
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/models'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    
    
    TARGET = 'Price'
    input_features, target_feature = split_train_data(data_path, TARGET)
    
    trained_model = train_model(input_features, target_feature, params['learning_rate'], params['max_depth'], params['min_samples_leaf'], params['min_samples_split'], params['n_estimators'])
    save_model(trained_model, output_path)
    

if __name__ == "__main__":
    main()
    
    
    