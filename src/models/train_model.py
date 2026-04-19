import pathlib
import sys
import yaml
import joblib

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(data_path):
    # Load your dataset from a given path
    df = pd.read_csv(data_path)
    return df

def data_preprocessor(df):
    
    numerical_col = df.select_dtypes(['int','float']).columns.to_list()
    categorical_col = df.select_dtypes(['object']).columns.to_list()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_col),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_col)
        ],
        remainder='passthrough'
    )
    return preprocessor


def split_train_data(train_df, target):
    # Splitting the train data into target_features and target for feeding model
    input_features = train_df.drop(columns=[target])
    target_feature = train_df[target]
    return input_features, target_feature


def train_model(input_features, target_feature, learning_rate, max_depth, min_samples_leaf, min_samples_split, n_estimators):
    # Train your machine learning model
    model = GradientBoostingRegressor(learning_rate=learning_rate, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, n_estimators=n_estimators)
    model.fit(input_features, target_feature)
    return model

    
def save_model(model, output_path):
    # Save the trained model to the specified output path
    joblib.dump(model, output_path + '/model.joblib')
    

def main():
    
    # Setup paths
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))["train_model"]
    
    input_file = sys.argv[1]
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/models'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # 1. Load and Split
    data = load_data(data_path)
    TARGET = 'Price'
    input_features, target_feature = split_train_data(data, TARGET)
    
    # 2. Preprocess
    preprocessor = data_preprocessor(input_features)
    transformed_input = preprocessor.fit_transform(input_features)
    
    # 3. Train
    trained_model = train_model(transformed_input, target_feature, params['learning_rate'], params['max_depth'], params['min_samples_leaf'], params['min_samples_split'], params['n_estimators'])
    
    # 4. Save
    save_model(trained_model, output_path)
    

if __name__ == "__main__":
    main()
    
    
    