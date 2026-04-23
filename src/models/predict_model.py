import pathlib
import sys
import yaml
import joblib

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def import_model(model_path):
    model = joblib.load(model_path)
    return model

def load_data(data_path):
    data = pd.read_csv(data_path)
    return data

def build_and_train_pipeline(train_df, target, model):
    # Creating  a pipeline
    
    X_train_df = train_df.drop(columns=[target])
    y_train_df = train_df[target]
    
    numerical_col = X_train_df.select_dtypes(['number']).columns.to_list()
    categorical_col = X_train_df.select_dtypes(['object', 'category']).columns.to_list()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_col),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_col)
        ],
        remainder='passthrough'
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    pipeline.fit(X_train_df, y_train_df)
    
    return pipeline
    
def save_pipeline(pipeline, output_path):
    # Save the Pipeline to the specified output path
    joblib.dump(pipeline, output_path + '/pipeline.joblib')
    
def main():
    
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    
    
    model_file = sys.argv[1]
    input_file = sys.argv[2]
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/models'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    
    TARGET = "Price"
    model = import_model(model_file)
    train_data = load_data(data_path)
    pipeline = build_and_train_pipeline(train_data, TARGET, model)
    save_pipeline(pipeline, output_path)
    
if __name__ == "__main__":
    main()
    