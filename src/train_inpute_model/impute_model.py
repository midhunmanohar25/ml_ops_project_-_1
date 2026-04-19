import pathlib
import sys
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor


def load_data(data_path):
    # Load your dataset from a given path
    df = pd.read_csv(data_path)
    return df


def split_data(df, col):
    # Splitting the data into known and unknown    
    known = df[df[col].notna()]
    unknown = df[df[col].isna()]
    
    
    X_train = known.drop(columns=[col])
    y_train = known[col]
    
    X_test = unknown.drop(columns=[col])
    
    return  X_train, y_train, X_test


def scale_data(X_train, X_test):
    # Perform Scaling and Encoding of columns
    
    # Scaling the numerical columns
    scaler = StandardScaler()
    
    numerical_col = X_train.select_dtypes(include=['int', 'float']).columns
    
    X_train[numerical_col] = scaler.fit_transform(X_train[numerical_col])
    X_test[numerical_col] = scaler.transform(X_test[numerical_col])
    
    
    # Transfroming the categorical columns
    categorical_col = X_train.select_dtypes(include=['object']).columns
    
    transformer = ColumnTransformer(
        transformers=[
            ('tnf',OneHotEncoder(handle_unknown='ignore',sparse_output=False,drop='first'),categorical_col)
        ],remainder='passthrough'
    )
    
    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)
    
    return X_train, X_test


def impute_model(X_train, X_test, y_train, df, col):
    # Run Model and impute the missing values inside columns
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    df.loc[df[col].isna(), col] = model.predict(X_test)
    
    return df
 
 
def save_data(df, output_path):
    # Save the processed dataset
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path + '/cleaned_car_v2.csv', index=False)  



def main():
    
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    
    input_file = sys.argv[1]
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/data/processed'
    
    # Processes
    data = load_data(data_path)
    
    # Get missing columns
    missing_cols = data.isnull().sum()
    missing_cols = missing_cols[missing_cols > 0].sort_values().index
    
    for col in missing_cols:
        
        X_train, y_train, X_test = split_data(data, col)
        X_train_scale, X_test_scaled = scale_data(X_train, X_test)
        data = impute_model(X_train_scale, X_test_scaled, y_train, data, col)
    
    save_data(data, output_path)
    
if __name__ == "__main__":
    main()