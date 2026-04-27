import pathlib
import joblib
import sys
import pandas as pd
import numpy as np

from sklearn import metrics
from dvclive import Live
from matplotlib import pyplot as plt

def evaluate(pipeline, X, y, split, live):
    """
    Log regression metrics and actual vs predicted plots using DVCLive.
    """
    # 1. Get predictions
    # Note: If your model requires encoded data, ensure X is preprocessed
    predictions = pipeline.predict(X)

    # 2. Calculate Regression Metrics
    mae = metrics.mean_absolute_error(y, predictions)
    mse = metrics.mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y, predictions)

    # 3. Log metrics to DVCLive summary
    if not live.summary:
        live.summary = {}
    
    live.summary[f"{split}_mae"] = mae
    live.summary[f"{split}_rmse"] = rmse
    live.summary[f"{split}_r2"] = r2

    # 4. Create Actual vs Predicted plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y, predictions, alpha=0.4, color='teal')
    # Draw the "Perfect Prediction" line
    line_coords = [y.min(), y.max()]
    ax.plot(line_coords, line_coords, 'r--', lw=2)
    
    ax.set_xlabel('Actual Log-Price')
    ax.set_ylabel('Predicted Log-Price')
    ax.set_title(f'Actual vs Predicted - {split.capitalize()} Set')
    
    # Save to DVCLive
    live.log_image(f"plots/actual_vs_pred_{split}.png", fig)
    plt.close(fig)

def save_feature_importance(live, pipeline, feature_names):
    """
    Saves a plot of which features (year, kms, company) matter most.
    """
    model_step = pipeline.named_steps['model']
    
    if hasattr(model_step, 'feature_importances_'):
        importances = model_step.feature_importances_
        # Sort features by importance
        indices = np.argsort(importances)[-10:]  # Top 10
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(indices)), importances[indices], align='center', color='skyblue')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_title("Top 10 Feature Importances (Gradient Boosting)")
        
        live.log_image("plots/importance.png", fig)
        plt.close(fig)

def main():
    
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    
    # Load the model.
    pipeline_file = sys.argv[1]
    pipeline = joblib.load(pipeline_file)
    
    # Load the data.
    input_file = sys.argv[2]
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir / 'dvclive'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    
    
    # 2. Load Data (assuming names from your pipeline)
    TARGET = 'Price'
    
    # We drop NaNs just in case for the evaluation step
    train_df = pd.read_csv(data_path + '/train.csv')
    test_df = pd.read_csv(data_path + '/test.csv')

    # Separate Features and Target BEFORE preprocessing
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]

    # Correctly extract feature names from the preprocessor step
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    
    # 4. Start DVCLive session
    with Live(output_path.as_posix(), dvcyaml=True) as live:
        evaluate(pipeline, X_train, y_train, "train", live)
        evaluate(pipeline, X_test, y_test, "test", live)
        save_feature_importance(live, pipeline, feature_names)

if __name__ == "__main__":
    main()
    
    