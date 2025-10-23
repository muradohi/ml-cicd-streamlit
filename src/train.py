import yaml
from data import load_data
from model import build_model, train_eval, save_model, load_model, model_pipeline
config_path = '/Users/murad/ml-cicd-streamlit/ml-cicd-streamlit/configs/config.yml'
def main():
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    X_train, X_test, y_train, y_test, num_feat, cat_feat = load_data(config_path)
    print("Data loaded successfully.")
    print("Training data shape:", X_train.shape, y_train.shape)

    preprocessor = model_pipeline(num_feat, cat_feat)
    clf, grid_cv = build_model(config_path, num_feat, cat_feat)
    print("Model pipeline created.")

    best_params, best_score, best_model, y_pred, accuracy, confusion_matrix = train_eval(X_train, y_train, X_test, y_test, num_feat, cat_feat)
    print("Model trained successfully.")

    print("Best Parameters:", best_params)
    print("Best Cross-Validation Score:", best_score)

    model_path = config['model']['path']
    save_model(best_model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()