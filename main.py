from preprocess import load_and_preprocess
from train import train_models
from evaluate import evaluate_model

def main():
    X, y = load_and_preprocess()

    lr, rf, X_test, X_test_scaled, y_test = train_models(X, y)

    evaluate_model(lr, X_test_scaled, y_test, "Logistic Regression")

    evaluate_model(rf, X_test, y_test, "Random Forest")

if __name__ == "__main__":
    main()