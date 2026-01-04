from preprocess import load_and_preprocess
from train import train_models
from evaluate import evaluate_model
from sklearn.metrics import classification_report

def main():
    X, y = load_and_preprocess()

    lr, rf, X_test, X_test_scaled, y_test, y_pred_iso = train_models(X, y)

    evaluate_model(lr, X_test_scaled, y_test, "Logistic Regression")

    evaluate_model(rf, X_test, y_test, "Random Forest")

    print("Isolation Forest")
    print(classification_report(y_test, y_pred_iso))

if __name__ == "__main__":
    main()