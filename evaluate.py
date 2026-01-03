from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)

    print(f"{model_name}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test, y_pred))