from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)

    print(f"{model_name}")
    print(classification_report(y_test, y_pred))

    try:
        auc = roc_auc_score(y_test, y_pred)
        print(f"ROC-AUC Score: {auc:.4f}")
    except:
        pass

    print("Confusion Matrix: ")
    print(confusion_matrix(y_test, y_pred))