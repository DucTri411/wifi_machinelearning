from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import time

def train_models(X, y):
    print("Đang chia dữ liệu Train/Test...")
    # Div train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scaling data
    print("Đang chuẩn hoá dữ liệu (Scaling Data)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Logicstic Regression
    print("Đang train Logicstic Regression...")
    t0 = time.time()
    lr = SGDClassifier(
        loss='log_loss',
        max_iter=1000,
        tol=1e-3,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    lr.fit(X_train_scaled, y_train)
    print(f"-> Đã train xong LR trong {time.time() - t0:.2f} giây")

    # Randon Forest (Main Model)
    print("Đang train Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        verbose=2
    )
    rf.fit(X_train, y_train)

    return lr, rf, X_test, X_test_scaled, y_test