import pandas as pd
import numpy as np
import glob

def load_and_preprocess(data_path="MachineLearningCVE/*.csv"):
    print(f"Đang tải dữ liệu...")
    # Gop 8 file
    files = glob.glob(data_path)
    df_list = [pd.read_csv(f) for f in files]
    data = pd.concat(df_list, ignore_index=True)

    data.columns = data.columns.str.strip()
    print(f"Kích thước ban đầu: {data.shape}")

    # Xoa cot khong can thiet
    drop_cols = [
        'Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'SimillarHTTP'
    ]
    data.drop(columns=drop_cols, inplace=True, errors='ignore')
    
    # Sua loi NaN/Infinity
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    data.drop_duplicates(inplace=True)
    print(f"Kích thước sau khi xóa duplicate: {data.shape}")
    
    # Chuan hoa Binary Label
    data['Label'] = data['Label'].apply(
        lambda x: 0 if x == 'BENIGN' else 1
    )

    # Tach x, y
    X = data.drop('Label', axis=1)
    y = data['Label']

    return X, y