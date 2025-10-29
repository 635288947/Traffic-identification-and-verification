import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

feature_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty_level'
]


def load_data(filepath):
    df = pd.read_csv(filepath, header=None, names=feature_names)
    return df
def preprocess_data(df, label_encoders=None, scaler=None, is_train=True):
    df = df.copy()
    print(f"（load_data后）: {df.shape[1]}")
    df['binary_class'] = df['attack_type'].apply(lambda x: 0 if x == 'normal' else 1)
    df = df.drop(['attack_type', 'difficulty_level'], axis=1)
    print(f"原始数据列数（load_data后）: {df.shape[1]}")
    categorical_cols = ['protocol_type', 'service', 'flag']
    if is_train:
        label_encoders = {col: LabelEncoder() for col in categorical_cols}
        for col in categorical_cols:
            df[col] = label_encoders[col].fit_transform(df[col].astype(str))
        numeric_cols = [col for col in df.columns if col not in categorical_cols + ['binary_class']]
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        for col in categorical_cols:
            df[col] = df[col].apply(
                lambda x: label_encoders[col].transform([x])[0]
                if x in label_encoders[col].classes_ else -1
            )
        numeric_cols = [col for col in df.columns if col not in categorical_cols + ['binary_class']]
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df, label_encoders, scaler
def main():
    train_df = load_data('/root/yzh/DQN/ids/KDDTrain+.txt')
    test_df = load_data('/root/yzh/DQN/ids/KDDTest+.txt')
    train_data, label_encoders, scaler = preprocess_data(train_df, is_train=True)
    test_data, _, _ = preprocess_data(test_df, label_encoders, scaler, is_train=False)
    def extract_features_labels(df):
        features = df.drop(['binary_class'], axis=1)
        labels = df['binary_class']
        return features.values.astype(np.float32), labels.values
    训练集形状: (125973, 42)
    X_train, y_train = extract_features_labels(train_data)
    X_test, y_test = extract_features_labels(test_data)

    feature_list = list(train_data.drop(['binary_class'], axis=1).columns)
    np.savez(
        'nsl_kddlabel.npz',
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        feature_names=feature_list
    )

    print("预处理完成！")
    print(f"训练集形状: {X_train.shape}")
    print(f"测试集形状: {X_test.shape}")


if __name__ == "__main__":
    main()