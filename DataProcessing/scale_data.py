import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, Normalizer, QuantileTransformer
from sklearn.preprocessing import RobustScaler



class DataPreprocessor:
    def __init__(self, raw_data_directory, output_directory, scaler_names=None):
        self.raw_data_directory = raw_data_directory
        self.output_directory = output_directory
        self.scaler_names = scaler_names if scaler_names else ['StandardScaler']
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

    def get_scaler(self, scaler_name):
        """Return the corresponding scaler based on name."""
        if scaler_name == 'StandardScaler':
            return StandardScaler()
        elif scaler_name == 'MinMaxScaler':
            return MinMaxScaler()
        elif scaler_name == 'Normalizer':
            return Normalizer()
        elif scaler_name == 'QuantileTransformer':
            return QuantileTransformer()
        elif scaler_name == 'RobustScaler':  # 👈 Thêm dòng này
            return RobustScaler()
        else:
            raise ValueError(f"Scaler {scaler_name} not recognized")

    def preprocess_file(self, file_path, target_label, columns_to_drop):
        """Preprocess the given file and return processed data."""
        print(f"Processing file: {file_path}")
        file_full_path = os.path.join(self.raw_data_directory, file_path)
        
        # Read data from file
        df = pd.read_csv(file_full_path)
        
        # Drop specified columns

        columns_exist = [col for col in columns_to_drop if col in df.columns]
        df.drop(columns=columns_exist, inplace=True)

        # Drop columns with all zeros
        columns_to_drop_zeros = [col for col in df.columns if (df[col] == 0).all()]
        df.drop(columns=columns_to_drop_zeros, inplace=True)
      
        df.dropna(inplace=True)

        # Drop duplicate rows
        df_unique = df.drop_duplicates()

        # Separate target column and features
        y = df_unique[target_label]
        X = df_unique.drop(columns=target_label)

        # Split into train and test datasets (70% train, 30% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Encode categorical features
        label_encoders = {}
        for col in X_train.select_dtypes(include=['object']).columns:
            label_encoders[col] = LabelEncoder()
            X_train[col] = label_encoders[col].fit_transform(X_train[col].astype(str))

        for col in X_test.select_dtypes(include=['object']).columns:
            X_test[col] = X_test[col].apply(lambda x: 'unknown' if x not in label_encoders[col].classes_ else x)
            label_encoders[col].classes_ = np.append(label_encoders[col].classes_, 'unknown')
            X_test[col] = label_encoders[col].transform(X_test[col].astype(str))

        # Handle missing values (fill with mean)
        # X_train = X_train.apply(lambda col: col.fillna(col.mean()), axis=0)
        # X_test = X_test.apply(lambda col: col.fillna(X_train[col.name].mean()), axis=0)

        # Apply scaling to data
        for scaler_name in self.scaler_names:
            print(f"Scaling data with {scaler_name}")
            scaler = self.get_scaler(scaler_name)
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Encode target variable y_train and y_test
            label_encoder_y = LabelEncoder()
            y_train_encoded = label_encoder_y.fit_transform(y_train)
            y_test_encoded = label_encoder_y.transform(y_test)
            if (file_path == "data_BoTIoT.csv"):
                y_train_encoded = 1 - y_train_encoded  # Chuyển 0 thành 1 và 1 thành 0
                y_test_encoded  = 1 - y_test_encoded    # Tương tự với y_test
            
            sort_index = np.argsort(y_train_encoded)
            X_train_sorted = X_train_scaled[sort_index]
            y_train_sorted = y_train_encoded[sort_index]

            # Convert arrays back to DataFrames
            X_train_df = pd.DataFrame(X_train_sorted)
            y_train_df = pd.DataFrame(y_train_sorted)
            X_test_df = pd.DataFrame(X_test_scaled)
            y_test_df = pd.DataFrame(y_test_encoded)

            # Concatenate train and test data
            train_data = pd.concat([X_train_df, y_train_df], axis=1, ignore_index=True)
            test_data = pd.concat([X_test_df, y_test_df], axis=1, ignore_index=True)

            # Save processed data
            train_file_path = os.path.join(self.output_directory, f'Train_{scaler_name}_{file_path.split("/")[-1]}')
            test_file_path = os.path.join(self.output_directory, f'Test_{scaler_name}_{file_path.split("/")[-1]}')

            train_data.to_csv(train_file_path, index=False)
            test_data.to_csv(test_file_path, index=False)

            print(f"Train data saved to: {train_file_path}")
            print(f"Test data saved to: {test_file_path}")

    def preprocess_multiple_files(self, file_paths, target_labels, columns_to_drop):
        """Preprocess multiple files."""
        for idx, file_path in enumerate(file_paths):
            print("="*50)
            self.preprocess_file(file_path, target_labels[idx], columns_to_drop[idx])
            print("="*50)


# Usage Example
if __name__ == "__main__":
    
    # Dùng thư mục hiện tại làm nơi chứa dữ liệu và lưu dữ liệu
    current_directory = os.getcwd()
    raw_data_directory = current_directory
    output_directory = "~/GMM-nfst/Datascaled"

    # Danh sách file cần xử lý
    file_paths = ['data_N_BaIoT.csv', 'data_CICIoT2023.csv','data_BoTIoT.csv','data_ToNIoT.csv']
    # file_paths = ['data_ToNIoT.csv']
    
    # Nhãn đích và cột cần loại bỏ tương ứng với từng file
    target_labels = ['Binary_label', 'Binary_label','Binary_label','Binary_label']
    columns_to_drop = [
        ["Unnamed: 0", 'Category_label', 'Default_label'],
        ["Drate",'DHCP', 'ece_flag_number', 'Telnet', 'SMTP', 'IRC', 'Category_label','Default_label'],
        ["smac", "dmac", "soui", "doui", "sco", "dco",'Default_label','category','Category_label', 'attack'],
        ['ts', 'src_ip', 'dst_ip', 'src_port', 'dst_port','dns_query', 'ssl_subject', 'ssl_issuer', 'http_uri', 'http_referrer', 'http_user_agent', 'http_orig_mime_types', 'http_resp_mime_types','weird_name', 'weird_addl', 'weird_notice','Default_label', 'Category_label', 'label']
    ]

    # Các phương pháp chuẩn hóa dữ liệu
    # scaler_names = ['StandardScaler','QuantileTransformer','MinMaxScaler','Normalizer','RobustScaler']
    scaler_names = ['StandardScaler','QuantileTransformer','MinMaxScaler','Normalizer','RobustScaler']
    
    # Khởi tạo và chạy xử lý
    preprocessor = DataPreprocessor(raw_data_directory, output_directory, scaler_names)
    preprocessor.preprocess_multiple_files(file_paths, target_labels, columns_to_drop)
