import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import shap

# Mapping of specific classes to major categories
# Define the mapping of specific classes to major categories
class_mapping = {
    'BenignTraffic': 'Benign',
    'DDoS-ICMP_Flood': 'DDoS',
    'DDoS-UDP_Flood': 'DDoS',
    'DDoS-TCP_Flood': 'DDoS',
    'DDoS-SYN_Flood': 'DDoS',
    'DDoS-PSHACK_Flood': 'DDoS',
    'DDoS-RSTFINFlood': 'DDoS',
    'DDoS-SynonymousIP_Flood': 'DDoS',
    'DDoS-ICMP_Fragmentation': 'DDoS',
    'DDoS-UDP_Fragmentation': 'DDoS',
    'DDoS-ACK_Fragmentation': 'DDoS',
    'DDoS-HTTP_Flood': 'DDoS',
    'DDoS-SlowLoris': 'DDoS',
    'DoS-UDP_Flood': 'DoS',
    'DoS-TCP_Flood': 'DoS',
    'DoS-SYN_Flood': 'DoS',
    'DoS-HTTP_Flood': 'DoS',
    'Mirai-greeth_flood': 'Mirai',
    'Mirai-udpplain': 'Mirai',
    'Mirai-greip_flood': 'Mirai',
    'Recon-HostDiscovery': 'Recon',
    'Recon-OSScan': 'Recon',
    'Recon-PortScan': 'Recon',
    'Recon-PingSweep': 'Recon',
    'VulnerabilityScan': 'Recon',
    'MITM-ArpSpoofing': 'Spoofing',
    'DNS_Spoofing': 'Spoofing',
    'BrowserHijacking': 'Web',
    'Backdoor_Malware': 'Web',
    'Uploading_Attack': 'Web',
    'SqlInjection': 'Web',
    'CommandInjection': 'Web',
    'XSS': 'Web',
    'DictionaryBruteForce': 'Bruteforce'

}

# Reshuffle the index to control sequence of task passed in the every episode in meta training.
label_mapping_key = {
    'Spoofing': 0,
    'Recon': 1,
    'DoS': 2,
    'DDoS': 3,
    'Benign': 4,
    'Mirai': 5,
    'Web': 6,
    'Bruteforce': 7
}


# Create a reverse mapping from index to label name
index_to_label = {index: label for label, index in label_mapping_key.items()}


# Function to get label name by index
def get_label_name(index):
    return index_to_label.get(index, "Unknown")


class DataGenerator(object):
    def __init__(self, total_classes, no_of_ways, no_of_shots_per_class, batch_size, data_file, config={}):
        """
        Args:
            total_classes: total of unique traffic classes
            no_of_ways: num of classes for (N-ways) classification
            num_samples_per_class: num samples to generate per class (K-shots) in one batch
            batch_size: size of meta batch size
        """
        self.batch_size = batch_size
        self.no_of_shots_per_class = no_of_shots_per_class
        self.no_of_ways = no_of_ways
        self.total_classes = total_classes

        # Load the network traffic data from the CSV file
        df = pd.read_csv(data_file)

        # Calculate the value counts for each label
        traffic_type_counts = df['label'].value_counts()

        # Sort the value counts
        sorted_traffic_type = traffic_type_counts.sort_values(ascending=False)
        print(sorted_traffic_type)

        # Apply the mapping to categorize the classes
        df['label'] = df['label'].map(class_mapping)

        # Calculate the value counts for each label
        traffic_type_counts = df['label'].value_counts()

        # Sort the value counts
        sorted_traffic_type = traffic_type_counts.sort_values(ascending=False)
        print(sorted_traffic_type)

        # Drop the corresponding rows from the DataFrame
        self.data_frame = df

        # Print the label mapping
        print("Label mapping:", label_mapping_key)

        # Print the index and count of each label
        label_counts = self.data_frame['label'].value_counts().sort_index()
        print("Label counts (index: count):")
        for index, count in label_counts.items():
            print(f"Index {index}: Count {count}")

        # Map the labels in the DataFrame to the predefined numeric values
        self.data_frame.loc[:, "label"] = self.data_frame["label"].map(label_mapping_key)

        # Print the label mapping
        print("Label mapping:", label_mapping_key)

        # Print the index and count of each label
        label_counts = self.data_frame['label'].value_counts().sort_index()
        print("Label counts (index: count):")
        for index, count in label_counts.items():
            print(f"Index {index}: Count {count}")

        self.labels_column = self.data_frame["label"]
        self.X = self.data_frame.drop("label", axis=1)  # Features
        self.y = self.data_frame['label']  # Labels

        self.no_of_features = len(self.data_frame.columns) - 1

    """
        Feature Engineering using Explainable Artificial Intelligence (XAI) 
    """

    def feature_selection(self):
        y = pd.Series(self.y)
        X_train, X_test, y_train, y_test = train_test_split(self.X, y, test_size=0.3, stratify=y, random_state=42)

        y_train = pd.Series(y_train)
        y_train = y_train.astype(int)

        print('-------------------------------------------------------------------------------------------------')
        print('                 Feature Engineering - Explainable Artificial Intelligence (XAI)                 ')
        print('-------------------------------------------------------------------------------------------------')

        # Build Random Forest Classifier Model for feature selection
        print('Create Random Forest Classifier Model for Feature Selection using XAI Technique')
        rf_clf = RandomForestClassifier(max_features=46, n_estimators=50, bootstrap=True)  # n_estimators=100

        y_train = pd.Series(y_train)
        y_train = y_train.astype(int)

        y_test = pd.Series(y_test)
        y_test = y_test.astype(int)

        print('Training Random Forest Model with X_train data ... ...')
        rf_clf.fit(X_train, y_train)

        # Make prediction on the testing data
        print('Predict y label with X_test data ...')
        y_pred = rf_clf.predict(X_test)

        # Classification Report
        print(classification_report(y_pred.astype(int), y_test))

        print('-------------------------------------------------------------------------------------------------')
        print('Create SHAP Tree Explainer on Random Forest Model')
        # Create the explainer
        explainer = shap.TreeExplainer(rf_clf)

        # Compute SHAP values
        print('Computing shap values ... ...')
        X = X_test.sample(n=500, random_state=42)
        shap_values = explainer.shap_values(X)

        shap.summary_plot(shap_values[:, :, 0], X.values, feature_names=X.columns,
                          max_display=46) 

        mean_shap_values = np.abs(shap_values[:, :, 0]).mean(axis=0)
        feature_importance = pd.Series(mean_shap_values, index=X.columns)
        sorted_features = feature_importance.sort_values(ascending=False)
        print(sorted_features)

        # Identify features with mean SHAP values less than zero
        negative_shap_features = feature_importance[feature_importance == 0]
        negative_feature_names = negative_shap_features.index.tolist()
        print("\nFeatures to be removed with mean SHAP values less than zero:", negative_feature_names)

        # Filter features where the SHAP value is not zero
        non_zero_features = feature_importance[feature_importance > 0]
        non_zero_feature_names = non_zero_features.index.tolist()
        print("Features with non-zero SHAP values:", non_zero_feature_names)

        # Append the y target label
        non_zero_feature_names.append('label')

        # Remove zero value features from the original dataframe
        self.data_frame = self.data_frame[non_zero_feature_names]
        print(f'No features in data frame: {len(self.data_frame.columns)}')
        self.no_of_features = len(self.data_frame.columns) - 1

    def generate_data_tensor(self, index, test_sample_size, train=True):
        """
        Generates a data tensor for 2-way K-shot learning.

        Args:
            train: Boolean, whether to use the training or evaluation split.

        Returns:
            support_set: Tensor containing the support set data.
            query_set: Tensor containing the query set data.
        """

        # Iterate through the dataset
        support_set = []
        query_set = []
        test_set = []

        for _ in range(self.batch_size):
            support_batch = []
            query_batch = []
            test_batch = []

            for i in range(self.no_of_ways):
                class_data = self.data_frame[self.data_frame['label'] == index]
                print('Total data sample in class ', get_label_name(index))
                print(class_data.shape)
                index = index + 1

                # # Select all samples for each class randomly
                # class_samples = class_data.sample(n=class_data.shape[0]) #, random_state=42

                # Select continuous samples for each class
                class_samples = class_data.reset_index(drop=True)

                # Select K shots samples as support set
                support_samples = class_samples.iloc[:self.no_of_shots_per_class]

                # The rest all samples will be query set
                # query_samples = class_samples.iloc[self.no_of_shots_per_class:]
                # Get the last no of shots as query set
                query_samples = class_samples.iloc[-self.no_of_shots_per_class:]

                remaining_samples = class_samples.iloc[self.no_of_shots_per_class:-self.no_of_shots_per_class]
                test_samples = remaining_samples.sample(n=test_sample_size, random_state=42)

                print('|\tSupport Sample Set\t|\tQuery Sample Set\t|\t Test Sample\t|')
                print("|\t\t", support_samples.shape, "\t\t|\t\t", query_samples.shape, "\t\t|\t", test_samples.shape,
                      "\t|")

                support_batch.extend(support_samples.values.tolist())
                query_batch.extend(query_samples.values.tolist())
                test_batch.extend(test_samples.values.tolist())

            support_set.append(support_batch)
            query_set.append(query_batch)
            test_set.append(test_batch)

        support_set = tf.convert_to_tensor(support_set, dtype=tf.float32)
        query_set = tf.convert_to_tensor(query_set, dtype=tf.float32)
        test_set = tf.convert_to_tensor(test_set, dtype=tf.float32)

        return support_set, query_set, test_set

