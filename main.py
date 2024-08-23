import tensorflow as tf
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import Reshape, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data_generator import DataGenerator
import numpy as np
from tensorflow.keras.losses import MeanSquaredError

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


"""
    Create 2D-CNN model 
"""


def create_cnn_model(no_of_features):
    model = Sequential()

    # Reshape the input to a 2D shape suitable for Conv2D layersl
    model.add(Reshape((1, no_of_features, 1), input_shape=(no_of_features,)))

    # Add convolutional layers
    model.add(Conv2D(32, (1, 1), activation='relu', input_shape=(1, no_of_features, 1)))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Conv2D(64, (1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))

    # Flatten the output and add dense layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='CategoricalCrossentropy', metrics=['accuracy'])

    return model


"""
    Plot Loss vs Epochs graphs to visualize the behavior of training loss and validation loss. 
"""


def plot_metrics(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.plot(epochs, val_loss, marker='o', linestyle='-', color='r', label='Validation Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


"""
    Plot confusion matrix. 
"""


def plot_cm(y, y_pred, class_A, class_B):
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[f'{class_A}', f'{class_B}'],
                yticklabels=[f'{class_A}', f'{class_B}'])

    # Using numpy to get unique values and their counts
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"Class y {label}: {count} instances")

    # Using numpy to get unique values and their counts
    unique, counts = np.unique(y_pred, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"Class y pred {label}: {count} instances")

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def convert_to_binary(labels):
    # Convert labels to a tensor if they are not already
    labels = tf.convert_to_tensor(labels)

    # Get unique labels
    unique_labels = tf.unique(labels).y

    # Ensure there are exactly two distinct labels
    if tf.size(unique_labels) != 2:
        raise ValueError("The label list must contain exactly two distinct classes.")

    # Sort the unique labels to consistently map to 0 and 1
    sorted_unique_labels = tf.sort(unique_labels)

    # Map the labels to binary values using tf.where
    binary_labels = tf.where(labels == sorted_unique_labels[0], 0, 1)

    return binary_labels


def performance_evaluation(y, y_pred):
    #####################################################################################
    #              Model Performance [Accuracy, F1-score, Detection Rate]               #
    #####################################################################################

    # Calculate F1 score for each class separately
    f1_per_class = f1_score(y, y_pred, average=None)

    # Calculate micro-average F1 score
    f1_micro = f1_score(y, y_pred, average='micro')

    # Calculate macro-average F1 score
    f1_macro = f1_score(y, y_pred, average='macro')

    # Calculate weighted-average F1 score
    f1_weighted = f1_score(y, y_pred, average='weighted')

    # Calculate recall score (detection rate)
    recall = recall_score(y, y_pred)

    print("F1 score per class:", f1_per_class)
    print("Micro-average F1 score:", f1_micro)
    print("Macro-average F1 score:", f1_macro)
    print("Weighted-average F1 score:", f1_weighted)
    print("Detection rate (Recall):", recall)


def maml(meta_model, total_classes, train=True):
    losses = []
    total_loss = 0.

    test_sample_size = 2000

    if train:
        class_index = 0
        end_index = total_classes - 2
        meta_epochs = 2
        # test_sample_size = 2000
    else:
        class_index = total_classes - 2
        end_index = total_classes
        meta_epochs = 3
        # test_sample_size = 70

    for step in range(meta_epochs):
        sum_gradients = [tf.zeros_like(variable) for variable in meta_model.trainable_variables]

        for episode in range(class_index, end_index, 2):
            ##########################################################################
            #   Data Partitioning
            ##########################################################################
            label_A = get_label_name(episode)
            label_B = get_label_name(episode + 1)
            print('------------------------------------------------------------------------------------')
            print('Generated data for classes', label_A, 'and', label_B)
            print('------------------------------------------------------------------------------------')
            support_set, query_set, test_set = data_generator.generate_data_tensor(episode, test_sample_size,
                                                                                   train=True)
            print('------------------------------------------------------------------------------------')

            support_set_data = tf.squeeze(support_set, axis=0)
            query_set_data = tf.squeeze(query_set, axis=0)
            test_set_data = tf.squeeze(test_set, axis=0)

            y_label_support_set = support_set_data[:, -1]
            print(y_label_support_set)
            y_label_support_set = convert_to_binary(y_label_support_set)
            print(y_label_support_set)
            x_features_support_set = support_set_data[:, :-1]

            y_label_query_set = query_set_data[:, -1]
            y_label_query_set = convert_to_binary(y_label_query_set)
            x_features_query_set = query_set_data[:, :-1]

            y_label_test_set = test_set_data[:, -1]
            y_label_test_set = convert_to_binary(y_label_test_set)
            x_features_test_set = test_set_data[:, :-1]

            # One-hot encode the labels
            y_label_support_set_one_hot = tf.keras.utils.to_categorical(y_label_support_set, num_classes=no_of_ways)
            y_label_query_set_one_hot = tf.keras.utils.to_categorical(y_label_query_set, num_classes=no_of_ways)
            y_label_test_set_one_hot = tf.keras.utils.to_categorical(y_label_test_set, num_classes=no_of_ways)

            # Create a Base Learner Model for each episode
            learner_model = create_cnn_model(data_generator.no_of_features)
            learner_optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

            history = learner_model.fit(x_features_support_set, y_label_support_set_one_hot, batch_size=32,
                                        epochs=epochs,
                                        validation_data=(x_features_query_set, y_label_query_set_one_hot))

            plot_metrics(history)

            # 1. Inner loop: Update the model copy on the current task
            with tf.GradientTape() as support_tape:
                # Assuming `learner_model` is your Keras model
                trainable_variables = learner_model.trainable_variables
                # Ensure variables are of type tf.Variable
                trainable_variables = [tf.convert_to_tensor(var) for var in trainable_variables]

                support_tape.watch(trainable_variables)
                support_pred = learner_model(tf.convert_to_tensor(x_features_support_set, dtype=tf.float32))
                support_loss = meta_loss_fn(tf.convert_to_tensor(y_label_support_set_one_hot, dtype=tf.float32),
                                            support_pred)
            learner_grads = support_tape.gradient(support_loss, learner_model.trainable_variables)
            learner_optimizer.apply_gradients(zip(learner_grads, learner_model.trainable_variables))

            # 2. Compute gradients with respect to the test data
            with tf.GradientTape() as query_tape:
                query_tape.watch(trainable_variables)
                query_pred = learner_model(tf.convert_to_tensor(x_features_query_set, dtype=tf.float32))
                query_loss = meta_loss_fn(tf.convert_to_tensor(y_label_query_set_one_hot, dtype=tf.float32), query_pred)

            #######################################################################
            # Evaluate inner learner model on few shot data and test data
            ######################################################################

            print('^^^^^^^^^^^^^^^^^^ Evaluate model on few shot data ^^^^^^^^^^^^^^')
            y_pred = learner_model.predict(x_features_query_set)
            y_pred = np.argmax(y_pred, axis=1)
            print(y_pred)
            cm = confusion_matrix(y_label_query_set, y_pred)
            plot_cm(y_label_query_set, y_pred, label_A, label_B)
            # print(cm)
            accuracy = accuracy_score(y_label_query_set, y_pred)
            print(f'Accuracy:{accuracy}')

            print('^^^^^^^^^^^^^^^^^^ Evaluate model on test data ^^^^^^^^^^^^^^')
            y_pred = learner_model.predict(x_features_test_set)
            y_pred = np.argmax(y_pred, axis=1)
            print(y_pred)
            cm = confusion_matrix(y_label_test_set, y_pred)
            plot_cm(y_label_test_set, y_pred, label_A, label_B)
            accuracy = accuracy_score(y_label_test_set, y_pred)
            print(f'Accuracy:{accuracy}')
            performance_evaluation(y_label_test_set, y_pred)

            meta_grads = query_tape.gradient(query_loss, learner_model.trainable_variables)
            meta_optimizer.apply_gradients(zip(meta_grads, meta_model.trainable_variables))

            for i, gradient in enumerate(meta_grads):
                sum_gradients[i] += gradient

        # 3. Meta-update: apply the accumulated gradients to the original model
        cumul = True
        cumul_gradients = [grad / (1.0 if cumul else 5.0) for grad in sum_gradients]
        meta_optimizer.apply_gradients(zip(cumul_gradients, meta_model.trainable_variables))
        total_loss += query_loss.numpy()

        print(f'Total loss:{total_loss}')
        loss_evol = total_loss / (step + 1)
        losses.append(loss_evol)
        if step % meta_epochs == 0:
            print(f'Meta epoch: {step + 1}/{meta_epochs},  Loss: {loss_evol}')

    return meta_model


if __name__ == "__main__":
    data_file = 'part-00002-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv'  # Path to your CSV file
    batch_size = 1
    no_of_ways = 2
    no_of_shots_per_class = 20
    total_classes = 6
    epochs = 120

    # Create a data generator and label mapping
    data_generator = DataGenerator(total_classes, no_of_ways, no_of_shots_per_class, batch_size, data_file)

    # Feature Engineering - XAI
    data_generator.feature_selection()

    # Create a 2D CNN Meta Model Traffic Classifier
    meta_model = create_cnn_model(data_generator.no_of_features)
    meta_optimizer = optimizers.SGD(learning_rate=0.001)
    meta_loss_fn = MeanSquaredError()
    meta_model.compile(optimizer=meta_optimizer, loss=meta_loss_fn, metrics=['accuracy'])
    meta_model.summary()

    print('-------------------------------------------------------------------------------------------------')
    print('                       Part A: Meta Training - Traffic Tasks, T(i = 1, 2, ...,N)                         ')
    print('-------------------------------------------------------------------------------------------------')
    meta_model = maml(meta_model, total_classes, train=True)

    print('-------------------------------------------------------------------------------------------------')
    print('                      Part B: Meta Testing - Traffic Task X (new / unseen traffic)                       ')
    print('-------------------------------------------------------------------------------------------------')
    maml(meta_model, total_classes, train=False)
