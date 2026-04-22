import functions_image_formatting as fif
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, hamming_loss
import joblib
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier

#!!!ASTA E CODUL BUN !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Step 1: Resize and normalize images
fif.resize_normalize_images(
    input_folder="photos", 
    output_folder="photos4", 
    size=(300, 300)
)

# Step 2: Split dataset into train and test
fif.split_dataset(
    input_folder="photos4",
    output_folder="dataset_split"
)

# Step 3: Create CSV files for the split dataset
fif.create_csv_split_dataset(
    input_folder="dataset_split/test",
    csv_input_file="metadata.csv",
    csv_output_file="test_data.csv"
)

fif.create_csv_split_dataset(
    input_folder="dataset_split/train",
    csv_input_file="metadata.csv",
    csv_output_file="train_data.csv"
)

# Step 4: Load the vectors (image features) and labels
X_test, Y_test = fif.create_vectors("test_data.csv")
X_train, Y_train = fif.create_vectors("train_data.csv")

# Step 5: Transform multi-labels using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
Y_train = mlb.fit_transform(Y_train)  # Fit and transform on training labels
Y_test = mlb.transform(Y_test)  # Transform test labels using the same mapping

# Step 6: Normalize image data (flatten and scale)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))  # Flatten images for scaling
X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1))  # Flatten images for scaling

# Step 7: Initialize and train the SVM model (OneVsRestClassifier for multi-label)
model = OneVsRestClassifier(SVC(kernel="linear", probability=True))

# Train the model
model.fit(X_train, Y_train)

# Step 8: Make predictions
Y_pred = model.predict(X_test)

# Step 9: Evaluate the model
print("Forma Y_test:", Y_test.shape)
print("Forma Y_pred:", Y_pred.shape)

# Convert the labels from MultiLabelBinarizer to strings for target_names
target_names = [str(label) for label in mlb.classes_]

# Handle undefined metric warning by using zero_division parameter
print("Hamming Loss:", hamming_loss(Y_test, Y_pred))
print("Accuracy:", accuracy_score(Y_test, Y_pred))

# Classification report with zero_division parameter set to 0 (or 1 depending on your preference)
print("Raport de clasificare:\n", classification_report(Y_test, Y_pred, target_names=target_names, zero_division=0))

# Step 10: Save the trained model and scaler
joblib.dump(model, "multi_label_svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Step 11: Load the saved model and scaler
loaded_model = joblib.load("multi_label_svm_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")
