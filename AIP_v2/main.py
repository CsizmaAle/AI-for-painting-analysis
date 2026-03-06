import functions_image_formatting as fif
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, hamming_loss
import joblib
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier


fif.resize_normalize_images(
    input_folder="photos", 
    output_folder="photos4", 
    size=(300, 300)
)

fif.split_dataset(
    input_folder= "photos4",
    output_folder="dataset_split"
)

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

X_test=[]
Y_test=[]

X_train=[]
Y_train=[]

X_test, Y_test= fif.create_vectors("test_data.csv")
X_train, Y_train = fif.create_vectors("train_data.csv")

mlb = MultiLabelBinarizer()

Y_train = mlb.fit_transform(Y_train)
Y_test = mlb.fit_transform(Y_test)
    
# Normalizează datele de intrare
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

data_train = pd.read_csv("train_data.csv")
data_test = pd.read_csv("test_data.csv")

X_train = data_train.iloc[:, 0].values  # Calea imaginilor
Y_train = data_train.iloc[:, 1:].values  # Etichetele (multi-label)

X_test = data_test.iloc[:, 0].values  # Calea imaginilor
Y_test = data_test.iloc[:, 1:].values  # Etichetele (multi-label)

# Creează un model SVM pentru clasificare multi-label
model = SVC(kernel="linear", probability=True)

# Antrenează modelul
model.fit(X_train, Y_train)

# Realizează predicții
Y_pred = model.predict(X_test)

# Exemplu de conversie în binar dacă `Y_pred` este un vector de probabilități
Y_pred = (model.predict_proba(X_test) > 0.5).astype(int)

print("Forma Y_test:", Y_test.shape)
print("Forma Y_pred:", Y_pred.shape)


print("Hamming Loss:", hamming_loss(Y_test, Y_pred))
print("Accuracy:", accuracy_score(Y_test, Y_pred))

# Raport detaliat de clasificare
print("Raport de clasificare:\n", classification_report(Y_test, Y_pred))

# Salvează modelul și scalerul
joblib.dump(model, "multi_label_svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Încărcare model salvat
loaded_model = joblib.load("multi_label_svm_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")


