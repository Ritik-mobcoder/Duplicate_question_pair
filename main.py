from Load_data import DataLoader
from Data_Preprocess import PreProcess
from Vectorization import Vectorizer
from Split_data import Split
from Feature_engineering import FeatureManipulator
import numpy as np
import pandas as pd
from Model import Random_Model
from sklearn.metrics import accuracy_score


# Load the data
file_path = "questions.csv"
loader = DataLoader(file_path)
data = loader.load_data()
data = data.iloc[:80000, :]

print("Preview of raw data:")
print(loader.get_preview())
print("Data shape:", loader.get_shape())

text_columns = ["question1", "question2"]
target_column = "is_duplicate"
split = Split(data)
X, y = split.split_data(text_columns, target_column)
# print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

preprocessor = PreProcess(X)

# Preprocess the data
print("\nPreprocessing the data...")

cleaned_data = preprocessor.preprocess()

df = cleaned_data

X_train, X_test, y_train, y_test = split.tesin_test_split(
    df, y, random_state=42, train_size=0.8
)

# Vectorization
vec = Vectorizer(X_train, X_test)
model = vec.vectorization()


def document_vector(data):
    doc = [word for word in data.split() if word in model.wv.index_to_key]
    if not doc:
        return np.zeros(model.vector_size)
    return np.mean(model.wv[doc], axis=0)


X_train_combine = []
for column in X_train.columns:
    for doc in X_train[column].values:
        X_train_combine.append(document_vector(doc))
train = np.array(X_train_combine)

data_q1, data_q2 = np.vsplit(train, 2)

temp1 = pd.DataFrame(data_q1)
temp2 = pd.DataFrame(data_q2)
X_train = pd.concat([temp1, temp2], axis=1)
print(X_train)

X_test_combine = []
for column in X_test.columns:
    for doc in X_test[column].values:
        X_test_combine.append(document_vector(doc))
test = np.array(X_test_combine)

data_q1, data_q2 = np.vsplit(test, 2)

temp1 = pd.DataFrame(data_q1)
temp2 = pd.DataFrame(data_q2)
X_test = pd.concat([temp1, temp2], axis=1)
print(X_test)
# vec.save_model()

print("Vectorization")

ext = FeatureManipulator(X_train, X_test)
X_train, X_test = ext.extraction()
ext.save()
print("PCA applied successfully")

model = Random_Model(X_train, y_train)
model.rand_forest_clf()
model.fit()
model.save()
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
