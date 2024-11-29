from Data_Preprocess import PreProcess
import numpy as np
import pandas as pd
import joblib


def test_pipline(input_data):
    preprocess = PreProcess(input_data)
    X_train = preprocess.preprocess()
    # return preprocess.data

    model = joblib.load("vector_model.joblib")

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

    pca = joblib.load("pca.joblib")
    X_train = pca.transform(X_train)

    return X_train
