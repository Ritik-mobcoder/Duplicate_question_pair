import gensim
from gensim.utils import simple_preprocess
from nltk import sent_tokenize

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import joblib


class Vectorizer:

    def __init__(self, train, test):
        self.train = train
        self.test = test

    def create_story(self):
        story = []
        for column in self.train:
            for word in self.train[column]:
                raw_sent = sent_tokenize(word)
                for sent in raw_sent:
                    story.append(simple_preprocess(sent))
        for column in self.test:
            for word in self.test[column]:
                raw_sent = sent_tokenize(word)
                for sent in raw_sent:
                    story.append(simple_preprocess(sent))
            return story

    def create_model(self):
        model = gensim.models.Word2Vec(window=4, min_count=2, workers=4)
        return model

    def vectorization(self):
        print("Vectorizing data...")
        print("Train data before vectorization:", self.train)
        print("Test data before vectorization:", self.test)
        story = self.create_story()
        self.model = self.create_model()
        self.model.build_vocab(story)
        self.model.train(story, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        return self.model
    
    def save_model(self):
        joblib.dump(self.model, "vector_model.joblib")
        

    # def vectorize(X_train, max_features=3000):
    #     print("Checking data format:")
    #     print(X_train.head())

    #     if "question1" not in X_train or "question2" not in X_train:
    #         raise KeyError("X_train must contain 'question1' and 'question2' columns")
    #     cv = CountVectorizer(max_features=max_features)

    #     print("Vectorizing 'question1'")
    #     q1_train_arr = cv.fit_transform(X_train["question1"].astype(str)).toarray()

    #     print("Vectorizing 'question2'")
    #     q2_train_arr = cv.transform(X_train["question2"].astype(str)).toarray()
    #     temp1 = pd.DataFrame(q1_train_arr)
    #     temp2 = pd.DataFrame(q2_train_arr)
    #     train_idf = pd.concat([temp1, temp2], axis=1)

    #     return train_idf
