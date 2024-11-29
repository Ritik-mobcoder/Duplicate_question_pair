import pandas as pd
import re
import string
from textblob import TextBlob
from nltk.corpus import stopwords


class PreProcess:
    def __init__(self, data):
        self.data = data

    def handle_missing_values(self, method="drop", column=None):
        if column:
            if method == "drop":
                self.data = self.data.dropna(subset=[column])
            elif method == "fill_mean":
                if self.data[column].dtype in ["float64", "int64"]:
                    self.data[column] = self.data[column].fillna(
                        self.data[column].mean()
                    )
                else:
                    raise ValueError(
                        "Mean filling is only applicable for numerical columns."
                    )
            elif method == "fill_median":
                if self.data[column].dtype in ["float64", "int64"]:
                    self.data[column] = self.data[column].fillna(
                        self.data[column].median()
                    )
                else:
                    raise ValueError(
                        "Median filling is only applicable for numerical columns."
                    )
            elif method == "fill_mode":
                self.data[column] = self.data[column].fillna(
                    self.data[column].mode()[0]
                )
        return self.data

    def clean_text(self):
        data = self.data.str.lower()
        # Convert to lowercase
        data = re.sub(r"<.*?>", "", data)  # Remove HTML tags
        data = re.sub(
            r"http\S+|www\S+|https\S+", "", data, flags=re.MULTILINE
        )  # Remove URLs
        data = data.translate(
            str.maketrans("", "", string.punctuation)
        )  # Remove punctuation
        data = data.strip()  # Remove leading/trailing whitespace
        return data  # Return the cleaned text

    def remove_stopwords(self):
        data = self.data
        stop_words = set(stopwords.words("english"))
        words = data.split()  # Split the text into words
        new_text = [word for word in words if word not in stop_words]
        return " ".join(new_text)  # Return text without stopwords

    def preprocess(self):
        # Clean text
        print("Cleaning text...")
        self.data = self.clean_text()

        # Remove stopwords
        print("Removing stopwords...")

        self.data = self.remove_stopwords()

        # Tokenize
        print("Tokenizing text...")
        # self.data[column] = self.data[column].apply(sent_tokenize)

        return self.data
