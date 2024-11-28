import pandas as pd
import re
import string 
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split


        
class PreProcess:
    def __init__(self, data):
        self.data = data         

    def handle_missing_values(self, method="drop", column=None):
        if column:
            if method == "drop":
                self.data = self.data.dropna(subset=[column])
            elif method == "fill_mean":
                if self.data[column].dtype in ['float64', 'int64']:
                    self.data[column] = self.data[column].fillna(self.data[column].mean())
                else:
                    raise ValueError("Mean filling is only applicable for numerical columns.")
            elif method == "fill_median":
                if self.data[column].dtype in ['float64', 'int64']:
                    self.data[column] = self.data[column].fillna(self.data[column].median())
                else:
                    raise ValueError("Median filling is only applicable for numerical columns.")
            elif method == "fill_mode":
                self.data[column] = self.data[column].fillna(self.data[column].mode()[0])
        return self.data
    
    
    def clean_text(self, text):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        text = text.strip()  # Remove leading/trailing whitespace
        return text  # Return the cleaned text

    def remove_stopwords(self, text):
        stop_words = set(stopwords.words('english'))
        words = text.split()  # Split the text into words
        new_text = [word for word in words if word not in stop_words]
        return ' '.join(new_text)  # Return text without stopwords

    def preprocess(self, text_columns):

        for column in text_columns:
            # Clean text
            print("Cleaning text...")
            self.data[column] = self.data[column].apply(self.clean_text)
            
            # Remove stopwords
            print("Removing stopwords...")
            self.data[column] = self.data[column].apply(self.remove_stopwords)
            
            # Tokenize
            print("Tokenizing text...")
            self.data[column] = self.data[column].apply(word_tokenize)
        
        print("Preprocessing complete for all columns.")
        return self.data
      
    def split_data(self, text_columns, target_column, test_size=0.2, random_state=42):
        X = self.data[text_columns]
        y = self.data[target_column]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
  