from sklearn.model_selection import train_test_split

class Split:
    def __init__(self, data):
        self.data = data

    def split_data(self, text_columns, target_column):
        X = self.data[text_columns]
        y = self.data[target_column]
        return X, y
    
    def tesin_test_split(self, X, y, random_state=42, train_size=0.8):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=random_state, train_size=train_size
        )
        return X_train, X_test, y_train, y_test

