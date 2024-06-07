import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

np.random.seed(52)


def convert_embarked(x):
    if x == 'S':
        return 0
    elif x == 'C':
        return 1
    else:
        return 2


if __name__ == '__main__':
    data = pd.read_csv('https://www.dropbox.com/s/4vu5j6ahk2j3ypk/titanic_train.csv?dl=1')

    data.drop(
        ['PassengerId', 'Name', 'Ticket', 'Cabin'],
        axis=1,
        inplace=True
    )
    data.dropna(inplace=True)

    # Separate these back
    y = data['Survived'].astype(int)
    X = data.drop('Survived', axis=1)

    X['Sex'] = X['Sex'].apply(lambda x: 0 if x == 'male' else 1)
    X['Embarked'] = X['Embarked'].apply(lambda x: convert_embarked(x))

    X_train, X_val, y_train, y_val = \
        train_test_split(X.values, y.values, stratify=y, train_size=0.8)

    # Create function for bootstrapping
    def create_bootstrap(X_train, y_train):
        mask = np.random.choice(len(X_train), len(y_train), replace=True)
        return X_train[mask], y_train[mask]

    # Create Random Forest class
    class RandomForestClassifier:
        def __init__(self, n_trees=10, max_depth=np.iinfo(np.int64).max, min_error=1e-6):
            self.n_trees = n_trees
            self.max_depth = max_depth
            self.min_error = min_error

            self.forest = []
            self.is_fit = False

        # Bootstrap the training datasets and leveraging DecisionTreeClassifier to fit them into the model
        def fit(self, X_train, y_train):
            for _ in tqdm(range(self.n_trees)):
                bootstrap_X, bootstrap_y = create_bootstrap(X_train, y_train)
                dtc = DecisionTreeClassifier(max_features='sqrt', max_depth=np.iinfo(np.int64).max,
                                             min_impurity_decrease=1e-6)
                dtc.fit(bootstrap_X, bootstrap_y)
                self.forest.append(dtc)
            self.is_fit = True
            return self

        # Create a DataFrame for the forest and predict each element
        def predict(self, X_test):
            if not self.is_fit:
                raise AttributeError('The forest is not fit yet! Consider calling .fit() method.')
            prediction = pd.DataFrame([dtc.predict(X_test) for dtc in self.forest])
            return prediction.mode(axis=0).values[0]

    result = []
    for i in range(1, 21):
        forest = RandomForestClassifier(n_trees=i).fit(X_train, y_train)
        prediction = forest.predict(X_val)
        result.append(round(accuracy_score(y_val, prediction), 3))
    print(result)
