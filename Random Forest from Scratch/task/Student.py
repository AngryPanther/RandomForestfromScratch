import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

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

    def create_bootstrap(X, y):
        bootstrap_X = X[np.random.choice(len(X), size=(len(X)))]
        bootstrap_y = y[np.random.choice(len(X), size=(len(y)))]
        return bootstrap_X, bootstrap_y

    class RandomForestClassifier:
        def __init__(self, n_trees=600, max_depth=np.iinfo(np.int64).max, min_error=1e-6):
            self.n_trees = n_trees
            self.max_depth = max_depth
            self.min_error = min_error

            self.forest = []
            self.is_fit = False

            self.clf = DecisionTreeClassifier(max_features='sqrt', max_depth=self.max_depth,
                                         min_impurity_decrease=self.min_error)

        def fit(self, X_train, y_train):
            for _ in tqdm(range(self.n_trees)):
                bootstrap_X, bootstrap_y = create_bootstrap(X_train, y_train)
                self.clf.fit(bootstrap_X, bootstrap_y)
                self.clf.fit(X_train, y_train)
                self.forest.append(self.clf)
            self.is_fit = True

        def predict(self, X_test):
            if not self.is_fit:
                raise AttributeError('The forest is not fit yet! Consider calling .fit() method.')
            prediction = self.clf.predict(X_test)
            return prediction

    result = []
    for n_trees in range(1, 21):
        attempt = RandomForestClassifier(n_trees=n_trees)
        attempt.fit(X_train, y_train)
        result.append(round(accuracy_score(y_val, attempt.predict(X_val)), 3))
    print(result[:20])

    plt.title("Random Forest Accuracy Score")
    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy")
    plt.plot(range(1, 21), result[:20])
    plt.show()
