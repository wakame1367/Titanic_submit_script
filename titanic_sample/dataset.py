from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from titanic_sample.utils import ON_KAGGLE

DATA_ROOT = Path('../input/titanic' if ON_KAGGLE else './resources')
Categorical_Features = ['Embarked', 'Pclass', 'Sex']
train_path = DATA_ROOT / "train.csv"
test_path = DATA_ROOT / "test.csv"
submit_path = DATA_ROOT / "gender_submission.csv"


def preprocess(data):
    data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
    data['Embarked'].fillna('S', inplace=True)
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(
        int)
    data['Fare'].fillna(np.mean(data['Fare']), inplace=True)
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['FamilySize'] = data['Parch'] + data['SibSp'] + 1
    data['IsAlone'] = 0
    data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1
    return data


def load_submit():
    return pd.read_csv(submit_path)


def load_dataset(test_size=0.3):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    target_col = "Survived"
    data = pd.concat([train, test], sort=False)
    data = preprocess(data)
    delete_columns = ['Name', 'PassengerId', 'Ticket', 'Cabin']
    data.drop(delete_columns, axis=1, inplace=True)

    train = data[:len(train)]
    test = data[len(train):]

    y_train = train[target_col]
    x_train = train.drop(target_col, axis=1)
    x_test = test.drop(target_col, axis=1)

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                          test_size=test_size,
                                                          random_state=0,
                                                          stratify=y_train)
    return (x_train, x_valid, x_test), (y_train, y_valid)
