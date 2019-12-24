import argparse

import lightgbm as lgb

from titanic_sample.dataset import DATA_ROOT, load_dataset

train_path = DATA_ROOT / "train.csv"
test_path = DATA_ROOT / "test.csv"


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--test_size", type=float, default=0.3)
    args = parser.parse_args()
    return args


def train(x_train, y_train, x_valid, y_valid, categorical_features):
    lgb_train = lgb.Dataset(x_train, y_train,
                            categorical_feature=categorical_features)
    lgb_eval = lgb.Dataset(x_valid, y_valid, reference=lgb_train,
                           categorical_feature=categorical_features)

    params = {
        'objective': 'binary'
    }

    model = lgb.train(
        params, lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        verbose_eval=10,
        num_boost_round=1000,
        early_stopping_rounds=10
    )
    return model


def main():
    args = get_arguments()
    test_size = args.test_size
    (x_train, x_valid, x_test), (y_train, y_valid) = load_dataset(test_size)
    categorical_features = ['Embarked', 'Pclass', 'Sex']
    model = train(x_train=x_train, y_train=y_train,
                  x_valid=x_valid, y_valid=y_valid,
                  categorical_features=categorical_features)
    model.save_model(args.model_path)


if __name__ == '__main__':
    main()
