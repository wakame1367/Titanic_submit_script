import argparse

import lightgbm as lgb

from titanic_sample.dataset import load_dataset, load_submit


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("output_submit_path")
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    (_, _, x_test), (_, _) = load_dataset()
    sub = load_submit()
    model = lgb.Booster(model_file=args.model_path)
    y_pred = model.predict(x_test)
    y_pred = (y_pred > 0.5).astype(int)
    sub['Survived'] = y_pred
    sub.to_csv(args.output_submit_path, index=False)


if __name__ == '__main__':
    main()
