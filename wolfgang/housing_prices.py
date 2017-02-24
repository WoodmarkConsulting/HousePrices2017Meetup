"""Experimental Kaggle challenge script"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew


def main():
    """Experimental Kaggle challenge script"""
    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")

    print("Number of features: {}".format(train.shape[1]))
    print("Training data size: {}".format(train.shape[0]))
    print("Test data size: {}".format(test.shape[0]))
    print(train.columns)

    train.iloc[:,:-1].boxplot(showfliers=False, rot=90)
    plt.show()

    # print(train.describe(train.describe()["max"])

    # print(any(train["SalePrice"].isnull()))
    print("Skewness of house price distribution: {:.2f}"
          .format(skew(train["SalePrice"])))
    # train["SalePrice"].hist()
    np.log1p(train["SalePrice"]).hist()
    plt.show()

    print("Skewness of house price distribution after log1p trafo: {:.2f}"
          .format(skew(np.log1p(train["SalePrice"]))))

    num_feat = (train.dtypes != "object").as_matrix()

    train_num = train.iloc[:, num_feat]
    train_num = train_num.fillna(train_num.mean())
    train.iloc[:, num_feat] = train_num

    # important: only allowed to use training mean
    test_num = test.iloc[:, num_feat[:-1]]
    test_num = test_num.fillna(train_num.mean())
    test.iloc[:, num_feat[:-1]] = test_num

    train_test = pd.concat([train, test])
    # print(skew(train_test.iloc[:, num_feat]))

    train_test = pd.get_dummies(train_test)

    train = train_test.iloc[0:train.shape[0], :]
    test = train_test.iloc[train.shape[0]:, :]


if __name__ == "__main__":
    main()
