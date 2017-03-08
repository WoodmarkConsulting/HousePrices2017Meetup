# coding: utf-8
'''House Price Challenge'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


def main():
    plot = False

    # load input data
    train = pd.read_csv("input/train.csv")
    test = pd.read_csv("input/test.csv")

    print("Number of features in training set: {}".format(train.shape[1]))
    print("Number of training data entries: {}".format(train.shape[0]))
    print("Number of test data entries: {}".format(test.shape[0]))

    print("First column in both sets is: {}".format(train.columns[0]))
    print("Last column in training set is: {}".format(train.columns[-1]))

    # use Id column as DataFrame index
    train.set_index('Id', inplace=True)
    test.set_index('Id', inplace=True)

    # split price from training data
    train_price = train["SalePrice"]
    train.drop("SalePrice", axis=1, inplace=True)

    # boxplot of data
    if plot:
        plt.rcParams['figure.figsize'] = (16, 6)
        train.boxplot(showfliers=False, rot=90)
        plt.show()

    # extract locations of numerical features
    num_feat = (train.dtypes != "object").as_matrix()

    train_num = train.iloc[:, num_feat]
    print("All numerical values in training set positive? {}"
          .format(not (train_num < 0).any().any()))
    # fill missing values in training set with column means
    train_num = train_num.fillna(train_num.mean())
    train.iloc[:, num_feat] = train_num

    test_num = test.iloc[:, num_feat]
    print("All numerical values in test set positive? {}"
          .format(not (test_num < 0).any().any()))
    # fill missing values in test set with TRAINING column means
    test_num = test_num.fillna(train_num.mean())
    test.iloc[:, num_feat] = test_num

    # check out skeweness
    print("Numerical feature columns: {}".format(train.columns[num_feat]))
    print("Skeweness of numerical training features: {}"
          .format(skew(train_num)))

    # log(1+p) transform for skewed features
    skewed = (np.absolute(skew(train_num)) > 1)
    train_num.iloc[:, skewed] = np.log1p(train_num.iloc[:, skewed])
    test_num.iloc[:, skewed] = np.log1p(test_num.iloc[:, skewed])
    train_price = np.log1p(train_price)

    # normalize numerical features
    scaler = StandardScaler().fit(train_num)
    train_num = scaler.transform(train_num)
    test_num = scaler.transform(test_num)

    # apply the transformed values to the orginial sets
    train.iloc[:, num_feat] = train_num
    test.iloc[:, num_feat] = test_num

    print("Skeweness of numerical training features after transformation: {}"
          .format(skew(train_num)))

    # transform categorical to numerical features
    train_test = pd.concat([train, test])
    train_test = pd.get_dummies(train_test)

    # split sets again
    train = train_test.iloc[:train.shape[0], :]
    test = train_test.iloc[train.shape[0]:, :]

    print("Linear Regression")
    lin_reg = linear_model.LinearRegression()
    scores = cross_val_score(lin_reg, train, train_price,
                             cv=5, scoring='neg_mean_squared_error')
    print("Mean of 5 CV sqrt MSE: {}".format(np.sqrt(-scores.mean())))

    print("Ridge Regression")
    ridge = linear_model.Ridge(alpha=10.)
    scores = cross_val_score(ridge, train, train_price,
                             cv=5, scoring='neg_mean_squared_error')
    print("Mean of 5 CV sqrt MSE: {}".format(np.sqrt(-scores.mean())))

    ridge.fit(train, train_price)

    # predict test set
    preds = ridge.predict(test)
    preds_price = np.expm1(preds)
    test_results = pd.DataFrame({'SalePrice': preds_price,
                                 'Id': test.index})
    test_results.set_index('Id', inplace=True)

    # save as csv
    test_results.to_csv("output/test_results.csv")

    return

if __name__ == "__main__":
    main()
