import sys
import numpy as np
import pandas as pd

testdata = pd.read_csv('./test.csv', header = None, encoding = 'big5')

test_data = testdata.iloc[:, 2:]

test_data[test_data == 'NR'] = 0

test_data = test_data.to_numpy()

test_x = np.empty([240, 18*9], dtype = float)



for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)

test_x