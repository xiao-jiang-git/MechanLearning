import sys
import numpy as np
import pandas as pd

data = pd.read_csv('./train.csv', encoding = 'big5')
print (data)

data = data.iloc[:, 3:]  # 抽取数据，抽取的是包含所有行，第三列往后的数据， 也就是把前面的文字数据给去除了
data[data == 'NR'] = 0
print(data)
raw_data = data.to_numpy()
print (raw_data)

# 将数据划分成每个月 18个feature * 20天每小时的480份数据
month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
    month_data[month] = sample

# 将数据
x = np.empty([12 * 471, 18 * 9], dtype = float)
y = np.empty([12 * 471, 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]  #value #提取PM2，5
print(x)
print(y)

#进行标准化处理
mean_x = np.mean(x, axis = 0)  # 18 * 9
std_x = np.std(x, axis = 0)  # 18 * 9
for i in range(len(x)):  # 12 * 471
    for j in range(len(x[0])):  # 18 * 9
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

# adagrad 动态调整学习率
# y = b + wx
dim = 18 * 9 + 1
w = np.zeros([dim, 1])

# 按照y轴将np.array组成一个新的array
x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float)

learning_rate = 100

iter_time = 20000

adagrad = np.zeros([dim, 1])

# 防止分母为0
eps = 0.0000000001

for t in range(iter_time):

    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/471/12)  # rmse（均方根误差） 衡量观测值与真实值的误差

    if (t%100==0):
        print(str(t) + ":" + str(loss))

    # 计算梯度
    gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y) #dim*1
    adagrad += gradient ** 2  # 梯度累积变量

    # 应用参数更新
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)

np.save('weight.npy', w)



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
print(test_x)

# 预测
w = np.load('weight.npy')
ans_y = np.dot(test_x, w)
print('预测PM2.5值')
print(ans_y)