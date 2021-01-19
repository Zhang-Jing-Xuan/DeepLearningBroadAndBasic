import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import csv

os.chdir("/Users/admin/Desktop/CL/Python/DeapLeaning/H1")

'''
读数据：
1.读取数据保存到data中，路径根据你保存的train.csv位置而有变化
2.行保留所有，列从第三列开始往后才保留，这样去除了数据中的时间、地点、参数等信息
3.将所有NR的值全部置为0方便之后处理
4.将data的所有数据转换为二维数据并用raw_data来保存
'''
data = pd.read_csv('train.csv') 
data = data.iloc[:, 3:]  
data[data == 'NR'] = 0  
raw_data = data.to_numpy() 

'''
特征提取1：计算month_data
month_data[i]=[a][b],i代表第i个月(0~11),[a][b]代表24*12个小时的18个属性
先遍历12个月，对每个月分别计算
sample：临时数组，存储每个月的24*12个小时的18个属性信息
再遍历20天，计算sample数组：raw_data每次一次性将18行（24列）的数据赋值给sample的24列（18行）中，重复20次
'''
month_data = {}
for month in range(12): 
    sample = np.empty([18, 480])  
    for day in range(20):  
        sample[:, day * 24: (day + 1) * 24] = raw_data[18 * (20 * month + day): 18 * (20 * month + day + 1),:]  
    month_data[month] = sample

'''
特征提取2:计算x,y
x:一个月一共480个小时，0～9算第0个数据，480为第471个数据（一个月的最后一个数据),数据类型为float
x.shape=(12 * 471, 18 * 9)
y：对应与每一个x的PM2.5浓度，数据类型为float
y.shape=(12 * 471, 1)
依次遍历月（0-11），日（0-19），小时（0-23）
每个月最后一天的小时数据为23-9=14
每次一次性取9列数据并将整个数据reshape成一行数据赋值给x，
每次取第9行的那个数据赋值给y
'''
x = np.empty([12 * 471, 18 * 9],dtype=float)  
y = np.empty([12 * 471, 1], dtype=float)  
for month in range(12):  
    for day in range(20):  
        for hour in range(24):  
            if day == 19 and hour > 14:  
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:, day * 24 + hour: day * 24 + hour + 9].reshape(1,-1) 
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9] 

'''
归一化
np.mean(x, axis=0):求均值，axis = 0表示对各列求均值，返回 1* 列数 的矩阵
np.std(x, axis=0):求标准差，axis = 0表示对各列求标准差，返回 1* 列数 的矩阵
len(x)= x第一维大小 =12 * 471
len(x[0])= x第二维大小 =18 * 9
'''
mean_x = np.mean(x, axis=0)  
std_x = np.std(x, axis=0) 
for i in range(len(x)):  
    for j in range(len(x[0])):  
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

#训练集和验证集，八二分
x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8):, :]
y_validation = y[math.floor(len(y) * 0.8):, :]

'''
模型训练:
dim:用来做参数vector的维数，加1是为了对bias好处理（还有个误差项）。即最后的h(x)=w1x1+w2x2+...+WnXn+b
np.ones([dim, 1]):生成一个dim行1列的数组用来保存参数值，对比源码我这里改成了ones而不是zeros
np.ones([12 * 471, 1])来生成12*471行1列的全1数组，np.concatenate，axis=1表示按列将两个数组拼接起来，即在x最前面新加一列内容，之前x是12*471行18*9列的数组，新加一列之后变为12*471行18*9+1列的数组
学习率=100,迭代次数=10000
adagrad:生成dim行即163行1列的数组，用来使用adagrad算法更新学习率
eps:因为新的学习率是learning_rate/sqrt(sum_of_pre_grads**2),而adagrad=sum_of_grads**2,所以处在分母上而迭代时adagrad可能为0，所以加上一个极小数，使其不除0（1e-10）
np.dot向量点积和矩阵乘法
gradient: 通过矩阵求导算得
adagrad: 累计gradient的平方和
'''
dim = 18 * 9 + 1  
w = np.ones([dim, 1]) 
x = np.concatenate((np.ones([12 * 471, 1]), x), axis=1).astype(float)  
learning_rate = 100
iter_time = 10000
adagrad = np.zeros([dim, 1]) 
eps = 0.0000000001
costs=[]
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y,2)) / 471 / 12)  
    if (t % 100 == 0): 
        costs.append(loss)
        print('After {} epochs, the loss on train data is'.format(t),loss)
    gradient = 2 * np.dot(x.transpose(), np.dot(x,w) - y) 
    adagrad += gradient ** 2 
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)

#绘制损失函数(每一百次迭代一个点)
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per tens)')
plt.title("AdaGrad")
plt.show()

# 将参数保存下来
np.save('weight.npy', w)  

#载入验证集进行验证
w = np.load('weight.npy')

# 使用x_validation和y_validation来计算模型的准确率，因为X已经normalize了，所以不需要再来一遍，只需在x_validation上添加新的一列作为bias的乘数即可
x_validation = np.concatenate((np.ones([1131, 1]), x_validation), axis=1).astype(float)
ans_y = np.dot(x_validation, w)
loss = np.sqrt(np.sum(np.power(ans_y - y_validation, 2)) / 1131)
print("The loss on validation data is",loss)

'''预测testdata得到预测结果
1.取csv文件中的全行数即第3列到结束的列数所包含的数据保存到test_data
2.将testdata中的NR替换为0
3.将其转换为数组,数据类型为浮点格式
4.创建一个240行18*9列的空数列保存testdata的输入
5.跟training data是同一种方法进行Normalize
6.在test_x前面拼接一列全1数组，构成240行，163列数据
7.加载之前算得得参数进行预测
'''
testdata = pd.read_csv('test.csv', header=None)
test_data = testdata.iloc[:, 2:]  
test_data[test_data == 'NR'] = 0  
test_data = test_data.to_numpy()  
test_x = np.empty([240, 18 * 9], dtype=float)  
for i in range(240):  # 共240个测试输入数据
    test_x[i, :] = test_data[18 * i: 18 * (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis=1).astype(float) 
w = np.load('weight.npy')
ans_y = np.dot(test_x, w)

#将结果写入submit.csv
with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)
