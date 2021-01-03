import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

#一张训练图片的例子
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
index = 100
plt.imshow(train_set_x_orig[index])
plt.show()
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

#训练图片和测试图片的大小
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

"""
接下来，我们需要对将每幅图像转为一个矢量，即矩阵的一列。
最终，整个训练集将会转为一个矩阵，其中包括num_px*numpy*3行，m_train列。
Ps：其中X_flatten = X.reshape(X.shape[0], -1).T可以将一个维度为(a,b,c,d)的矩阵转换为一个维度为(b∗c∗d, a)的矩阵。
"""
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

"""
接下来，我们需要对图像值进行归一化。
由于图像的原始值在0到255之间，最简单的方式是直接除以255即可。
"""
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.


def sigmoid(x):
    s=1.0/(1+np.exp(-x))
    return s

def initialize_with_zeros(dim):
    w=np.zeros((dim,1))
    b=0
    return w,b

def propagate(w,b,X,Y):
    m = X.shape[1] #样本个数
    # FORWARD PROPAGATION (FROM X TO COST) 前向传播
    A = sigmoid(np.dot(w.T, X) + b)# compute activation
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) #成本函数     Y ==Y_hat   
    # BACKWARD PROPAGATION (TO FIND GRAD) 反向传播
    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)
    cost = np.squeeze(cost) #压缩维度    
    grads = {"dw": dw,
             "db": db} #梯度
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    for i in range(num_iterations): #每次迭代循环一次， num_iterations为迭代次数
        # Cost and gradient calculation 
        grads, cost = propagate(w, b, X, Y)
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        # update rule 
        w = w - learning_rate * dw
        b = b - learning_rate * db
        # Record the costs
        if i % 100 == 0:  #每100次记录一次成本值
            costs.append(cost)
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:   #打印成本值
            print ("Cost after iteration %i: %f" %(i, cost))
    params = {"w": w,
              "b": b}  #最终参数值
    grads = {"dw": dw,
             "db": db} #最终梯度值
    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1] #样本个数
    Y_prediction = np.zeros((1,m)) #初始化预测输出
    w = w.reshape(X.shape[0], 1) #转置参数向量w
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)  #最终得到的参数代入方程
    for i in range(A.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0][i] > 0.5:
            Y_prediction[0][i] = 1
        else:
            Y_prediction[0][i] = 0
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    # initialize parameters with zeros 
    w, b = initialize_with_zeros(X_train.shape[0])
    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost) #初始化参数w，b
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]  #梯度下降找到最优参数
    b = parameters["b"]
    # Predict test/train set examples 
    Y_prediction_test = predict(w, b, X_test) #训练集的预测结果
    Y_prediction_train = predict(w, b, X_train) #测试集的预测结果
    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))#训练集识别准确度
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100)) #测试集识别准确度
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    return d
#train model
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

#test
index = 8
plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
plt.show()
print(test_set_y[0,index])
print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + 
       classes[int(d["Y_prediction_test"][0,index])].decode("utf-8") +  "\" picture.")

# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

"""
学习速率对于最终的结果有着较大影响
分析：不同的学习速率会导致不同的预测结果。较小的学习速度收敛速度较慢，而过大的学习速度可能导致震荡或无法收敛。
"""
# learning_rates = [0.01, 0.001, 0.0001]
# models = {}
# for i in learning_rates:
#     print ("learning rate is: " + str(i))
#     models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
#     print ('\n' + "-------------------------------------------------------" + '\n')

# for i in learning_rates:
#     plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

# plt.ylabel('cost')
# plt.xlabel('iterations')

# legend = plt.legend(loc='upper center', shadow=True)
# frame = legend.get_frame()
# frame.set_facecolor('0.90')
# plt.show()

"""
用一副你自己的图像，而不是训练集或测试集中的图像
"""
## START CODE HERE ## (PUT YOUR IMAGE NAME) 
my_image = "cat.jpg"   # change this to the name of your image file 
## END CODE HERE ##

# We preprocess the image to fit your algorithm.
fname = "/Users/admin/Desktop/CL/Python/DeapLeaning/L1W2H2/datasets/" + my_image
image = np.array(plt.imread(fname))  #读取图片
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T #放缩图像
my_predicted_image = predict(d["w"], d["b"], my_image)  #预测

plt.imshow(image)
plt.show()
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" +
      classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")