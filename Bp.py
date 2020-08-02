import numpy as np
import math
import random
import string
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn import decomposition
from numpy import*
from sklearn.model_selection import train_test_split

# pandas导入数据
df = pd.read_excel('Airpak_data.xlsx')
a = np.array(df)

# numpy.around(保留两位小数点)
X = np.around(a,decimals=2)
x = X[:,[0,2,4,5]]
y = X[:,[1,3]]

# 随机划分数据集
X_train,X_test,Y_train,Y_test = train_test_split(x, y, test_size=0.1, random_state=0)

# 归一化
m1 = MinMaxScaler()
x_train = m1.fit_transform(X_train)
x_test = m1.fit_transform(X_test)
m2 = MinMaxScaler()
y_train = m2.fit_transform(Y_train)

# random.seed(0)  #设置相同的seed，每次生成的随机数相同。如果不设置seed，则每次会生成不同的随机数
# 生成区间[a,b]内的随机数
def random_number(a, b):
    return (b - a) * random.random() + a # random.random是生成0到1的随机数

# 生成一个矩阵，大小为m*n,并且设置默认零矩阵
def makematrix(m, n, fill=0.0):
    a = []
    for i in range(m):
        a.append([fill] * n)
    return a

# 设tanh为激活函数
def sigmoid(x):
    return math.tanh(x)

# 函数tanh求导
def derived_sigmoid(x):
    return 1.0 - x ** 2

# 构造三层BP网络
class BPNN:
    def __init__(self, num_in, num_hidden, num_out):
        # 输入层，隐藏层，输出层的节点数
        self.num_in = num_in + 1  # 增加一个偏置结点
        self.num_hidden = num_hidden + 1  # 增加一个偏置结点
        self.num_out = num_out

        # 激活神经网络的所有节点
        self.active_in = [1.0] * self.num_in
        self.active_hidden = [1.0] * self.num_hidden
        self.active_out = [1.0] * self.num_out

        # 创建权重矩阵
        self.wight_in = makematrix(self.num_in, self.num_hidden)
        self.wight_out = makematrix(self.num_hidden, self.num_out)

        # 对权值矩阵赋初值
        for i in range(self.num_in):
            for j in range(self.num_hidden):
                self.wight_in[i][j] = random_number(-0.2, 0.2)
        for i in range(self.num_hidden):
            for j in range(self.num_out):
                self.wight_out[i][j] = random_number(-0.2, 0.2)

        # 建立动量因子（矩阵）
        self.ci = makematrix(self.num_in, self.num_hidden)
        self.co = makematrix(self.num_hidden, self.num_out)

     # 信号正向传播
    def update(self, inputs):
        inputs = np.array(inputs)
        if len(inputs) != self.num_in - 1:
            raise ValueError('与输入层节点数不符')

        # 数据输入输入层
        for i in range(self.num_in - 1):
            # self.active_in[i] = sigmoid(inputs[i])  #或者先在输入层进行数据处理
            self.active_in[i] = inputs[i]  # active_in[]是输入数据的矩阵

        # 数据在隐藏层的处理
        for i in range(self.num_hidden - 1):
            sum = 0.0
            for j in range(self.num_in):
                sum = sum + self.active_in[j] * self.wight_in[j][i]
            self.active_hidden[i] = sigmoid(sum)  # active_hidden[]是处理完输入数据之后存储，作为输出层的输入数据

        # 数据在输出层的处理
        for i in range(self.num_out):
            sum = 0.0
            for j in range(self.num_hidden):
                sum = sum + self.active_hidden[j] * self.wight_out[j][i]
            self.active_out[i] = sigmoid(sum)  # 与上同理
        return self.active_out[:]

    # 误差反向传播
    def errorbackpropagate(self, targets, lr, m):  # lr是学习率， m是动量因子
        if len(targets) != self.num_out:
            raise ValueError('与输出层节点数不符！')

        # 首先计算输出层的误差
        out_deltas = [0.0] * self.num_out
        for i in range(self.num_out):
            error = targets[i] - self.active_out[i]
            out_deltas[i] = derived_sigmoid(self.active_out[i]) * error

        # 然后计算隐藏层误差
        hidden_deltas = [0.0] * self.num_hidden
        for i in range(self.num_hidden):
            error = 0.0
            for j in range(self.num_out):
                error = error + out_deltas[j] * self.wight_out[i][j]
            hidden_deltas[i] = derived_sigmoid(self.active_hidden[i]) * error

        # 首先更新输出层权值
        for i in range(self.num_hidden):
            for j in range(self.num_out):
                change = out_deltas[j] * self.active_hidden[i]
                self.wight_out[i][j] = self.wight_out[i][j] + lr * change + m * self.co[i][j]
                self.co[i][j] = change

        # 然后更新输入层权值
        for i in range(self.num_in):
            for j in range(self.num_hidden):
                change = hidden_deltas[j] * self.active_in[i]
                self.wight_in[i][j] = self.wight_in[i][j] + lr * change + m * self.ci[i][j]
                self.ci[i][j] = change

        # 计算总误差
        error = 0.0
        for i in range(len(targets)):
            error = error + 0.5 * (targets[i] - self.active_out[i]) ** 2
        return error

    # 测试
    def test(self, patterns):
        for i in range(len(patterns)):
            inp = patterns[i]
            a = self.update(inp)
            print(i, '->', a)
            if i == 0:
                b = a
            else:
                b = np.row_stack((b,a))
        return b

    #训练
    def train(self,x_train,y_train,itera,lr=0.1, m=0.1):
        for i in range(itera):
            error = 0.0
            for j in range(len(x_train)):
                inputs = x_train[j]
                targets = y_train[j]
                self.update(inputs)
                error = error + self.errorbackpropagate(targets, lr, m)
            if i % 100 == 0:
                print('误差 %-.5f' % error)

# 网络训练与仿真
my_Bp = BPNN(4,50,2)
my_Bp.train(x_train,y_train,200)
x_train = np.array(x_train)
bp_test = my_Bp.test(x_test)
yy_test = m2.inverse_transform(bp_test)
print(yy_test)

# 重组结果矩阵
result_elm = np.hstack((X_test[:,[0]],Y_test[:,[0,1]]))
result_elm = np.hstack((result_elm,yy_test[:,[0,1]]))
a = result_elm[np.lexsort(result_elm[:,::-1].T)]

plt.plot(a[:,0], a[:,1], '.g')
plt.plot(a[:,0], a[:,3], '-r')
plt.title('BP_PMV')
plt.xlabel('temperature')
plt.ylabel('y')
plt.legend([['original'], ['prediction']], loc='upper left')
plt.show()
