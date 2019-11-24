import random
import operator 
import math
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("Andhra_dataset2.csv")
#len(df)
df.isnull().sum()
df.drop(['education'], axis = 1) 

#print(df)
def mean_fill(df):
    for col in df.columns: 
        mean = df[col].mean() #imputing item_weight with mean
        df[col].fillna(mean, inplace =True)
   # print(df)
    
#fill with mode
def mode_fill(df):
    for col in df.columns: 
        mode = df[col].mode() #imputing outlet size with mode
        df[col].fillna(mode[0], inplace =True)
   # print(df)
    
mode_fill(df)
#mean_fill(df)
df.drop(['education'], axis=1, inplace=True)
df = pd.get_dummies(df)
#print(df)
df=df.values.tolist()
#print(df)
def minmax(df):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*df)]
    return stats
mm=minmax(df)

def normalize(df, minmax):
    for row in df:
        for i in range(len(row)-1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

normalize(df,mm)
#print(df)

for i in range(len(df)):
    df[i][8]=int((df[i][8]))

def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup
str_column_to_int(df, len(df[0])-1)
from sklearn.model_selection import train_test_split
train , test = train_test_split(df, test_size = 0.33)

def Euclideandist(x,y, length):
    d = 0.0
    for i in range(length):
        d += pow(float(x[i])- float(y[i]),2)
    return math.sqrt(d)


#Getting the K neighbours having the closest
#Euclidean distance to the test instance
def k_nearest(train, test, k):
    dis = []
    l = len(test)-1
    for x in range(len(train)):
        dist = Euclideandist(test, train[x], l)
        dis.append((train[x], dist))
    dis.sort(key=operator.itemgetter(1))
    knn = []
    for x in range(k):
        knn.append(dis[x][0])
    return knn

#After sorting the neighbours based on their respective classes,
#max voting to give the final class of the test instance
def Results(knn):
    abc={}
    for x in range(len(knn)):
        result=knn[x][-1]
        if result in abc:
            abc[result]+=1
        else:
            abc[result]=1
    sortedval=sorted(abc.items(),key=operator.itemgetter(1),reverse=True)
    return sortedval[0][0]

def Accuracy(test, predictions):
    correct = 0
    for x in range(len(test)):
        if test[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(test))) * 100.0

predictions=[]
k=9
fp=0
fn=0
tp=0
tn=0
predicted_out1=[]
actual_out1=[]
print(len(df[1]))
for x in range(len(test)):
    neigh = k_nearest(train, test[x], k)
    result = Results(neigh)
    predictions.append(result)
    predicted_out1.append(result)
    actual_out1.append(test[x][-1])
    #print('> predicted='+ (result) + ', actual=' + repr(test[x][-1]))
print("K-Nearest Neighbours:")
print("K value:",k)
print(actual_out1)
print(predicted_out1)
x=[]
'''
for i in df:
    l=[]
    l.append(i[-1])
    x.append(l)
print(x)
'''
cm=confusion_matrix(actual_out1,predicted_out1)
print("Confusion Matrix:")
print(cm)
accuracy = Accuracy(test, predictions)
print('Accuracy: ' + repr(accuracy) + '%')



for i in range(len(df)):
    df[i][8]=int((df[i][8]))

def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup
str_column_to_int(df, len(df[0])-1)
from sklearn.model_selection import train_test_split
train , test = train_test_split(df, test_size = 0.33)


class Neural_Network(object):
    def __init__(self, W1=None, W2= None):
        #define Parameters
        self.inputLayerSize = 8
        self.outputLayerSize = 1
        self.hiddenLayerSize = 10
        
        #weights
        if(W1 is None and W2 is None):
            self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
            self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        else:
            self.W1=W1
            self.W2=W2
            
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def relu(self, Z):
        return np.maximum(0,Z)

    def der_relu(self, dA, Z):
        dZ = np.array(dA, copy = True)
        dZ[Z <= 0] = 0;
        return dZ;
    
    def forward(self, X):
        #progate through the network
        self.z2 = np.dot(X,self.W1)
        self.a2 = self.tanh(self.z2)
        self.z3 = np.dot(self.a2,self.W2)
        y_hat = self.sigmoid(self.z3)
        return y_hat
    
    def costFunction(self, X, y):
        self.y_hat=self.forward(X)
        J= 0.5*sum((y-self.y_hat)**2)
        return J
    
    def der_sigmoid(self, z):
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def der_costFunction(self, X, y):
        self.y_hat = self.forward(X)
        delta3 = np.multiply(-(y-self.y_hat),self.der_tanh(self.z3))
        djdW2 = np.dot(self.a2.T,delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.der_sigmoid(self.z2)
        djdW1 = np.dot(X.T, delta2)
        return djdW1, djdW2
    
    def tanh(self,X):
        return np.tanh(X)
    
    def der_tanh(self,X):
        return 1.0- np.tanh(X)**2
#print(df)
X=[]
for i in train:
    X.append(i[:-1])
#print(len(X[0]))
#print(X)
y=[]
for i in train:
    y.append(i[-1])
#print(y)
X=np.asarray(X)
y=np.asarray(y)
y=np.reshape(y,(len(y),1))
#print(X.shape)
#print(y.shape)
NN=Neural_Network()
max_iter = 100000
iter = 0
learningRate = 0.02
while(iter<max_iter):
    djdW1, djdW2 = NN.der_costFunction(X,y)
    NN.W1 = NN.W1 -learningRate*djdW1
    NN.W2 = NN.W2 -learningRate*djdW2
    #print(NN.W1,NN.W2)
    if(iter%10000 == 0):
        #print(NN.costFunction(X,y))
        NN.costFunction(X,y)
    iter=iter+1
#print(y)
#print(NN.forward(X))
def sse(y,yhat):
    #print(yhat)
    #print(y)
    l=[]
    for i,j in zip(y,yhat):
        l.append(pow((i-j),2))
    x=sum(l)
    x=x/10000
    return x

    #return (sum(map(lambda a,b : pow(a[0]-b[0],2),zip(y,yhat)))/10000)

def sigm(z):
    return 1/(1 + np.exp(-z)) 

#print(sse(y,NN.forward(X)))
bestW1=[]
bestW2=[]
min=10000
for i in range(1,100):
    NN=Neural_Network()
    if min>sse(y,NN.forward(X)):
        min=sse(y,NN.forward(X))
        bestW1=NN.W1
        bestW2=NN.W2
        
#Best Weights
#print(bestW1)
#print(bestW2)

X=[]
for i in test:
    X.append(i[:-1])
#print(len(X[0]))
#print(X)
y=[]
for i in test:
    y.append(i[-1])
X=np.asarray(X)
y=np.asarray(y)
y=np.reshape(y,(len(y),1))
#print(X.shape)
#print(y.shape)

Best_NN=Neural_Network(bestW1,bestW2)
y_hat=Best_NN.forward(X)
def Accuracy(test, predictions):
    correct = 0
    for i in range(len(test)):
        if test[i][0] == round(predictions[i][0]):
            correct += 1
    return (correct/float(len(test))) * 100.0
#print(y_hat)
predicted_out2=[]
for i in y_hat:
    for j in i:
        predicted_out2.append(int(round(j)))
#print(predicted_out1)
#print(predicted_out2)
print("Neural Net:")
print("Actual output-",actual_out1)
print("Predicted output-",predicted_out2)
cm=confusion_matrix(actual_out1,predicted_out2)
print("Confusion Matrix:")
print(cm)
print("Accuracy=",Accuracy(y,y_hat))
