# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages required.
2. Read the dataset.
3. Define X and Y array.
4. Define a function for sigmoid, loss, gradient and predict and perform operations. 

## Program:

/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Sri hari R
RegisterNumber: 212223040202
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Placement_Data.csv")
dataset
dataset=dataset.drop("sl_no",axis=1)
dataset=dataset.drop("salary",axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes



dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes

dataset

X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values

Y

theta=np.random.randn(X.shape[1])
y=Y

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1-y) * np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta
    
theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
    
y_pred = predict(theta,X)

accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:", accuracy)

print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)


## Output:
### Read the file and display
![WhatsApp Image 2024-05-09 at 01 00 12_5d89b2e8](https://github.com/srrihaari/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145550674/dc5a626d-dafe-44e4-8917-58065ccffa8a)


### Categorizing columns
![WhatsApp Image 2024-05-09 at 01 00 21_f02c5876](https://github.com/srrihaari/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145550674/62a36609-b588-4b59-8033-12288c148076)


### Labelling columns and displaying dataset
![WhatsApp Image 2024-05-09 at 01 00 27_d39d0b23](https://github.com/srrihaari/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145550674/63c2bc1e-52ba-467d-9648-4640a0065059)


### Display dependent variable
![WhatsApp Image 2024-05-09 at 01 00 30_8dc9ce2b](https://github.com/srrihaari/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145550674/bebd3c69-28e0-46f1-9316-88df8cdf8994)

### Printing accuracy
![WhatsApp Image 2024-05-09 at 01 00 35_8b789c39](https://github.com/srrihaari/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145550674/de95682e-01b4-4442-9e90-89fb9bccbe61)


### Printing Y
![WhatsApp Image 2024-05-09 at 01 00 38_6b288e71](https://github.com/srrihaari/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145550674/6f029530-bd7d-4424-b695-63280a1aec56)



### Printing y_prednew
![WhatsApp Image 2024-05-09 at 01 00 43_2bf1ffea](https://github.com/srrihaari/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145550674/a5aae2b5-9985-48e4-93d4-6213be1c88d7)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
