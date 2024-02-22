""" Facial Attractiveness prediction Using OpenFace, Linear Regression, Random Forest """

import torch
import pandas as pd
import numpy as np
import generateFeatures
from sklearn import linear_model, decomposition
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

"""
Before running this python, Generate face landmark from openpose using below command on terminal
"""
## C:\Users\shann\OpenFace_2.2.0_win_x64>FeatureExtraction.exe  -fdir "C:\Users\shann\Workouts (Python & R)\Python Workouts\OpenCV_Bootcamp\sg_demo"

device = "CUDA" if torch.cuda.is_available() else "CPU"
print("Device: ", device)

df = pd.read_csv("C:/Users/shann/OpenFace_2.2.0_win_x64/processed/sg_demo.csv", header=0)

## 68 Face Landmarks locations in 2D
start_column = ' x_0'
end_column = ' y_67'

## Extract data from the specified columns
extracted_data = df.loc[:, start_column:end_column]

## Write the extracted data to the text file with comma-separated values
extracted_data.to_csv('landmarks.txt', sep=',', index=False, header=False)

## Calling python script to generate all possible combination of features
gen_feat = generateFeatures

x = np.loadtxt('features_ALL.txt',delimiter=',')
y = pd.read_csv('label.txt')

## Label Encoding
# one_hot = pd.get_dummies(y)
# print(one_hot)
# y = pd.concat([y,one_hot],axis=1)
encoder = LabelEncoder()
y = encoder.fit_transform(y['Label'])
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.35,shuffle=True,random_state=21)
print(x_train.shape, x_test.shape)

## Reducing the dimensionality
pca = decomposition.PCA(n_components=8)
pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)

## Linear Regression
regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)
prediction = regr.predict(x_test)
prediction = [1 if pred >= 0.5 else 0 for pred in prediction]   # Converting decimal to 0 & 1
print(prediction)
print(y_test)

## Pearson Correlation
corr = np.corrcoef(y_test,prediction)[0,1]
print("Linear Reg_Coeff: ", corr)
## MSE
mse = np.mean((prediction - y_test) ** 2)
print("Linear Reg_MSE: ", mse)

## Random Forest
rf = RandomForestRegressor(n_estimators=20,min_samples_split=2,random_state=21)
rf.fit(x_train, y_train)
prediction = rf.predict(x_test)
prediction = [1 if pred >= 0.5 else 0 for pred in prediction]   # Converting decimal to 0 & 1
print(prediction)
print(y_test)

## Pearson Correlation
corr = np.corrcoef(y_test,prediction)[0,1]
print("RF_Coeff: ", corr)
## MSE
mse = np.mean((prediction - y_test) ** 2)
print("RF_MSE: ", mse)

## __________ End _________ ##