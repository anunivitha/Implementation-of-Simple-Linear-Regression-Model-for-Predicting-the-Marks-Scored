# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries : Load libraries like pandas, numpy, matplotlib, and modules from sklearn.

2. Load the dataset : Read the dataset (studentscores.csv) using pandas.read_csv().

3. Display the first few rows for verification. : Visualize the dataset

4. Create a scatter plot of the given data points (Hours vs. Marks). : Define input and output variables

5. Assign the independent variable (X: Hours studied). : Assign the dependent variable (Y: Marks scored).

6. Split the dataset : Use train_test_split() to split the dataset into training data and testing data (e.g., 80% training, 20% testing).

7. Train the model : Create an instance of the LinearRegression model.

8. Fit the model with training data (X_train, Y_train). : Make predictions

9. Predict marks for given input values using the trained model. : Test with unseen test data (X_test) to check accuracy.

10. Plot the regression line : Plot the training points as a scatter plot.

11. Output the results : Display the predicted values, regression line, slope, and intercept.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Anu Nivitha U
RegisterNumber: 212223040016
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('/content/studentscores.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
x
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
X_train
Y_train
lr.predict(X_test.iloc[0].values.reshape(1,1))
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='red')
m=lr.coef_
m[0]
b=lr.intercept_
b
```

## Output:
<img width="375" height="414" alt="Screenshot 2025-10-04 at 10 00 37 AM" src="https://github.com/user-attachments/assets/d19ec2f6-1ad3-4f82-88ba-67272ffde349" />
<img width="600" height="502" alt="Screenshot 2025-10-04 at 10 00 48 AM" src="https://github.com/user-attachments/assets/b0fff9cd-392d-4569-b2b5-f3aa2af865be" />
<img width="259" height="408" alt="Screenshot 2025-10-04 at 10 00 56 AM" src="https://github.com/user-attachments/assets/8d60670c-5263-420c-be04-817ef55a8b04" />
<img width="233" height="116" alt="Screenshot 2025-10-04 at 10 01 04 AM" src="https://github.com/user-attachments/assets/40a62ec4-5836-434f-b237-6a96bd5fdcad" />
<img width="215" height="216" alt="Screenshot 2025-10-04 at 10 01 10 AM" src="https://github.com/user-attachments/assets/5fdb4642-2328-4ef9-b194-35ae12260a7d" />
<img width="835" height="67" alt="Screenshot 2025-10-04 at 10 01 17 AM" src="https://github.com/user-attachments/assets/ca699109-7eb0-4109-aa44-d8b46bc8baa0" />
<img width="638" height="525" alt="Screenshot 2025-10-04 at 10 04 20 AM" src="https://github.com/user-attachments/assets/1bfb58d1-6437-4c78-a655-7caf1918c08a" />
<img width="233" height="176" alt="Screenshot 2025-10-04 at 10 04 29 AM" src="https://github.com/user-attachments/assets/2e5ac3ab-9e4b-4c37-a04d-a1ec4c9f2803" />










## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
