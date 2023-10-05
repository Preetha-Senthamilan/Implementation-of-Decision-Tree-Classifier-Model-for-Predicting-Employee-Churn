# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas

2.Import Decision tree classifier

3.Fit the data in the model

4.Find the accuracy score

## Program:
```

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: PREETHA.S
RegisterNumber: 212222230110

import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()    #no departments and no left
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])



```

## Output:

data.head()

![Screenshot 2023-10-05 095636](https://github.com/Preetha-Senthamilan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119390282/95a99534-705f-462b-82a5-98dd6c7a5a46)

data.info()

![Screenshot 2023-10-05 100250](https://github.com/Preetha-Senthamilan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119390282/aabd1244-398a-4796-b4c8-935a4d045476)

data is null() ans sum()

![Screenshot 2023-10-05 100411](https://github.com/Preetha-Senthamilan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119390282/cc5525bd-f229-4b1a-a36f-1ef4a8b25dd8)


data value counts()

![Screenshot 2023-10-05 100450](https://github.com/Preetha-Senthamilan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119390282/3ed4910f-a6e8-4959-be3c-904fd7b9f761)

data.head() for salary

![Screenshot 2023-10-05 100450](https://github.com/Preetha-Senthamilan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119390282/7ac527d2-e259-4244-84e3-29444173b25a)

x.data()

![Screenshot 2023-10-05 100612](https://github.com/Preetha-Senthamilan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119390282/9d9cdb2f-a9b2-41a2-854f-0ba3beb9143d)


Accuracy value

![Screenshot 2023-10-05 100749](https://github.com/Preetha-Senthamilan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119390282/0d1ed4f9-4a3b-4997-b88e-cb274223a3ae)


Predicted value

![Screenshot 2023-10-05 100854](https://github.com/Preetha-Senthamilan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119390282/be1f75d9-290a-495a-bafc-aba7fef4e851)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
