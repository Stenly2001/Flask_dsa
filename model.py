import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
     

iris_data = pd.read_excel(r"C:\Users\stenl\Documents\Datasets\iris .xls")
     

iris_data

iris_data.describe()
     

iris_data.isna().sum()

#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#ris_data['Classification'] = le.fit_transform(iris_data['Classification'])
#iris_data['Classification'].unique()


x = iris_data.drop(['Classification'],axis = 1)
y = iris_data['Classification']
     

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =  train_test_split(x,y,test_size=0.2, random_state=42)
     

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
model  = lr.fit(x_train,y_train)
lr_predictions = model.predict(x_test)

     

from sklearn.metrics import accuracy_score



print('Logistic regression Accuracy : ',accuracy_score(y_test,lr_predictions))


# save the model
import pickle
filename = 'lr_model.pkl'
pickle.dump(model, open(filename, 'wb'))
     

load_model = pickle.load(open(filename,'rb'))
     

load_model.predict([[6.0,2.2,4.0,1.0]])