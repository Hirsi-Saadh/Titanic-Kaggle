import pickle

import pandas as pd

#importing the dataset
df=pd.read_csv(r'csv/train_titanic.csv')

#dropping the columns
df=df.drop(['Cabin','Name','Ticket','PassengerId'],axis=1)
df['Fmembers']= df.iloc[:,4:6].sum(axis=1)
df=df.drop(['SibSp','Parch'],axis=1)

#getting the mode of embarked and fill empty spaces
df['Embarked']=df.Embarked.fillna(df['Embarked'].mode()[0])

#take the mean age and fill empty values
df['Age']=df.Age.fillna(df['Age'].mean())

#value replacement
df['Embarked']=df.Embarked.replace( {'S': 0, 'Q': 1, 'C': 2})
data=pd.get_dummies(df,columns=['Sex'])
data=data.drop(['Sex_female'],axis=1)


Data_X=data.iloc[:,1:]
Data_y= data.iloc[:,0]

#implementing Cat RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#training the dataset
X_train,X_test,y_train,y_test=train_test_split(Data_X,Data_y,random_state=0,test_size=0.2)
classify=RandomForestClassifier()
classify.fit(X_train,y_train)

pickle.dump(classify, open('model.pkl','wb'))