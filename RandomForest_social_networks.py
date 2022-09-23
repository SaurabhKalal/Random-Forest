
import pandas as pd
data=pd.read_csv(r"Social_Network_Ads.csv")
x=data.iloc[:,1:4].values
y=data.iloc[:,-1].values
y=y.reshape(y.shape[0],1)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
CT=ColumnTransformer([('OHE',OneHotEncoder(drop='first'),[0])],remainder='passthrough')
x=CT.fit_transform(x)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc=sc.fit(x)
x=sc.transform(x)
#x=sc.fit_transform(x)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=.2,random_state=23)

from sklearn.ensemble import RandomForestClassifier as r
estimator=[]
for i in range (1,10):
    model=r(n_estimators=i,random_state=2)
    model.fit(X_train,Y_train)
    Y_pred=model.predict(X_test)

    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(Y_pred,Y_test)
    print(cm)

    from sklearn.metrics import accuracy_score
    score=accuracy_score(Y_pred,Y_test)
    print(score)
    estimator.append(score)
X1=[1,2,3,4,5,6,7,8,9]    
import matplotlib.pyplot as plt
plt.plot(X1,estimator,color='b',label='Accuracy plot')
plt.legend()
plt.show()

import pickle
Pur_model="C:/Users/Admin/Desktop/PYTHON/PROJECT/Social_Network_Ads.csv"
#file=open('C:/Users/Admin/Desktop/PYTHON/PROJECT/Social_Network_Ads.txt','w+')
#file.write("This is a test file")
#file.close()
pickle.dump(model,open(Pur_model,'wb'))
