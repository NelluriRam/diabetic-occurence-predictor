import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv('diabetes2.csv')
df   = pd.DataFrame(data)
#print(df)
df['Outcome'].value_counts()
array = data.values
X = array[:,0:4]
y = array[:,4]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=21)
reg=LogisticRegression()
reg.fit(X_train,y_train)
y_log_pred = reg.predict(X_test)
y_log = reg.predict(X_train)
print('Accuracy Score of your model in training is : ',metrics.accuracy_score(y_train,y_log))
print('Accuracy Score of your model is : ',metrics.accuracy_score(y_test,y_log_pred))
roc_auc_score(y_test,y_log_pred)
pickle.dump(reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))