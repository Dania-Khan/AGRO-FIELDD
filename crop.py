import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sb
import pickle

df = pd.read_csv("data.csv")
df.head()

df= df.drop(['DISTRICT','CROP'],axis=1)

#df.info()

df['YIELD'] = df['PRODUCTION-(TONNES)']/df['AREA-(HECTARE)']
df

'''C_mat = df.corr()
fig = plt.figure(figsize = (15,15))

sb.heatmap(C_mat, vmax = .8, square = True)
plt.show()'''

X=df[['YEAR',	'AVG-TEMP',	'RAINFALL-(MM)',	'AREA-(HECTARE)']]
y=df['YIELD']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

'''y_pred=regr.predict(X_test)
print(y_pred)'''

'''from sklearn.metrics import r2_score
r2_score(y_test,y_pred)'''

'''plt.figure(figsize=(15,10))
plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')'''

#regr.predict([[2009,27,114,3003]])

pickle.dump(regr, open('model.pkl','wb'))

#Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2009,27,114,3003]]))

