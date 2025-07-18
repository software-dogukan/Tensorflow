from sklearn.preprocessing import MinMaxScaler

from Model1.Model1 import y_train
print(dataframe.describe())#datamızın bir çok verisini burada gözlemliyoruz ilk olarak veri çekildiğinde bu yapılmalı
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tesorflow.keras.layers import Dense
model=Sequential()
model.add(Dense(5,activation="relu"))
model.add(Dense(5,activation="relu"))
model.add(Dense(5,activation="relu"))
model.add(Dense(1))
model.compile(optimizer="rmsprop",loss="mse")

model.fit(x_train,y_train,epochs=250)

loss=model.history.history["loss"]
sbn.lineplot(x=range(len(loss)),y=loss)
#buraya kadar loss grafiğine bakıyoruz

trainloss=model.evaluate(x_train,y_train,verbose=0)
testloss=model.evaluate(x_test,y_test,verbose=0)
#buradada lossların birbirine yakın ve ne kadar düşük olmasını inceliyoruz

predict=model.predict(x_test)
from sklearn.metrics import mean_absolute_error,mean_squared_error
print(mean_absolute_error(y_test,predict))
print(mean_squared_error())
from tensorflow.keras.models import load_model
model.save("deeplearning_model.h5")#oluşturulan modeli kaydetme
call=load_model("deeplearning_model.h5")#oluşturulan model çağırma