import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/user/OneDrive/Desktop/Python Bootcamp/22-Deep Learning/DATA/cancer_classification.csv')
df.info()
df.describe().transpose()

# EDA
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='benign_0__mal_1',data=df); plt.show()
sns.heatmap(df.corr()); plt.show()
df.corr()['benign_0__mal_1'].sort_values()
df.corr()['benign_0__mal_1'].sort_values().plot(kind='bar'); plt.show()
df.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar'); plt.show()

# TRAIN/TEST SPLIT
X = df.drop('benign_0__mal_1',axis=1).values
y = df['benign_0__mal_1'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=101)

# SCALING DATA
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# CREATING THE MODEL
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout

model = Sequential()
model.add(Dense(units=30,activation='relu'))
model.add(Dense(units=15,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))
# For a binary classification problem
model.compile(loss='binary_crossentropy', optimizer='adam')

# TRAINING THE MODEL
model.fit(x=X_train, 
          y=y_train, 
          epochs=600,
          validation_data=(X_test, y_test), verbose=1
          )

model_loss = pd.DataFrame(model.history.history)
model_loss.plot(); plt.show()

# EX2: EARLY STOPPING
model = Sequential()
model.add(Dense(units=30,activation='relu'))
model.add(Dense(units=15,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
model.fit(x=X_train, 
          y=y_train, 
          epochs=600,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop]
          )

model_loss = pd.DataFrame(model.history.history)
model_loss.plot(); plt.show()

# EX3: ADDING IN DROPOUT LAYERS
from tensorflow.keras.layers import Dropout

model = Sequential()
model.add(Dense(units=30,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=15,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x=X_train, 
          y=y_train, 
          epochs=600,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop]
          )

model_loss = pd.DataFrame(model.history.history)
model_loss.plot(); plt.show()























