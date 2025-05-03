# adapted from the kaggle submission notebook: https://www.kaggle.com/code/moghazy/simple-mlp-with-feature-engineering-and-eda

import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras import callbacks
from keras import optimizers
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np

training = pd.read_csv("Preparation/train.csv")
x_test = pd.read_csv("Preparation/test.csv")


training['n_new']= training['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
training['Title'] = 0
training.loc[training["n_new"] == 'Mr', 'Title'] = 1
training.loc[training["n_new"] == 'Miss', 'Title'] = 4
training.loc[training["n_new"] == 'Mrs', 'Title'] = 5
training.loc[training["n_new"] == 'Master', 'Title'] = 3
training.loc[training["n_new"] == 'Dr', 'Title'] = 2


C = training.Cabin[training.Cabin.isna()]
C_not = training.Cabin[training.Cabin.notna()]
C.values[:] = 0
C_not.values[:] = 1


cabine_not = pd.concat([C, C_not]).sort_index()


np.random.seed(0)
training['sp'] = training.SibSp + training['Parch']
training['cabine_n'] = cabine_not
training.cabine_n = training.cabine_n.astype(int)
training.drop(["n_new", "Name" ,"Embarked", "PassengerId","Ticket","Cabin"], inplace = True, axis = 1 )


x_train = training
repCol9 = {1 : 3 ,   2 : 2 , 3 : 1  }


x_train['IsAlone'] = 1
x_train['IsAlone'].loc[x_train['sp'] > 0] = 0

x_train.replace({'Pclass': repCol9} , inplace = True )
x_train = pd.get_dummies(x_train)

y_train = x_train["Survived"]
x_train = x_train.drop(['Survived'], axis = 1)



imp = IterativeImputer(max_iter=10, random_state=0)
x = imp.fit_transform(x_train)
x_train = pd.DataFrame(x, columns = x_train.columns)
x_train[[ 'Pclass', 'SibSp', 'Parch', 'Title', 'sp',
       'cabine_n', 'IsAlone', 'Sex_female', 'Sex_male']] = x_train[[ 'Pclass', 'SibSp', 'Parch', 'Title', 'sp',
       'cabine_n', 'IsAlone', 'Sex_female', 'Sex_male']].astype(int)



z = np.abs(stats.zscore(x_train))
zee = (np.where(z > 3))[1]


x_test['n_new']= x_test['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())


x_test['Title'] = 0

x_test.loc[x_test["n_new"] == 'Mr', 'Title'] = 1
x_test.loc[x_test["n_new"] == 'Miss', 'Title'] = 4
x_test.loc[x_test["n_new"] == 'Mrs', 'Title'] = 5
x_test.loc[x_test["n_new"] == 'Master', "Title"] = 3
x_test.loc[x_test["n_new"] == 'Dr', 'Title'] = 2


C = x_test.Cabin[x_test.Cabin.isna()]
C_not = x_test.Cabin[x_test.Cabin.notna()]
C.values[:] = 0
C_not.values[:] = 1

x_test['sp'] = x_test.SibSp + x_test['Parch']
x_test['cabine_n'] = cabine_not
x_test.cabine_n = x_test.cabine_n.astype(int)

x_test['IsAlone'] = 1
x_test['IsAlone'].loc[x_test['sp'] > 0] = 0


x_test.drop(["Name", "Embarked", "Ticket", "n_new", "Cabin"], inplace = True, axis = 1 )
x_test.replace({'Pclass': repCol9} , inplace = True )
x_test = pd.get_dummies(x_test)

fact = y_train[y_train == 0].count() / y_train[y_train == 1].count()

class_weight = {1: fact, 0: 1.}


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size= 0.3) 


id = x_test['PassengerId']

x_test.drop(['PassengerId'], inplace = True, axis = 1)

imp = IterativeImputer(max_iter=10, random_state=0)
x = imp.fit_transform(x_test)
x_test = pd.DataFrame(x, columns = x_train.columns)
x_test[[ 'Pclass', 'SibSp', 'Parch', 'Title', 'sp',
       'cabine_n', 'IsAlone', 'Sex_female', 'Sex_male']] = x_test[['Pclass', 'SibSp', 'Parch', 'Title', 'sp',
       'cabine_n', 'IsAlone', 'Sex_female', 'Sex_male']].astype(int)


InputDimension = 11
print(y_train.shape)

model = Sequential()
model.add(Dense(25,input_dim=InputDimension, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))

model.add(Dense(2, activation='softmax'))


earlystopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=0, mode='min')
optimizer = optimizers.Adam(learning_rate =0.001, weight_decay=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
history = model.fit(x_train, pd.get_dummies(y_train), epochs=2000, batch_size=60, validation_split=0.2, verbose=1, callbacks=[earlystopping], class_weight = class_weight)


# model.save("kaggle_titanic_model_new.keras")
x_test.to_csv("Preparation/processed_to_prune.csv", index = False)
x_val.insert(0, "y_test", y_val, True)
x_val.to_csv("Preparation/processed_to_test_new.csv", index = False)