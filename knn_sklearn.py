import pandas as pd
import numpy as np
import time, random
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

start_time =  time.time()

df = pd.read_csv('data/breast-cancer-wisconsin.txt')
# according to names file, missing values are denoted by a "?" and must be replaced
df.replace('?', -9999, inplace=True)
# drop the id column as it has nothing to do with the class
df.drop(['id'], axis=1, inplace=True)
df['class'] = df['class'].apply(lambda x: 'benign' if x == 2 else 'malignant')

x = np.array(df.drop(['class'],axis=1))
y = np.array(df['class'])

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)

accuracy = clf.score(x_test,y_test)

example_measures = np.array([random.randint(1,10) for i in range(x.shape[1])])
example_measures = example_measures.reshape(1,-1)

prediction = clf.predict(example_measures)

print(f"""point to predict: {example_measures}
prediction: {prediction}
accuracy: {accuracy}
{(time.time()-start_time)} seconds""")

