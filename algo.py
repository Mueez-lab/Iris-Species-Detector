import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('C:/Users/mueez/OneDrive/Desktop/Iris Classification/IRIS.csv')
df.dropna(inplace=True)

df.hist(bins=30, figsize=(15, 10), edgecolor='black')
plt.suptitle("Histogram of Trained data features", fontsize=16)
# plt.show()

plt.figure()
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap='YlGnBu')
plt.title("Correlation Matrix")
# plt.show()

# Feature Engineering 
df['sepal_ratio'] = df['sepal_length'] / df['sepal_width']
df['petal_area'] = df['petal_length'] * df['petal_width']
# we will use Random Forest so we willn't drop old features 

# Coverting specie data in string into numberic form 
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])
print(df)

x = df.drop(['species'], axis=1)
y = df['species']

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)

forest = RandomForestRegressor()
forest.fit(X_train,Y_train)
print(f"R2 on testing data score with Random Foest {forest.score(X_test,Y_test)}")

# Already Achived the best score so now we are using GridSearchCV for learning purpose only 
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [150, 200, 250],
    'max_depth': [None, 2, ],
    'min_samples_split': [2, 4]
}

grid = GridSearchCV(forest, param_grid, cv=5) # cv Split my data into 5 equal parts 
grid.fit(X_train, Y_train)

print("Best parameters:", grid.best_params_)
print("Best score (CV):", grid.best_score_)
print("Test accuracy with best model:", grid.score(X_test, Y_test))