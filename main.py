
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_csv("housing.csv")

data

data.info()

data.dropna(inplace=True)

data

data.info()

from sklearn.model_selection import train_test_split

X = data.drop(["median_house_value"], axis=1)

y = data["median_house_value"]

X

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

train_data = X_train.join(y_train)

train_data

train_data.hist(figsize=(20, 10))

train_data.corr(numeric_only=True)

plt.figure(figsize=(20, 10))
sns.heatmap(train_data.corr(numeric_only=True), annot=True, cmap="YlGnBu")

train_data.hist(figsize=(20, 10))


train_data["total_rooms"] = np.log(train_data["total_rooms"] + 1)
train_data["total_bedrooms"] = np.log(train_data["total_bedrooms"] + 1)
train_data["households"] = np.log(train_data["households"] + 1)
train_data["population"] = np.log(train_data["population"] + 1)

train_data.hist(figsize=(20, 10))


train_data.ocean_proximity.value_counts()

train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity, dtype=int)).drop(["ocean_proximity"], axis=1)

plt.figure(figsize=(20, 10))
sns.heatmap(train_data.corr(numeric_only=True), annot=True, cmap="YlGnBu")

plt.figure(figsize=(20, 10))
sns.scatterplot(x="latitude", y="longitude", data=train_data, hue="median_house_value", palette="coolwarm")

train_data["bedroom_ratio"] = train_data["total_bedrooms"] / train_data["total_rooms"]
train_data["household_rooms"] = train_data["total_rooms"] / train_data["households"]

plt.figure(figsize=(20, 10))
sns.heatmap(train_data.corr(numeric_only=True), annot=True, cmap="YlGnBu")

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train, y_train = train_data.drop(["median_house_value"], axis=1), train_data["median_house_value"]
X_train_s = scaler.fit_transform(X_train)

reg = LinearRegression()
reg.fit(X_train_s, y_train)

test_data = X_test.join(y_test)
test_data

test_data["total_rooms"] = np.log(test_data["total_rooms"] + 1)
test_data["total_bedrooms"] = np.log(test_data["total_bedrooms"] + 1)
test_data["households"] = np.log(test_data["households"] + 1)
test_data["population"] = np.log(test_data["population"] + 1)

test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity, dtype=int)).drop(["ocean_proximity"], axis=1)

test_data["bedroom_ratio"] = test_data["total_bedrooms"] / test_data["total_rooms"]
test_data["household_rooms"] = test_data["total_rooms"] / test_data["households"]

X_test, y_test = test_data.drop(["median_house_value"], axis=1), test_data["median_house_value"]
X_test_s = scaler.transform(X_test)

X_train

X_test

reg.score(X_test_s, y_test)

from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()

forest.fit(X_train_s, y_train)

forest.score(X_test_s, y_test)

