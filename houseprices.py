from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
import pandas as pd

dataset = pd.read_csv("data/train.csv", header=0)

X = dataset.drop(columns=['YearRemodAdd', "Id", "SalePrice"], axis=1)
Y = dataset[['SalePrice']]

X = pd.get_dummies(X, columns=["MSSubClass", "MSZoning",
                              "Street", "Alley", "LotShape",
                              "LandContour", "Utilities", "LotConfig",
                              "LandSlope", "Neighborhood", "Condition1",
                              "Condition2", "BldgType", "HouseStyle",
                              "YearBuilt", "RoofStyle", "RoofMatl",
                              "Exterior1st", "Exterior2nd", "MasVnrType",
                              "ExterQual", "ExterCond", "Foundation",
                              "BsmtQual", "BsmtCond", "BsmtExposure",
                              "BsmtFinType1", "BsmtFinType2", "Heating",
                              "HeatingQC", "CentralAir", "Electrical",
                              "KitchenQual", "Functional", "FireplaceQu",
                              "GarageType", "GarageFinish", "GarageQual",
                              "GarageCond", "PavedDrive", "PoolQC",
                              "Fence", "MiscFeature", "MoSold",
                              "YrSold", "SaleType", "SaleCondition"])

Ymax = Y['SalePrice'].max()
Y = Y['SalePrice'].apply(lambda x: float(x) / Ymax)

input_units = X.shape[1]
print(X)
print(Y)

model = Sequential()
model.add(Dense(input_units, input_dim=input_units, activation='relu'))
model.add(Dense(input_units, activation='relu'))
model.add(Dense(input_units, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=250, batch_size=250,
          shuffle=True, validation_split=0.05, verbose=2)

scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

