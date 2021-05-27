import numpy
import pandas
import joblib
from sklearn.linear_model import LinearRegression


print("How much year of experience do you have")

exp = int(input("Enter your years of experience: "))

print()

model = joblib.load("salary.pk1")

print("Your salary predicted is {}".format(model.predict([[exp]])))


