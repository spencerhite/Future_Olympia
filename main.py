import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

teams = pd.read_csv("teams.csv")

teams = teams[["team", "country", "year", "athletes", "age", "prev_medals", "medals"]]
from sklearn.metrics import mean_absolute_error



#sns.lmplot(x="athletes", y="medals", data=teams, fit_reg=True, ci=None)
#sns.lmplot(x="age", y="medals", data=teams, fit_reg=True, ci=None)
#teams.plot.hist(y="medals")
#plt.show()

teams[teams.isnull().any(axis=1)]

teams=teams.dropna()

train = teams[teams["year"] < 2012].copy()
test = teams[teams["year"] >= 2012].copy()
train.shape
test.shape

reg = LinearRegression()
predictors = ["athletes", "prev_medals"]
target = "medals"

reg.fit(train[predictors], train["medals"])

predictions = reg.predict(test[predictors])
test["predictions"] = predictions
test.loc[test["predictions"] < 0, "predictions"]= 0
test["predictions"] = test["predictions"].round()

error = mean_absolute_error(test["medals"], test["predictions"])


print(test[test["team"] == "IND"])

errors = (test["medals"] - test["predictions"]).abs()
print(errors)

error_by_team = errors.groupby(test["team"]).mean()
print(error_by_team)

medals_by_team = test["medals"].groupby(test["team"]).mean()
error_ratio = error_by_team / medals_by_team

error_ratio[~pd.isnull(error_ratio)]
error_ratio = error_ratio[np.isfinite(error_ratio)]
error_ratio.plot.hist()
#plt.show()

#print(error_ratio.sort_values())

#Could add more predictors in
#Could Try different models
#Try reshaping columns that aren't non-linear
#Measure the error more predictably



