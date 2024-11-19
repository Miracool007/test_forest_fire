import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import seaborn as sns
pd.set_option("display.max_columns", 20)


df = pd.read_csv("C:/Users/USER/Desktop/Datasets/Algerian_forest_fire_cleaned_dataset.csv")

# X = df.drop(["Classes"], axis=1)
# y = df["Classes"]

# plt.scatter(X,y, color="green")



# df.loc[:123, "Region"] = 0
# df.loc[123:, "Region"] = 1
#
# df["Region"] = df["Region"].astype(int)
#
# df = df.dropna().reset_index(drop=True)
#
# df = df.drop(122).reset_index(drop=True)
#
# df.columns = df.columns.str.strip()
#
# df[["day", "month", "year","Temperature", "RH", "Ws"]] = df[["day", "month",
#                                                              "year", "Temperature", "RH", "Ws"]].astype(int)
#
#
# x = [features for features in df.columns if df[features].dtypes == "O"]
#
#
# for i in x:
#     if i != "Classes":
#         df[i] = df[i].astype(float)
#
#
# df = df.drop(["day", "month", "year"], axis=1)
#
#
# df["Classes"] = np.where(df["Classes"].str.contains("not fire"),0,1)

# plt.style.use("seaborn-v0_8")
# df.hist(bins=50, figsize=(20,15))

# x = (df["Classes"].value_counts(normalize=True) * 100)
#
# classlabel = ["Fire", "Not Fire"]
# plt.figure(figsize=(8,5))
# plt.pie(x, labels=classlabel, autopct="%1.1f%%")
# plt.title("Percentage of Fire Distribution")


# dftemp = df.loc[df["Region"] == 1]
# plt.subplots(figsize=(8,4))
# sns.set_style("whitegrid")
# sns.countplot(x="month", hue="Classes", data=df)
# -------------------------------------------------- Feature Engineering ----------------------------------------------

df = df.drop(["day", "month", "year","Unnamed: 0"], axis=1)

df["Classes"] = np.where(df["Classes"].str.contains("not fire"),0, 1)

X = df.drop("FWI", axis=1)
y = df["FWI"]

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# corr = X_train.corr()
# plt.figure(figsize=(10,5))
# sns.heatmap(corr, annot=True)

def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr



xy = correlation(X_train, 0.85)


X_test = X_test.drop(xy, axis=1)
X_train = X_train.drop(xy, axis=1)



scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# plt.subplots(figsize=(10,5))
# plt.subplot(1, 2, 1)
# sns.boxplot(data=X_train)
# plt.title("X_train before Scaling")
# plt.subplot(1, 2, 2)
# sns.boxplot(data=X_train_scaled)
# plt.title("X_train after Scaling")


model = LinearRegression()

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

score = r2_score(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

# print(mae)
# print()
# print(score)


# plt.scatter(y_test, y_pred)

lasso = Lasso()

lasso.fit(X_train_scaled, y_train)

ypred = lasso.predict(X_test_scaled)

score1 = r2_score(y_test, ypred)


ridge = Ridge()
ridge.fit(X_train_scaled, y_train)
ypredd = ridge.predict(X_test_scaled)

score2 = r2_score(y_test, ypredd)

elastic = ElasticNet()
elastic.fit(X_train_scaled, y_train)
ypreds = elastic.predict(X_test_scaled)
score3 = r2_score(y_test, ypreds)

print("Lasso Score:",score1)
print("Ridge Score:", score2)
print("Elastic Score:", score3)

# pickle.dump(scaler,open("scaler.pkl", "wb"))
# pickle.dump(ridge,open("ridge.pkl", "wb"))

pickle.dump(scaler,open("scalers.pkl", "wb"))




plt.show()
