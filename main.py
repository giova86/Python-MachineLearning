import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (5,5)


# import data
df = pd.read_csv('./data/california_housing_train.csv')
print(df.isnull().sum())

df = df[df['median_house_value']<200000]

# data preparation
y = df['median_house_value']
X = df.drop('median_house_value', axis=1)

# for i in df.columns:
#     plt.scatter(X[i], y)
#     plt.show()

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# models 1
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

y_rf_predicion_train = rf.predict(X_train)
y_rf_predicion_test = rf.predict(X_test)

rf_train_mse = mean_squared_error(y_train, y_rf_predicion_train)
rf_train_r2 = r2_score(y_train, y_rf_predicion_train)
rf_train_mae = mean_absolute_error(y_train, y_rf_predicion_train)
rf_test_mse = mean_squared_error(y_test, y_rf_predicion_test)
rf_test_r2 = r2_score(y_test, y_rf_predicion_test)
rf_test_mae = mean_absolute_error(y_test, y_rf_predicion_test)

rf_summary = pd.DataFrame(['Random Forest',rf_train_mse,rf_train_r2,rf_train_mae,rf_test_mse,rf_test_r2,rf_test_mae]).transpose()
rf_summary.columns = ['Model', 'MSE (Train)', 'R2 (Train)','MAE (Train)', 'MSE (Test)', 'R2 (Test)', 'MAE (Test)']

plt.scatter(y_rf_predicion_test, y_test)
plt.xlim([min(min(y_rf_predicion_test), min(y_test)), max(max(y_rf_predicion_test), max(y_test))])
plt.ylim([min(min(y_rf_predicion_test), min(y_test)), max(max(y_rf_predicion_test), max(y_test))])
plt.show()
plt.hist(y_rf_predicion_test-y_test)
plt.show()

# models 2
lr = LinearRegression()
lr.fit(X_train, y_train)

y_lr_predicion_train = lr.predict(X_train)
y_lr_predicion_test = lr.predict(X_test)

lr_train_mse = mean_squared_error(y_train, y_lr_predicion_train)
lr_train_r2 = r2_score(y_train, y_lr_predicion_train)
lr_train_mae = mean_absolute_error(y_train, y_lr_predicion_train)
lr_test_mse = mean_squared_error(y_test, y_lr_predicion_test)
lr_test_r2 = r2_score(y_test, y_lr_predicion_test)
lr_test_mae = mean_absolute_error(y_test, y_lr_predicion_test)

lr_summary = pd.DataFrame(['Linear Regression',lr_train_mse,lr_train_r2,lr_train_mae,lr_test_mse,lr_test_r2,lr_test_mae]).transpose()
lr_summary.columns = ['Model', 'MSE (Train)', 'R2 (Train)','MAE (Train)', 'MSE (Test)', 'R2 (Test)', 'MAE (Test)']

plt.scatter(y_lr_predicion_test, y_test)
plt.xlim([min(min(y_lr_predicion_test), min(y_test)), max(max(y_lr_predicion_test), max(y_test))])
plt.ylim([min(min(y_lr_predicion_test), min(y_test)), max(max(y_lr_predicion_test), max(y_test))])
plt.show()

# models 3
svm = svm.SVR()
svm.fit(X_train, y_train)

y_svm_predicion_train = svm.predict(X_train)
y_svm_predicion_test = svm.predict(X_test)

svm_train_mse = mean_squared_error(y_train, y_svm_predicion_train)
svm_train_r2 = r2_score(y_train, y_svm_predicion_train)
svm_train_mae = mean_absolute_error(y_train, y_svm_predicion_train)
svm_test_mse = mean_squared_error(y_test, y_svm_predicion_test)
svm_test_r2 = r2_score(y_test, y_svm_predicion_test)
svm_test_mae = mean_absolute_error(y_test, y_svm_predicion_test)

svm_summary = pd.DataFrame(['SVM',svm_train_mse,svm_train_r2,svm_train_mae,svm_test_mse,svm_test_r2,svm_test_mae]).transpose()
svm_summary.columns = ['Model', 'MSE (Train)', 'R2 (Train)','MAE (Train)', 'MSE (Test)', 'R2 (Test)', 'MAE (Test)']

plt.scatter(y_svm_predicion_test, y_test)
plt.xlim([min(min(y_svm_predicion_test), min(y_test)), max(max(y_svm_predicion_test), max(y_test))])
plt.ylim([min(min(y_svm_predicion_test), min(y_test)), max(max(y_svm_predicion_test), max(y_test))])
plt.show()

# models 4
GDR = SGDRegressor()
GDR.fit(X_train, y_train)

y_GDR_predicion_train = GDR.predict(X_train)
y_GDR_predicion_test = GDR.predict(X_test)

GDR_train_mse = mean_squared_error(y_train, y_GDR_predicion_train)
GDR_train_r2 = r2_score(y_train, y_GDR_predicion_train)
GDR_train_mae = mean_absolute_error(y_train, y_GDR_predicion_train)
GDR_test_mse = mean_squared_error(y_test, y_GDR_predicion_test)
GDR_test_r2 = r2_score(y_test, y_GDR_predicion_test)
GDR_test_mae = mean_absolute_error(y_test, y_GDR_predicion_test)

GDR_summary = pd.DataFrame(['GDR',GDR_train_mse,GDR_train_r2,GDR_train_mae,GDR_test_mse,GDR_test_r2,GDR_test_mae]).transpose()
GDR_summary.columns = ['Model', 'MSE (Train)', 'R2 (Train)','MAE (Train)', 'MSE (Test)', 'R2 (Test)', 'MAE (Test)']

plt.scatter(y_GDR_predicion_test, y_test)
plt.xlim([min(min(y_GDR_predicion_test), min(y_test)), max(max(y_GDR_predicion_test), max(y_test))])
plt.ylim([min(min(y_GDR_predicion_test), min(y_test)), max(max(y_GDR_predicion_test), max(y_test))])
plt.show()

summary = pd.concat([rf_summary, lr_summary, svm_summary, GDR_summary])
print(summary)
