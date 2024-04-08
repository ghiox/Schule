# %% [markdown]
# ## Data Import

# %%
import pandas as pd

# import data from data/ directory
df_train = pd.read_csv('data/train.csv')
df_store = pd.read_csv('data/store.csv')

# %% [markdown]
# ## Data overview

# %%
df_train.head(5)

# %%
df_train.info()

# %%
# check for missing values
df_train.isnull().sum()

# take log of sales
import numpy as np
df_train['Sales'] = np.log(df_train['Sales'])
df_train.head()

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Annahme: Die Spalte 'Date' ist im datetime-Format
df_train['Date'] = pd.to_datetime(df_train['Date'])

# Verkaufstrends im Zeitverlauf plotten
plt.figure(figsize=(12, 6))
plt.plot(df_train['Date'], df_train['Sales'], label='Verkaufszahlen', color='blue')

# Peaks markieren
plt.plot(df_train['Date'], df_train['Sales'])

plt.title('Verkaufstrends im Zeitverlauf mit Peaks')
plt.xlabel('Datum')
plt.ylabel('Verkaufszahlen')
plt.legend()
plt.show()


# %%
import seaborn as sns

# Boxplot zur Visualisierung des Einflusses von Werbeaktionen auf Verkäufe
plt.figure(figsize=(8, 6))
sns.boxplot(x=df_train['Promo'], y=df_train['Sales'])
plt.title('Einfluss von Werbeaktionen auf Verkäufe')
plt.xlabel('Promo')
plt.ylabel('Verkaufszahlen')
plt.show()


# %%
# Durchschnittliche Verkaufszahlen pro Wochentag
average_sales_per_day = df_train.groupby('DayOfWeek')['Sales'].mean()

# Balkendiagramm zur Visualisierung der durchschnittlichen Verkaufszahlen pro Wochentag
plt.figure(figsize=(8, 6))
average_sales_per_day.plot(kind='bar')
plt.title('Durchschnittliche Verkaufszahlen pro Wochentag')
plt.xlabel('Wochentag')
plt.ylabel('Durchschnittliche Verkaufszahlen')
plt.show()


# %%
df_trainstore = pd.merge(df_train, df_store, on='Store')

df_trainstore

# %%
df_trainstore.dtypes


# %%
df_trainstore['Month'] = df_trainstore['Date'].dt.month


# %%
df_trainstore.info()

# %%
# import datetime
from datetime import datetime

# splitt date into day, month, year
df_trainstore['Day'] = df_trainstore['Date'].dt.day
df_trainstore['Month'] = df_trainstore['Date'].dt.month
df_trainstore['Year'] = df_trainstore['Date'].dt.year

# get week number with iso format
df_trainstore['Week'] = df_trainstore['Date'].dt.isocalendar().week
df_trainstore['Quarter'] = df_trainstore['Date'].dt.quarter

# remove date column
df_trainstore = df_trainstore.drop('Date', axis=1)

# %%
df_trainstore.sample(10)

# %%
# use one hot encoding for categorical variables
df_trainstore = pd.get_dummies(df_trainstore, columns=['StoreType', 'Assortment', 'StateHoliday'])

# convert dummies to int
df_trainstore = df_trainstore.astype({'StateHoliday_0': 'int', 'StateHoliday_a': 'int', 'StateHoliday_b': 'int', 'StateHoliday_c': 'int'})

# convert dummies storetype, assortment to int
df_trainstore = df_trainstore.astype({'StoreType_a': 'int', 'StoreType_b': 'int', 'StoreType_c': 'int', 'StoreType_d': 'int'})
df_trainstore = df_trainstore.astype({'Assortment_a': 'int', 'Assortment_b': 'int', 'Assortment_c': 'int'})

df_trainstore.sample()

# %%
# map promo interval to numerical
promo_interval_map = {'Jan,Apr,Jul,Oct': 1, 'Feb,May,Aug,Nov': 2, 'Mar,Jun,Sept,Dec': 3}
df_trainstore['PromoInterval'] = df_trainstore['PromoInterval'].map(promo_interval_map)

# if promo2 is 0 then promo2sinceweek, promo2sinceyear, promointerval are 0
df_trainstore['Promo2SinceWeek'] = df_trainstore['Promo2SinceWeek'].fillna(0)
df_trainstore['Promo2SinceYear'] = df_trainstore['Promo2SinceYear'].fillna(0)
df_trainstore['PromoInterval'] = df_trainstore['PromoInterval'].fillna(0)

df_trainstore.sample(10)

# %%
# check for missing values
df_trainstore.isnull().sum()

# fill missing values with 0
df_trainstore.fillna(0, inplace=True)

df_trainstore.info()

# %%
# Assuming your DataFrame is named df_trainstore
df_trainstore['Date'] = pd.to_datetime(df_trainstore['Year'].astype(str) + '-' + df_trainstore['Month'].astype(str) + '-' + df_trainstore['Day'].astype(str))
print(len(df_trainstore))


# Sort the DataFrame by Store and Date
df_trainstore = df_trainstore.sort_values(by=['Store', 'Date'])


# Create a new DataFrame for the last 7 days for each store
df_last7days = df_trainstore.groupby('Store').tail(7)

# Create a new DataFrame with the sum of sales and customers for the last 7 days
df_last7days_sum = df_last7days.groupby('Store').agg({
    'Sales': 'sum',
    'Customers': 'sum'
}).reset_index()

# Merge the sum data back into the original DataFrame
df_trainstore = pd.merge(df_trainstore, df_last7days_sum, on='Store', how='left', suffixes=('', '_Last7Days'))

# Create new columns
df_trainstore['SalesPerCustomer'] = df_trainstore['Sales'] / df_trainstore['Customers']
df_trainstore['SalesLastWeek'] = df_trainstore['Sales_Last7Days']
df_trainstore['CustomersLastWeek'] = df_trainstore['Customers_Last7Days']


print("len sales_last7days: ", df_trainstore['Sales_Last7Days'].isna().sum())
print("len customers_last7days: ", df_trainstore['Customers_Last7Days'].isna().sum())
print("count 0 values in customer last 7 days: ", df_trainstore[df_trainstore['Customers_Last7Days'] == 0].shape[0])

df_trainstore['SalesPerCustomerLastWeek'] = df_trainstore['Sales_Last7Days'] / df_trainstore['Customers_Last7Days']

print("len salespercustomerlastweek: ", df_trainstore['SalesPerCustomerLastWeek'].isna().sum())

# Drop unnecessary columns
df_trainstore = df_trainstore.drop(['Sales_Last7Days', 'Customers_Last7Days'], axis=1)

# Print or view the updated DataFrame
df_trainstore.sample()

# %%
# if sales is 0 then sales per customer is 0
df_trainstore['SalesPerCustomer'] = df_trainstore['SalesPerCustomer'].fillna(0)

df_trainstore.info()

# %%
# fill na values with 0 for sales per customer last week
df_trainstore['SalesPerCustomerLastWeek'] = df_trainstore['SalesPerCustomerLastWeek'].fillna(0)

df_trainstore.info()

# %%
# remove date column
df_trainstore = df_trainstore.drop('Date', axis=1)

# %%
# plot correlation matrix
plt.figure(figsize=(20, 20))
correlation_matrix = df_trainstore.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=2)
plt.title('Correlation Matrix')
plt.show()

# %%


