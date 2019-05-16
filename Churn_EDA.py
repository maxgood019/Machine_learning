
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv('churn_data.csv')

#check the data
df.head()
df.columns
df.describe()

#data preprocessing

#removing nan

df.isna().any() #age, credit_score, rewards_earned have NA value
df.isna().sum()
df = df[pd.notnull(df['age'])] #only 4 missing value.
df = df.drop(columns = ['credit_score','rewards_earned'])

# histogram

df2 = df.drop(columns = ['user','churn'])
fig = plt.figure(figsize=(15,12))
plt.suptitle('Histogram of Numerical columns', fontsize = 20)

for i in range(1, df2.shape[1] + 1):
    plt.subplot(6, 5, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(df2.columns.values[i - 1])

    vals = np.size(df2.iloc[:, i - 1].unique())
    
    plt.hist(df2.iloc[:, i - 1], bins=vals,color="#3F5D7D")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#pie plots
df2 = df[['housing', 'is_referred', 'app_downloaded',
                    'web_user', 'app_web_user', 'ios_user',
                    'android_user', 'registered_phones', 'payment_type',
                    'waiting_4_loan', 'cancelled_loan',
                    'received_loan', 'rejected_loan', 'zodiac_sign',
                    'left_for_two_month_plus', 'left_for_one_month', 'is_referred']]

fig = plt.figure(figsize=(15, 12))
plt.suptitle('Pie Chart', fontsize=20)
for i in range(1, df2.shape[1] + 1):
    plt.subplot(6, 3, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(df2.columns.values[i - 1])
    
    values = df2.iloc[:,i-1].value_counts(normalize = True).values
    index = df2.iloc[:,i-1].value_counts(normalize = True).index
    plt.pie(values, labels= index, autopct = '%1.1f%%')    
    plt.axis('equal')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#check uneaven features
df[df2.waiting_4_loan== 1].churn.value_counts()
df[df2.cancelled_loan == 1].churn.value_counts()
df[df2.rejected_loan == 1].churn.value_counts()
df[df2.left_for_one_month == 1].churn.value_counts()

# find correlation 
df.drop(columns = ['user','housing','payment_type','churn',
                   'zodiac_sign']).corrwith(df.churn).plot.bar(
    figsize = (20,10), title = 'Correlation with the variables', fontsize = 10, rot = 45,
    grid = True)

#correlation Matrix 
sns.set(style = 'white') #background

corr = df.drop(columns = ['user','churn']).corr()

mask = np.zeros_like(corr,dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

#set up plt figure
f, ax = plt.subplots(figsize= (18,15))

cmap = sns.diverging_palette(220,10,as_cmap = True)

sns.heatmap(corr,mask = mask, cmap= cmap, vmax=3,
            center = 0,
            square = True, linewidths = 5,
            cbar_kws={"shrink": .5})

df = df.drop(columns = ['app_web_user']) #app_web_user are depending on web_user and app_downloaded

df.to_csv('modified_churn_data.csv', index = True)












































