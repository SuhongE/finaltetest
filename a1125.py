

#In[1]
import pandas as pd
edges = pd.DataFrame({'source': [0, 1, 2], 'target': [2, 2, 3],
                      'weight': [3, 4, 5], 'color': ['red', 'blue', 'blue']})

weight_dict = {3:"M", 4:"L", 5:"XL"}
edges["weight_sign"] = edges["weight"].map(weight_dict)
weight_sign = pd.get_dummies(edges["weight_sign"])
weight_sign
#In[2]
raw_data = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 'Dragoons', 'Dragoons', 'Dragoons', 'Dragoons', 'Scouts', 'Scouts', 'Scouts', 'Scouts'],
            'company': ['1st', '1st', '2nd', '2nd', '1st', '1st', '2nd', '2nd','1st', '1st', '2nd', '2nd'],
            'name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze', 'Jacon', 'Ryaner', 'Sone', 'Sloan', 'Piger', 'Riani', 'Ali'],
            'preTestScore': [4, 24, 31, 2, 3, 4, 24, 31, 2, 3, 2, 3],
            'postTestScore': [25, 94, 57, 62, 70, 25, 94, 57, 62, 70, 62, 70]}

df = pd.DataFrame(raw_data, columns = ['regiment', 'company', 'name', 'preTestScore', 'postTestScore'])
df
#In[3]
bins = [0, 25, 50, 75, 100] # bins 정의(1-25, 26-50, 51-75, 76-100)
group_names = ['Low', 'Okay', 'Good', 'Great']
categories = pd.cut(
df['postTestScore'], bins, labels=group_names)
categories
#In[4]
df = pd.DataFrame(
    {'A':[14.00,90.20,90.95,96.27,91.21],
     'B':[103.02,107.26,110.35,114.23,114.68],
     'C':['big','small','big','small','small']})
df


#In[5]
( df["A"] - df["A"].min() ) / (df["A"].max() - df["A"].min()) * (100-60) + 60

#In[6]
( df["A"] - df["A"].mean() ) / (df["A"].std())

#In[7]
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
sns.set_theme(style="whitegrid", color_codes=True)

DATA_DIR = '../source/ch06/'
os.listdir(DATA_DIR)

#In[8]
DATA_DIR = '../source/ch06/'
data_files = sorted([os.path.join(DATA_DIR, filename)
for filename in os.listdir(DATA_DIR)], reverse=True)
data_files

#In[9]
df_list = []
for filename in data_files:
    df_list.append(pd.read_csv(filename))
# (2) 두 개의 데이터프레임을 하나로 통합
df = pd.concat(df_list, sort=False)

# (3) 인덱스 초기화
df = df.reset_index(drop=True)

# (4) 결과 출력
df.head(5)
#In[10]
# (1) 데이터를 소수점 두 번째 자리까지 출력
pd.options.display.float_format = '{:.2f}'.format

# (2) 결측치 값의 합을 데이터의 개수로 나눠 비율로 출력
df.isnull().sum() / len(df) * 100

#In[11]
#셩별을 기준으로 결측치 처리
df[df["Age"].notnull()].groupby(["Sex"])["Age"].mean()

#객실 등급을 기준으로 결측치 처리
#df[df["Age"].notnull()].groupby(["Pclass"])["Age"].mean()

#In[12]
#df["Age"].fillna(df.groupby("Pclass")["Age"].transform("mean"), inplace=True)
df["Age"] = df["Age"].fillna(
    df.groupby("Sex")["Age"].transform("mean"))
df.info()
#In[13]
def merge_and_get(ldf, rdf, on, how="inner", index=None):
    if index is True:
        return pd.merge(ldf,rdf, how=how, left_index=True,
                        right_index=True)
    else:
        return pd.merge(ldf,rdf, how=how, on=on)
#In[14]
one_hot_df = merge_and_get(
    df, pd.get_dummies(df["Sex"], prefix="Sex"), on=None, index=True)
one_hot_df = merge_and_get(
    one_hot_df, pd.get_dummies(
        df["Pclass"], prefix="Pclass"), on=None, index=True)
one_hot_df = merge_and_get(
    one_hot_df, pd.get_dummies(
        df["Embarked"], prefix="Embarked"), on=None, index=True)
#In[15]
temp_columns = ["Sex", "Pclass", "Embarked"]

for col_name in temp_columns:
    temp_df = pd.merge(
        one_hot_df[col_name], y_true, left_index=True, right_index=True)
    sns.countplot(x="Survived", hue=col_name, data=temp_df)
    plt.show()

# %%
