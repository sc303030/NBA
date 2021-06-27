# NBA 선수 은퇴나이 추청 해보기_07

### 클래스 만들기

#### 아킬레스, 십자인대 df 만들기

```python
class AclAndAchilles:
    def __init__(self, df):
        self.df = df
        self.yes_no_df()
        self.seasonout_df()
        self.yes_no2_df()
        self.one_or_zero_df()
        
    # 우선은 ACL과 아킬레스가 들어간  선수 구분
    @staticmethod
    def yes_no(x):
        words = x.split(' ')
        print(words)
        for word in words:
            if word.upper() in ['ACL', 'PCL', 'ACHILLES']:
                return True
                break   
                
    def yes_no_df(self):
        self.df['tf'] =  self.df['Notes2'].apply(lambda x:self.yes_no(x))
        
    # 시즌아웃 부상
    @staticmethod
    def seasonout(x):
        words = re.split('\(|\)', x)
        print(words)
        for word in words:
            if word in ['out for season']:
                return True
                break
                
    def seasonout_df(self):
        self.df['out']  = self.df['Notes2'].apply(lambda x:self.seasonout(x))
            
    # 횟수 카운트
    @staticmethod
    def yes_no2(x):
        words = x.split(' ')
        print(words)
        sum_sum = 0
        for word in words:
            if word.upper() in ['ACL', 'PCL', 'ACHILLES'] or word.upper() in['TORN','RUPTURE']:
                sum_sum += 1
            if sum_sum >= 2:
                return True
                break   
            
    def yes_no2_df(self):
        self.df['tf2'] = self.df['Notes2'].apply(lambda x:self.yes_no2(x))
        
    # 아킬레스와 십자인대 부상 전적 여부
    @staticmethod
    def one_or_zero(x):
        two=0
        three=0
        if x['out'] == True:
            two = 1
        if  x['tf2'] == True:
            three = 1
        print(two, three)    
        return  pd.Series([two,three])
    
    def one_or_zero_df(self):
        self.df[['outnum','tf2num']] = self.df[['out','tf2']].apply(self.one_or_zero,axis=1)
        
    def df1(self):
        return  self.df.groupby('Relinquished',as_index=False).agg({'outnum':'sum','tf2num':'sum'})
```

- 기존에 하나씩 썼던 함수들을 클래스로 묶어서 한 번에 실행하려고 한다.
- self를 받지 않는 함수들은 @staticmethod를 달아줘야 한다.

```python
df = pd.read_csv('nba_injury_1998.csv')
test1 = AclAndAchilles(df)
test2 =test1.df1()
test2.head()
```

|      |          Relinquished | outnum | tf2num |
| ---: | --------------------: | -----: | -----: |
|    0 |    (James) Mike Scott |      0 |      0 |
|    1 | (William) Tony Parker |      2 |      0 |
|    2 |                 76ers |      0 |      0 |
|    3 |         A.J. Bramlett |      0 |      0 |
|    4 |           A.J. Guyton |      0 |      0 |

#### 다른 데이터와 합치기

```python
class injury:
    def __init__(self, acl_achilles_df, injury_df, nba_all_df):
        self.acl_achilles_df = acl_achilles_df
        self.injury_df = injury_df
        self.nba_all_df = nba_all_df
        self.merge()
        self.age_func()
        
    def merge(self):
        self.df_merge1 = pd.merge(self.acl_achilles_df,self.injury_df,left_on='Relinquished',right_on='name').drop('name',axis=1)
        
    def age_func(self):
        # 평균을 구하고 모두 소수 2번쨰까지만 살리기
        self.nba_all_group = self.nba_all_d.groupby('player_name',as_index=False).mean()
        for i in range(self.nba_all_group.shape[0]):
            for i2 in range(len(list(self.nba_all_group.columns))):
                if i2 == 0:
                    continue
                elif i2 == 1:
                    self.nba_all_group.iloc[i,i2] = self.nba_all_group.iloc[i,i2].astype('int64')
                else:
                    self.nba_all_group.iloc[i,i2] = round(self.nba_all_group.iloc[i,i2],2)

        self.nba_all_group['age'] = self.nba_all_group['age'].astype('int64')
        
    def final_df(self):
        self.df_merge_final = pd.merge(df_merge1,nba_all_group,left_on='Relinquished',right_on='player_name',how='left').\
                                                            drop('age_y',axis=1).rename(columns={'age_x':'age'})

        self.df_merge_final.drop('player_name',axis=1,inplace=True)

        self.df_merge_final['Notes'] = self.df_merge_final['Notes'].astype(int)
        self.df_merge_final.head()
```

```python
injury_df = pd.read_csv('nba_injury_merge_position.csv')
nba_all = pd.read_csv('all_seasons.csv').drop('Unnamed: 0',axis=1)
test3 = injury(test2, injury_df, nba_all)
```

```python
df_final = test3.final_df()
df_final.head()
```

|      |  Relinquished | outnum | tf2num |  age | season | Notes | position | player_height | player_weight |    gp |   pts |  reb |  ast | net_rating | oreb_pct | dreb_pct | usg_pct | ts_pct | ast_pct |
| ---: | ------------: | -----: | -----: | ---: | -----: | ----: | -------: | ------------: | ------------: | ----: | ----: | ---: | ---: | ---------: | -------: | -------: | ------: | -----: | ------: |
|    0 |  Aaron Brooks |      0 |      0 |   35 |     10 |     9 |        G |        182.88 |         73.03 | 64.50 |  8.88 | 1.55 | 2.76 |      -3.31 |     0.02 |     0.07 |    0.22 |   0.52 |    0.23 |
|    1 |    Aaron Gray |      0 |      0 |   30 |      7 |    14 |        C |        213.36 |        122.47 | 45.43 |  3.24 | 3.73 | 0.66 |      -4.73 |     0.13 |     0.23 |    0.15 |   0.53 |    0.09 |
|    2 | Adam Morrison |      1 |      1 |   29 |      3 |     8 |        F |        203.20 |         92.99 | 53.67 |  6.07 | 1.80 | 1.17 |      -7.83 |     0.03 |     0.10 |    0.19 |   0.44 |    0.11 |
|    3 |  Adonal Foyle |      1 |      0 |   35 |     12 |    18 |        C |        208.28 |        118.88 | 61.08 |  3.78 | 4.48 | 0.44 |      -4.40 |     0.11 |     0.19 |    0.13 |   0.50 |    0.04 |
|    4 | Al Harrington |      0 |      1 |   35 |     16 |    15 |        F |        205.74 |        112.49 | 61.31 | 12.20 | 5.13 | 1.52 |      -1.45 |     0.06 |     0.17 |    0.23 |   0.51 |    0.10 |

- 잘 작동한다.

### Tensorflow 클래스 만들기

#### import

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pandas as pd
import numpy as np
import seaborn as sns

print(tf.__version__)
```

- 필요한 패키지들을 import 한다.
- 맨 처음에 같이 불러오도록 맨 위에 다시 붙여넣기 한다.

```python
from sklearn.preprocessing import LabelEncoder
# object인 컬럼만 찾기
df_final.info()
class Encoder_df:
        def __init__(self, df):
            self.df = df
            self.digit_label_Relinquished, self.digit_label_position = self.labelencoder()
            self.df_new = self.label_add_colums()
    
        def labelencoder(self):
            self.encoder = LabelEncoder()
            self.encoder.fit(list(self.df['Relinquished']))
            self.digit_label_Relinquished = self.encoder.transform(self.df['Relinquished'])

            self.encoder.fit(list(self.df['position']))
            self.digit_label_position = self.encoder.transform(self.df['position'])
            return self.digit_label_Relinquished, self.digit_label_position
             
        def label_add_colums(self):
             # 새로운 컬럼으로 넣어주기
            self.df['Relinquished_digit'] = self.digit_label_Relinquished
            self.df['position_digit'] = self.digit_label_position
            self.df_new  = self.df.drop(['Relinquished','position'],axis=1)
            self.df_new.head()
            return self.df_new
             
        def tensorflow(self):
            self.train_set = self.df_new.sample(frac=.8, random_state=0)
            self.test_set = self.df_new.drop(self.train_set.index)
            print(self.test_set)
```

```python
tensor = Encoder_df(df_final)
```

- 중간까지 했을 때 잘 작동한다.
