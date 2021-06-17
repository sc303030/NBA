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



