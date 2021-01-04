# NBA 선수 은퇴나이 예측해보기

### 데이터 준비

1. kaggle
   1. 2020년까지 NBA Players 정보
   2. 2010-2020년까지 NBA 선수들 부상 정보
2. 위키백과
   1. 크롤링으로 2010-2020년까지 은퇴 선수 목록

### 위키백과 NBA 선수 은퇴 정보 크롤링

#### 필요한 패키지 불러오기

```python
from urllib.request import urlopen
from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.error   import HTTPError
from urllib.error   import URLError
import pandas as pd
```

#### 셀리엄 불러오기

```python
from selenium import webdriver
import time
path = './driver/chromedriver.exe'
driver = webdriver.Chrome(path)
```

#### 크롤링 함수 만들기

```python
def craw():
    time_list = ['2010-11','2011-12','2012-13','2013-14','2014-15','2015-16','2016-17','2017-18','2018-19','2019-20','2020-21']
    day_list = ['2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020']
    name_list = []
    age_list = []
    year_list = []
    cnt = 0
    for day in time_list:
        driver.get('https://en.wikipedia.org/wiki/List_of_'+str(day)+'_NBA_season_transactions')
        page = driver.find_elements_by_css_selector('.wikitable')
        page = page[0]
        for i in page.find_elements_by_tag_name('tbody'):
            k = i.find_elements_by_tag_name('tr')
            for idx,j in enumerate(k):
                if idx == 0:
                    continue
                else:
                    td_list = j.find_elements_by_tag_name('td')
                    if len(td_list) == 6: 
                        name_list.append(j.find_elements_by_tag_name('td')[1].text)
                        age_list.append(j.find_elements_by_tag_name('td')[3].text)
                        year_list.append(day_list[cnt])
                    elif len(td_list) == 5:
                        name_list.append(j.find_elements_by_tag_name('td')[0].text)
                        age_list.append(j.find_elements_by_tag_name('td')[2].text)
                        year_list.append(day_list[cnt])
            cnt +=1
            print(cnt)
    return name_list, age_list, year_list
```

1. ![01](./img/01.png)

   - 가져와야 할 데이터가 wikitable class로 되어있고 그 안에 tbody와 td, tr로 이루어져 있었다.
   - 그거에 맞게 크롤링 구조를 구성하였다.

#### 크롤링하기

```python
name_list, age_list, year_list = craw()
```

#### df로 만들기

```python
nba_df = pd.DataFrame({
    'name' : name_list,
    'age' : age_list,
    'season' : year_list
})
```

#### csv로 저장

```python
nba_df.to_csv('./data/nba_df.csv',mode='w',index=False)
```

### 분석해보기

#### 필요한 패키지 import

```python
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
import datetime 
%matplotlib inline
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
%matplotlib inline

import platform
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from matplotlib import font_manager, rc
from matplotlib import style

if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    print('Unknown system... sorry~~~~') 
```

#### 파일 불러오기

```python
nba_retire = pd.read_csv('nba_df.csv')
nba_all = pd.read_csv('all_seasons.csv')
nba_injury = pd.read_csv('injuries_2010-2020.csv')
```

#### 정보 확인 함수

```python
def info(df):
    display(df.describe())
    display(df.info())
    display(df.isna().sum())
```

- 비어있는 값이 있는지 어떤 행의 값이 어떤 type인지 확인해본다.

#### EDA

![02](./img/02.jpg)

```python
nba_all.drop('Unnamed: 0',axis=1,inplace=True)
```

- 필요없는 행이 있어서 제거

```python
nba_injury_sum = nba_injury.groupby('Relinquished', as_index=False).agg({'Notes':'count'}).sort_values('Notes',ascending=False).reset_index(drop=True)
```

- 선수들의 부상 횟수를 묶어서 저장한다.

```python
nba_player = nba_all.groupby('player_name',as_index=False).agg({'season':'count'}).sort_values('season',ascending=False).reset_index(drop=True)
```

- 선수들의 선수생활 기간을 저장한다.

```python
nba_1020_injury = pd.merge(nba_player,nba_injury_sum,left_on='player_name',right_on='Relinquished')

nba_1020_injury.drop('Relinquished',axis=1,inplace=True)
```

- 부상횟수와 선수생활 파일을 합친다.

```python
nba_retire_merge = pd.merge(nba_1020_injury, nba_retire,left_on='player_name',right_on='name')

nba_retire_merge.drop('name',axis=1,inplace=True)

nba_retire_merge.columns=['name','year','count','retire_age','retire_season']
```

- 위에서 만든 파일과 은퇴 파일을 합친다.

```python
nba_retire_merge.head()
```

![03](./img/03.jpg)

- 최종적으로 만들어진 파일은 다음과 같다.

```python
nba_retire_merge.describe()
```

- 요약해보았다.

![07](./img/07.jpg)

#### 상관관계 구해보기

```python
def corr(data,text):
    corr = data.corr(method='pearson')
    display(corr)
    style.use('ggplot')
    plt.title(text)
    sns.heatmap(data = corr, annot=True, fmt = '.2f', linewidths=.5, cmap='Blues')
```

```python
corr(nba_retire_merge,'상관관계')
```

![05](./img/05.jpg)

![04](./img/04.png)

- 생각처럼 부상횟수가 은퇴에 큰 영향을 미친다고 보기 어려웠다. 오히려 year(리그에서 생활한 기간)이 은퇴나이와 연관이 더 높았다. 

#### 부상 횟수 상위 10명 알아보기

```python
nba_retire_merge.sort_values('count',ascending=False).head(10).reset_index(drop=True)
```

![06](./img/06.jpg)

- 드웨인 웨이드는 아직 은퇴하기 아쉬울 정도로 일찍 은퇴한 감이 있었다. 많은 부상에도 불구하고 평균보다 더  많이 뛰었다. 1위와 2위는 챔피언을 경험한 선수들이다. 많은 경기를 뛴 만큼 데미지가 많았을 텐데 현대 의학의 발전덕분일까?

### 부상 정보 다시 크롤링해서 가져오기

- 2010~2020년 사이에 은퇴한 선수들의 데뷔초부터 부상정보를 가져오기 위해 다시 크롤링 하였다.
  - 기준은 빈스카터의 데뷔 연도 1998년도를 기준으로 가져왔다.

#### 필요한 패키지 import

```python
from urllib.request import urlopen
from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.error   import HTTPError
from urllib.error   import URLError

import pandas as pd

from selenium import webdriver
import time
path = '../driver/chromedriver.exe'
driver = webdriver.Chrome(path)
```

#### 크롤링 함수 만들기

```python
def craw(start,end):
    page_list = [ i for i in range(start,end,25)]
    Date = []
    Team = []
    Acquired = []
    Relinquished = []
    Notes = []
    for page in page_list:
        driver.get('http://www.prosportstransactions.com/basketball/Search/SearchResults.php?Player=&Team=&BeginDate=1998-01-01&EndDate=2020-12-31&ILChkBx=yes&Submit=Search&start='+str(page))
        page = driver.find_elements_by_css_selector('.datatable')
        if len(page) != 0:
            for i in page[0].find_elements_by_tag_name('tbody'):
                k = i.find_elements_by_tag_name('tr')
                for data in k:
                    ll = data.find_elements_by_tag_name('td')
                    Date.append(ll[0].text)
                    Team.append(ll[1].text)
                    Acquired.append(ll[2].text)
                    Relinquished.append(ll[3].text)
                    Notes.append(ll[4].text)
    return   Date,Team, Acquired,Relinquished, Notes
```

![08](./img/08.png)

- 테이블이 datatable class로 되어있어서 이걸로 tbody와 tr, td로 찾아서 저장하였다.

#### 함수 실행하기

```python
Date,Team, Acquired,Relinquished, Notes = craw(0,5001)
Date1,Team1, Acquired1,Relinquished1, Notes1 = craw(5001,10001)
Date2,Team2, Acquired2,Relinquished2, Notes2 = craw(10001,15001)
Date3,Team3, Acquired3,Relinquished3, Notes3 = craw(15000,20001)
Date4,Team4, Acquired4,Relinquished4, Notes4 = craw(20001,25001)
Date5,Team5, Acquired5,Relinquished5, Notes5 = craw(25001,28526)
```

- 25개씩  이루어져있어 start, end로 url을 맞춰주었다. 28525를 해버리니 중간에 렉이 걸려서 따로 따로 실행하였다.

#### df로 만들기

```python
df1 = pd.DataFrame({
    'Date':Date,
    'Team' : Team,
    'Acquired' : Acquired,
    'Relinquished' : Relinquished,
    'Notes' : Notes    
                   })
```

- 이런 식으로 5개의 df를 만들었다.

```python
df1.to_csv('df1.csv',mode='w',index=False)
```

- 혹시 몰라서 csv로 우선 저장하였다.

#### 중복 컬럼 제거하기

```python
nba_injury_1998 = pd.concat([df1,df2,df3,df4,df5,df6])
drop_index = list(nba_injury_1998[nba_injury_1998['Date']==' Date'].index)
nba_injury_1998 = nba_injury_1998.drop(drop_index).reset_index(drop=True)
none_Relinquished = list(nba_injury_1998[nba_injury_1998['Relinquished'] ==''].index)
nba_injury_1998 = nba_injury_1998.drop(none_Relinquished).reset_index(drop=True)
nba_injury_1998 = nba_injury_1998.drop(['Acquired'],axis=1)  
nba_injury_1998.to_csv('nba_injury_1998.csv',mode='w',index=False)
```

- 데이터들을 행으로 합치고 중간에 컬럼이 계속 중복으로 들어가서 그것의 인덱스를 찾아서 제거해준다.
- 또한 Relinquished가 비어있는 행을 지우고 Acquired 열도 지운다.

#### 정리

```python
for i in range(nba_injury_1998.shape[0]):
    if nba_injury_1998.loc[i,'Relinquished'] != '':
        nba_injury_1998.loc[i,'Relinquished'] = nba_injury_1998.loc[i,'Relinquished'].split('•')[1].strip()
        nba_injury_1998.loc[i,'Date'] = nba_injury_1998.loc[i,'Date'].strip()
        nba_injury_1998.loc[i,'Team'] = nba_injury_1998.loc[i,'Team'].strip()
        nba_injury_1998.loc[i,'Notes'] = nba_injury_1998.loc[i,'Notes'].strip()
    if nba_injury_1998.loc[i,'Relinquished'] =='':
        nba_injury_1998.loc[i,'Relinquished'] = nba_injury_1998.loc[i,'Relinquished']
        nba_injury_1998.loc[i,'Date'] = nba_injury_1998.loc[i,'Date'].strip()
        nba_injury_1998.loc[i,'Team'] = nba_injury_1998.loc[i,'Team'].strip()
        nba_injury_1998.loc[i,'Notes'] = nba_injury_1998.loc[i,'Notes'].strip()
```

- `• Elliot Williams` 데이터 앞에 기호와 띄어쓰기가 있어서 정리해주었다. 다른 행도 띄어쓰기를 정리하였다.

```python
for i in range(nba_injury_1998.shape[0]):
    data = nba_injury_1998.loc[i,'Notes'].split('with')
    print(data)
    if data[0] in  ['placed on IL ','placed on IR ']:
        nba_injury_1998.loc[i,'Notes2'] = data[1].strip()
    else:
        nba_injury_1998.loc[i,'Notes2'] = nba_injury_1998.loc[i,'Notes']
```

- 부상 앞에 placed on IR with 혹은 placed on IL with 가 있어서 뒤에것만 저장하였다.

### EDA2

```python
name_list = nba_retire.groupby(['name']).count().sort_values('age',ascending=False)
name_list = list(name_list[name_list['age'] == 2].index)
name_lis

>
['Keyon Dooling',
 'Rasheed Wallace',
 'Elton Brand',
 'Nazr Mohammed',
 'Brandon Roy',
 'Nick Collison',
 'Boštjan Nachbar']
```

- 다음의 선수들이 2개씩 있어서 이른 시즌에 있는 것들을 지우기로 하였다.

```python
index_list = []
while len(name_list) > 0:
    cnt = len(name_list)
    for idx, value in nba_retire.iterrows():
        if value[0] in name_list:
            index_list.append(idx)
            name_list.remove(value[0]) 
nba_retire = nba_retire.drop(index_list).reset_index(drop=True)
```

- 정상적으로 지워졌다.

```python
nba_player = nba_all.groupby('player_name',as_index=False).agg({'season':'count'}).sort_values('season',ascending=False).reset_index(drop=True)
```

#### 이름 바꿔주기

```python
nba_01 = pd.merge(nba_retire, nba_player, left_on='name', right_on='player_name',how='left').sort_values('season_y').reset_index(drop=True)
nba_01 = nba_01.drop(['season_x','player_name'],axis=1).rename({'season_y':'season'},axis=1)
name_list = ['Rasho Nesterovic','Zydrunas Ilgauskas','Peja Stojakovic','T.J. Ford','Eduardo Najera','Vladimir Stepania','Darko Milicic',
             'Hedo Turkoglu','Kosta Perovic','Raul Lopez','Andres Nocioni','Primoz Brezec','Bostjan Nachbar','Jiri Welsch',
            'PJ Hairston','Manu Ginobili','Mike Dunleavy','Mirza Teletovic','Gerald Henderson','Jose Calderon','Kevin Seraphin']
cnt = 0
for i in range(155,176):
    nba_01.loc[i,'name'] = name_list[cnt]
    cnt += 1
```

- 은퇴 정보와 player정보를 합쳐서 어떤 선수의 정보가 합쳐지지 않았는지 확인하고 nba_all 원래 파일에 이름을 대조하여 리스트를 만들었다.
  - 그 다음 해당 리스트의 있는 정보들을 바꾸어주었다.

```python
nba_02 = pd.merge(nba_01, nba_player, left_on='name', right_on='player_name').drop(['season_x','player_name'],axis=1).rename({'season_y':'season'},axis=1)
```

- 이름을 바꿔준 파일을 다시 merge하였다.

```python
nba_injury_sum = nba_injury.groupby('name', as_index=False).agg({'Notes':'count'}).sort_values('Notes',ascending=False).reset_index(drop=True)
```

- 부상 횟수를 합쳤을 때 이름에 전처리 해야할 것들이 많았다.

#### injury에 있는 이름 바꿔주기

```python
import re
re.split('[/)]',nba_injury_sum.loc[0,'name'])
```

- 이러한 식으로 바꿔주려고 한다.