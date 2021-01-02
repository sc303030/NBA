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

