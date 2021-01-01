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

