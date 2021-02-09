# NBA 선수 은퇴나이 예측해보기_04

- 지금까지 했던건 이진분류가 계속 정확도가 낮은 것으로 예측하기보다 우선 변수를 추가하기로 했다.
- 십자인대, 아킬레스 부상이 대표적으로 선수들의 은퇴를 앞당기는 부상이다. 그래서 십자인대,  아킬레스로 컬럼을 만들어 해당 부상을 당했으면 1, 아니면 0으로 부여하려고 한다.
  - 십자인대 부상 종류 
  - 출처 : ROOKIE(http://www.rookie.co.kr)
    - ACL Injury 전방십자인대 부상 
    - PCL Injury 후방십자인대 부상
    - MCL Injury 내측측부인대 부상
    - LCL Injury 외측측부인대 부상
- 부상내역중에 4개의 단어가 들어가면 십자인대로 아킬레스는 Achilles Tendon Rupture가 들어가면 카운트하기로 결정하였다.
- 그리고 out for season도 따로 컬럼으로 만들어서 넣기로 하였다. 그만큼 운동능력에 영향을 주는 부상이라고 판단하였다.

### 필요한 패키지 import 

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

### 십자인대, 아켈레스 부상 여부

```python
df = pd.read_csv('nba_injury_1998.csv')

def yesno(x):
    words = x.split(' ')
    print(words)
    for word in words:
        if word.upper() in ['ACL', 'PCL', 'ACHILLES']:
            return True
            break   
df['tf'] = df['Notes2'].apply(lambda x:yesno(x))
```

![38](./img/38.jpg)

- 이렇게 분리해서 for문을 돌리면서 ['ACL', 'PCL', 'ACHILLES']가 있는지 확인하고 있으면 True를 리턴하고 나온다. 편리하게 모두 대문자로 바꿔서 비교하였다.

### 시즌아웃 부상 여부

```python
import re

# 시즌아웃 부상
def seasonout(x):
    words = re.split('\(|\)', x)
    print(words)
    for word in words:
        if word in ['out for season']:
            return True
            break
    
df['out']  = df['Notes2'].apply(lambda x:seasonout(x))
```

![39](./img/39.jpg)

- 이렇게 out for season가 있으면 또한 Ture를 리턴하고 나온다.

```python
display(df[(df['out'] == True) | (df['tf'] == True)])
df[(df['out'] == True) | (df['tf'] == True)].shape
>
(796, 7)
```

- 총 796명의 선수가 나왔다.

![40](./img/40.jpg)

- 이렇게 둘다 True인 선수도 있고 아닌 선수도 있다. 저기서 파열인 경우만 분리하자.