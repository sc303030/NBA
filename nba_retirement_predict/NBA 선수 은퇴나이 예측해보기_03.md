# NBA 선수 은퇴나이 예측해보기_03

### 포지션 크롤링

- nba-api에서 포지션을 뽑아오려고 했는데 아무리 찾아봐도 안 보여서 그냥 홈페이지에서 크롤링하기로 하였다.

```python
position_name = list(nba_injury_merge['name'])
position_name
```

- 우선 선수이름을 리스트로 만든다.

```python
position_dic = {}
for name in position_name:
    position_dic[name] = 0
position_dic
```

- 그 다음에 딕셔너리로 만들어서 관리한다.

```python
driver.get('https://www.nba.com/players')
page = driver.find_elements_by_css_selector('.Toggle_slider__hCMQQ')[0].click()
key_words = driver.find_elements_by_css_selector('.Input_input__3YQfM')[0]
for names in list(position_dic.keys()):
    try:
        key_words.send_keys(names)
        time.sleep(1)
        position_table = driver.find_elements_by_css_selector('.players-list')[0]
        position_tr = position_table.find_elements_by_tag_name('tr')[1]
        position_td = position_tr.find_elements_by_tag_name('td')[3].text
        position_dic[names] = position_td
        print(position_td)
        key_words.clear()
        time.sleep(1)
    except:
        position_dic[names] = 0
```

- 중간에 오류가 나서 try랑 except로 관리하였다.

![20](./img/20.jpg)

- 모든 선수들을 보려면 저 버튼을 클릭해야 한다.

`page = driver.find_elements_by_css_selector('.Toggle_slider__hCMQQ')[0].click()` 그래서 찾아서 클릭한다.

![21](./img/21.jpg)

- 그 다음에 저기에 선수이름을 입력하면 선수의 목록이 뜬다.

```python
# 인풋태그 찾기
key_words = driver.find_elements_by_css_selector('.Input_input__3YQfM')[0]
# 선수이름을 for문으로 돌려서 입력하기
for names in list(position_dic.keys()):
    try:
        #선수 이름을 input에 입력해서 보내기
        key_words.send_keys(names)
        time.sleep(1)
```

![23](./img/23.jpg)

- 그런다음 포지션이 있는 태그를 찾아서 text만 받아서 딕셔너리를 업데이트 시킨다.

```python
		position_tr = position_table.find_elements_by_tag_name('tr')[1]
    	# 포지션 인덱스 찾기
        position_td = position_tr.find_elements_by_tag_name('td')[3].text
        # 딕셔너리 업데이트
        position_dic[names] = position_td
        print(position_td)
        # 다음 선수입력을 위해 원래있던 이름 지우기
        key_words.clear()
        time.sleep(1)
    except:
        position_dic[names] = 0
```

![19](./img/19.gif)

- 작동시키면 이렇게 진행된다.

```python
for key, value in position_dic.items():
    if value == 0:
        print(key)
```

- 마지막으로 입력안된 선수가 있는지 최종확인한다.

#### df와 합치기

```python
nba_injury_merge['position'] = nba_injury_merge['name'].apply(lambda x:posi(x,position_dic))
nba_injury_merge
```

![24](./img/24.jpg)

- 이렇게 업데이트 되었다. csv로 저장하자.

```python
nba_injury_merge.to_csv('nba_injury_merge_position.csv',mode='w',index=False)
```

### 포지션 라벨링

#### 필요한 패키지 import

```python
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.tree     import DecisionTreeClassifier
from sklearn.model_selection import  GridSearchCV,train_test_split
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
```

```python
item_label  = list(injury_df.groupby('position').agg({'position':'count'}).index)

encoder = LabelEncoder()
encoder.fit(item_label)

digit_label = encoder.transform(item_label)
print('encoder 결과', digit_label)

print('decoder 결과', encoder.inverse_transform(digit_label))

digit_label = digit_label.reshape(-1,1)
print(digit_label)
print(digit_label.shape)
>
encoder 결과 [0 1 2 3 4 5 6]
decoder 결과 ['C' 'C-F' 'F' 'F-C' 'F-G' 'G' 'G-F']
[[0]
 [1]
 [2]
 [3]
 [4]
 [5]
 [6]]
(7, 1)
```

- 이렇게 하면 포지션 영어 순서대로 숫자 라벨링이 적용된다.

```python
ont_hot_encoder = OneHotEncoder()
ont_hot_encoder.fit(digit_label)
ont_hot_label = ont_hot_encoder.transform(digit_label)
print(ont_hot_label.toarray())
print(ont_hot_label.shape)
>
[[1. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 0. 0. 1.]]
(7, 7)
```

- 이렇게 하면 행렬로 채워진다.

```python
pd.get_dummies(injury_df)
```

![25](./img/25.jpg)

- 선수별로 해당되는 곳만 1이 채워진다. 

### 포지션 추가해서 상관관계 보기

```python
def posi_digt(x,item_label,digit_label):
    for idx, value in enumerate(item_label):
        if x == value:
            return digit_label[idx][0]
```

```python
injury_df['position_digtt'] = injury_df['position'].apply(lambda x:posi_digt(x,item_label,digit_label))
injury_df.head()
```

- 새로운 컬럼으로 추가한다.

![26](./img/26.jpg)

```python
def corr(data,text):
    corr = data.corr(method='pearson')
    display(corr)
    style.use('ggplot')
    plt.title(text)
    sns.heatmap(data = corr, annot=True, fmt = '.2f', linewidths=.5, cmap='Blues')
```

```python
corr(injury_df,'상관관계')
```

![27](./img/27.jpg)

- 거의 상관없다고 나온다. 다음에는 파라미터를 수정해서 머신러닝을 돌려보자.