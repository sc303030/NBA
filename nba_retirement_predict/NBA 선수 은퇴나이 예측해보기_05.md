# NBA 선수 은퇴나이 예측해보기_05

### 포지션 인코딩 하기

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(df_merge['position'])
digit = encoder.transform(df_merge['position'])
df_merge['position_label'] = digit
```

- 포지션이 영어라서 회귀분석 시 인식이 안되기 때문에 인코딩하여 숫자로 바꾼 후 새로운 컬럼으로 넣어준다.

### 다중회귀분석 

```python
X = df_merge.drop(['age','Relinquished','position'],axis=1)
y = df_merge['age']
```

- X,y 데이터를 나눈다. X는 독립변수고 y는 종속변수다.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=10)
```

- 트레인과 테스트로 데이터셋을 나눈다.

```python
lr_model.fit(X_train, y_train)

r_square = lr_model.score(X_test, y_r_square = lr_model.score(X_test, y_test)
print(f'결정계수 : {r_square}')
>
결정계수 : -0.011366196674897688
```

- 학습 한 후 결정계수를 보았는데 처참하다.

```python
print('기울기 : ', lr_model.coef_)
print('졀편 : ', lr_model.intercept_)
>
기울기 :  [[-6.32465742e-01 -1.75111611e-01  7.23861580e-01 -5.22521510e-02
   2.00270416e-02 -4.21996213e-02 -6.02528846e-02 -2.84698905e-01
  -3.07364264e-02  3.23139412e-01  1.32611533e-01 -8.96505609e+00
   1.27419389e+00  2.49837777e+01  1.47583025e+01 -7.27557627e+00
  -1.96844510e-01]]
졀편 :  [23.32210023]
```

```python
y_pred = lr_model.predict(X)

data_pre = pd.DataFrame({
        '예측값' : np.ravel(y_pred),
        '실제값' : y
})
data_pre.head()
```

- 예측값과 실제값을 비교해보자.

![48](./img/48.jpg)

- 생각보다는 잘 맞는것 같은데 다른 방법으로 더 자세하게 해보자.