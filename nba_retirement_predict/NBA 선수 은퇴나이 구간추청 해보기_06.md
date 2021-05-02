# NBA 선수 은퇴나이 추청 해보기_06

- 예측이라는 단어는 정확하게 맞추기 어렵다하여 추청으로 바꿔서 진행하려고 한다.

```python
model = keras.Sequential([
    layers.Dense(50, activation='relu', input_shape=[len(train_set.keys()) +1]),
    layers.Dense(50, activation='relu'),
    layers.Dense(1)
])

optimizer = tf.keras.optimizers.RMSprop()
model.compile(loss='mse', optimizer = optimizer, metrics=['mae', 'mse'])

model.summary()xxxxxxxxxx     layers.Dense(50, activation='relu', input_shape=[len(train_set.keys()) +1]),    layers.Dense(50, activation='relu'),    layers.Dense(1)])optimizer = tf.keras.optimizers.RMSprop()model.compile(loss='mse', optimizer = optimizer, metrics=['mae', 'mse'])model.summary()model = keras.Sequential([    layers.Dense(50, activation='relu', input_shape=[len(train_set.keys()) +1]),    layers.Dense(50, activation='relu'),    layers.Dense(1)])optimizer = tf.keras.optimizers.RMSprop()model.compile(loss='mse', optimizer = optimizer, metrics=['mae', 'mse'])model.summary()
```

- 계속 오류났던 부분이 `input_shape=[len(train_set.keys()) +1])` 여기였다.
- 보면 predict하려는 df의 열 개수는19개인데 input_shape가 18이여서 오류가 난것이다. 이걸 바꾸고 실행했다.

```python
sample_result = model.predict(norm_train_set[:10])
sample_result
>
array([[nan],
       [nan],
       [nan],
       [nan],
       [nan],
       [nan],
       [nan],
       [nan],
       [nan],
       [nan]], dtype=float32)
```

- 이번에는 아무 값도 나오지 않는 현상이 발생하였다. 이번에는 이걸 해결하자....