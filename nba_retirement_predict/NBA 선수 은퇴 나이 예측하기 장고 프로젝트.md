# NBA 선수 은퇴 나이 예측하기 장고 프로젝트

- 우선 리액트로 구조를 잡았으니 api를 먼저 구축하여 데이터를 가져오려고 한다.

- 작업 순서는 아래와 같이 진행하려고 한다.

  1. 모델 생성하기
  2. api 생성하기
  3. 선수들 이미지 사진 저장
  4. 테스트 케이스 생성
  5. 깃허브 액션 자동화
  6. pythonanywhere연동
  7. 기타 추가 사항 중간에 삽입


### 1. models 생성하기

> TimeStampedModel는 created_at, updated_at을 사용하지 않아도 알아서 기록하는 모듈이다. 매우 유용하다.

1. Player에는 선수의 이름과 유니폼 번호를 담았다.
2. Predict
3. Image에는 선수의 정보를 외래키로 잡고, 이미지 경로를 담았다.

```python
from django.db import models
from model_utils.models import TimeStampedModel


class Player(TimeStampedModel):
    name = models.CharField(max_length=100)
    uniform_number = models.IntegerField()

#TODO: 어떤 항목이 들어가야 할 지 ....
class Predict(TimeStampedModel):
    pass


class Image(TimeStampedModel):
    player_id = models.ForeignKey(Player, related_name="player", on_delete=models.CASCADE, db_column="player_id")
    url = models.ImageField(upload_to="nba_app", null=True)
```

### 2. api 생성하기

> views에 파일을 생성하는 기준은 아래 순서와 같다. 먼저 선수의 정보를 가져와서 db에 저장하고 그 다음에 예측 알고리즘을 작동하도록 순서를 구성하였다.
>
> 크롤링와 예측은 한 번 만 실행되고 해당 정보를 저장한 다음 요청이 들어올 때 예측한 정보를 제공한다.

#### 1. 은퇴한 선수 크롤링하여 db에 저장하기

[크롤링\_into_db_방법](https://maximum-curry30.tistory.com/404)

- 위의 방법처럼 하면 아래 사진처럼 데이터가 잘 들어간 것을 볼 수 있다.

![nba_django_01](img/nba_django_01.jpg)

- 이제 선수들의 부상 정보를 삽입하고, 이미지와 등번호도 찾아서 db를 완성하자.
