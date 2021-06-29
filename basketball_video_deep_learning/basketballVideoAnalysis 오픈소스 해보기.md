# basketballVideoAnalysis 오픈소스 해보기

https://github.com/stephanj/basketballVideoAnalysis

- 해당 깃허브에 있는 거 분석하고 해보기로 함

## color-detection

![17](./img2/17.jpg)

```python
#!/bin/bash
python show_colors.py -p images
```

- run.sh 파일을 실행하면 show_colors.py가 실행됨

- path는 이미지가 있는 폴더 지정 

  - image2라는 폴더를 새로 생성하고 path 변경

  - ```python
    #!/bin/bash
    python show_colors.py -p images2ㅇ
    ```

  - ![team_a_1](./img2/team_a_1.jpg)

  - 우선 한 장으로 시도

```
team,red,green,blue,percentage

team_a_1.jpg,1,0,1,67

team_a_1.jpg,86,99,141,20

team_a_1.jpg,157,165,201,12
```

- 다음과 같은 결과값이 csv 파일로 저장됨

- 5명씩 만들어서 진행해보려고 함
