second-derivate-pattern
---------
이차함수의 이계도함수와 2개의 함숫값을 알고 있으면, 다른 함숫값도 찾을 수 있다.

일반적인 방법 : f(x) 를 유추하여 함숫값을 찾아냄.  
새로 개발한 방법 : n차함수의 n계도함수가 n차함수의 함숫값에 어떤 영향을 끼치는지 패턴을 찾아내고, 쉽게 다른 함숫값을 찾아냄. (직접 발견한 패턴)  

Example
------
ex)  
f(x) = x^2 + 3x  
f''(x) = 2  
f(1) = 4  
f(4) = 28  

|f(1)|f(2)|f(3)|f(4) ...|
|:--:|:--:|:--:|:--:|
|4|4+t|4+2t+2|28 ...|
|t|t+2|t+4|t+6 ...|
     
4+3t+6 = 28  
3t = 18  
t = 6  

f(2) = 4 + 6 = 10  
f(3) = 4 + (2 x 6) + 2 = 18  
.  
.  
.  

위의 패턴을 공식화함.
그렇게 새로 연구하여 완성된 공식.
![image](https://user-images.githubusercontent.com/66504341/108023088-32218f00-7065-11eb-927d-c2e1db214230.png)

실제 이차함수와 새로운 공식으로 만든 함수를  
Functions 의 extract_f() 메소드를 이용하여 비교해보면 완전히 포개어지는 모습을 볼 수 있다.  
(빨강 = 새로운 공식으로 만든 함수, 파란 = 실제 함수)  
![image](https://user-images.githubusercontent.com/66504341/108022443-e6bab100-7063-11eb-91cb-e2f5fad17815.png)

``` python
from derivative_pattern import Functions

# f(x) = 2x^2 + 5x
f1 = (1, 7)  # (x, f(x))
f2 = (5, 75)  # (x, f(x))

functions = Functions(h=4)  # h=이계도함수
functions.add_func(f1)  # 첫번째 함숫값 추가
functions.add_func(f2)  # 두번째 함숫값 추가
functions.t()  # f(n+1) - f(n) = t. 증가량 구하기
print(functions.y(x=123942))  # f(123942) 의 값이 반환됨. 함수식도 반환됨.
print(functions.predict_func())  # ((x^2 의 계수, x 의 계수, 상수), 함수식) 을 반환함.
functions.extract_f(ran=range(-500, 501))  # x = -500 ~ 500 의 그래프를 반환함.
```
