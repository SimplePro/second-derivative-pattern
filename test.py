from derivative_pattern import Functions

# x^2 + 3x - 2
# f``(x) = 2
# f(1) = 2
# f(-3) = -2

f1 = (-3, -2)
f2 = (1, 2)
functions = Functions(h=2)  # h=이계도함수
functions.add_func(f1)  # 첫번째 함숫값 추가
functions.add_func(f2)  # 두번째 함숫값 추가
functions.t()  # f(n+1) - f(n) = t. 증가량 구하기
print(functions.y(x=48124))  # f(48124) 의 값이 반환됨. 함수식도 반환됨.
print(functions.predict_func())  # ((x^2 의 계수, x 의 계수, 상수), 예측된 이차함수식) 을 반환함.
functions.extract_f(ran=range(-100, 101))  # x = -100 ~ 100 의 그래프를 반환함.
