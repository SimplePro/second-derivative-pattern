import matplotlib.pyplot as plt
import numpy as np


# 시그마 메소드.
def sigma(i, k, h):
    result = 0

    for j in range(i, k + 1):
        result += h * j

    return result


# 지정된 변수의 계수를 반환하는 메소드.
def extract_coefficient(s, variable):
    result = ""
    index = s.index(variable)

    for i in range(index - 1, -1, -1):
        if s[i] == " ":
            break

        result += s[i]

    result = float(''.join(list(reversed(result))))
    return result


# 최대공약수
def euclid_numbering(a, b):
    if b > a:
        b, a = a, b
    while b != 0:
        c = b
        b = a % b
        a = c
    return a


# 최소공배수
def least_common_multiple(a, b):
    return a * b / euclid_numbering(a, b)


# 3차 연립방정식 메소드.
def simultaneous_equation(expressions, variables):
    coefficients = []  # 계수
    result = {}  # 결과
    calc_expressions = []  # 연립한 두 식

    # print("first:", expressions[0])
    # print("second:", expressions[1])
    # print("third:", expressions[2])

    # 문자별 계수만 추출.
    for i in expressions:
        coefficient = []

        for j in variables:
            coefficient.append(extract_coefficient(i, j))

        coefficient.append(float(i[i.index("=") + 1:]))
        coefficients.append(coefficient)

    # np.array 로 변환
    coefficients = np.array(coefficients, dtype=np.ndarray)

    # 각 순서의 식에 계수 추출한 값
    first, second, third = coefficients

    # print("first:", first)
    # print("second:", second)
    # print("third:", third)

    # 첫번째 식과 두번째 식 연립.
    calc_expressions.append(list(first * (least_common_multiple(first[0], second[0]) / first[0])
                                 - second * (least_common_multiple(first[0], second[0]) / second[0])))

    # 첫번째 식과 세번째 식 연립.
    calc_expressions.append(list(first * (least_common_multiple(first[0], third[0]) / first[0])
                                 - third * (least_common_multiple(first[0], third[0]) / third[0])))

    fe = calc_expressions[0]  # fe = first equation
    se = calc_expressions[1]  # se = second equation

    # print("fe(calc_expressions[0]):", fe)
    # print("se(calc_expressions[1]):", se)

    # 만일 두개의 문자의 계수가 0이라면.
    if fe[:3].count(0.0) == 2 or se[:3].count(0.0) == 2:
        nonzero_variable = 0

        # 계수가 0이 아닌 index 를 찾고, 그 문자를 구함.
        if fe[:3].count(0.0) == 2:
            nonzero_variable = [i for i, d in enumerate(fe[:3]) if d != 0.0][0]
            result[variables[nonzero_variable]] = fe[-1] / fe[nonzero_variable]

        elif se[:3].count(0.0) == 2:
            nonzero_variable = [i for i, d in enumerate(se[:3]) if d != 0.0][0]
            result[variables[nonzero_variable]] = se[-1] / se[nonzero_variable]

        # print("nonzero_variable:", nonzero_variable)
        # print(f"{variables[nonzero_variable]} is {result[variables[nonzero_variable]]}")

        # 구한 문자를 그 전의 식에 반영하여 이차연립방정식으로 만듦.
        first[3] += (-1 * first[nonzero_variable] * result[variables[nonzero_variable]])
        first = np.delete(first, nonzero_variable)

        second[3] += (-1 * second[nonzero_variable] * result[variables[nonzero_variable]])
        second = np.delete(second, nonzero_variable)

        third[3] += (-1 * third[nonzero_variable] * result[variables[nonzero_variable]])
        third = np.delete(third, nonzero_variable)

        # print("first:", first)
        # print("second:", second)
        # print("third:", third)

        # 구했던 문자의 index 가 0이였다면
        if nonzero_variable == 0:

            # 첫번째식과 두번째식이 연립하지 못하는 상황이라면, 첫번째식과 세번째식 연립
            if (np.array(first) - np.array(second)).tolist().count(0.0) == 3:
                equation = np.array([(least_common_multiple(first[0], third[0]) / first[0]) * i for i in first]) \
                           - np.array([(least_common_multiple(first[0], third[0]) / third[0]) * i for i in third])

            # 첫번째식과 세번째식이 연립하지 못하는 상황이라면, 첫번째식과 두번째식 연립
            elif (np.array(first) - np.array(third)).tolist().count(0.0) == 3:
                equation = np.array([(least_common_multiple(first[0], second[0]) / first[0]) * i for i in first]) \
                           - np.array([(least_common_multiple(first[0], second[0]) / second[0]) * i for i in second])

            # 그것도 아니라면, 두번째식과 세번째식 연립
            else:
                equation = np.array([(least_common_multiple(second[0], third[0]) / second[0]) * i for i in second]) \
                           - np.array([(least_common_multiple(second[0], third[0]) / third[0]) * i for i in third])

            # print("equation:", equation)

            result[variables[2]] = equation[2] / equation[1]
            result[variables[1]] = ((-1 * first[1] * result[variables[2]]) + first[2]) / first[0]

        # 구했던 문자의 index 1 이였다면
        elif nonzero_variable == 1:

            # 첫번째식과 두번째식이 연립하지 못하는 상황이라면, 첫번째식과 세번째식 연립
            if (np.array(first) - np.array(second)).tolist().count(0.0) == 3:
                equation = np.array([(least_common_multiple(first[0], third[0]) / first[0]) * i for i in first]) \
                           - np.array([(least_common_multiple(first[0], third[0]) / third[0]) * i for i in third])

            # 첫번째식과 세번째식이 연립하지 못하는 상황이라면, 첫번째식과 두번째식 연립
            elif (np.array(first) - np.array(third)).tolist().count(0.0) == 3:
                equation = np.array([(least_common_multiple(first[0], second[0]) / first[0]) * i for i in first]) \
                           - np.array([(least_common_multiple(first[0], second[0]) / second[0]) * i for i in second])

            # 그것도 아니라면, 두번째식과 세번째식 연립
            else:
                equation = np.array([(least_common_multiple(second[0], third[0]) / second[0]) * i for i in second]) \
                           - np.array([(least_common_multiple(second[0], third[0]) / third[0]) * i for i in third])

            # print("equation:", equation)

            result[variables[2]] = equation[2] / equation[1]
            result[variables[0]] = ((-1 * second[1] * result[variables[2]]) + second[2]) / second[0]

        # 구했던 문자의 index 가 2 였다면
        elif nonzero_variable == 2:

            # 첫번째식과 두번째식이 연립하지 못하는 상황이라면, 첫번째식과 세번째식 연립
            if (np.array(first) - np.array(second)).tolist().count(0.0) == 3:
                equation = np.array([(least_common_multiple(first[0], third[0]) / first[0]) * i for i in first]) \
                           - np.array([(least_common_multiple(first[0], third[0]) / third[0]) * i for i in third])

            # 첫번째식과 세번째식이 연립하지 못하는 상황이라면, 첫번째식과 두번째식 연립
            elif (np.array(first) - np.array(third)).tolist().count(0.0) == 3:
                equation = np.array([(least_common_multiple(first[0], second[0]) / first[0]) * i for i in first]) \
                           - np.array([(least_common_multiple(first[0], second[0]) / second[0]) * i for i in second])

            # 그것도 아니라면, 두번째식과 세번째식 연립
            else:
                equation = np.array([(least_common_multiple(second[0], third[0]) / second[0]) * i for i in second]) \
                           - np.array([(least_common_multiple(second[0], third[0]) / third[0]) * i for i in third])

            # print("equation:", equation)

            result[variables[1]] = equation[2] / equation[1]
            result[variables[0]] = ((-1 * first[1] * result[variables[1]]) + first[2]) / first[0]

    # 만일 한개의 문자의 계수가 0이라면.
    else:
        # 첫번째식과 두번째식을 연립함.
        equation = np.array([(least_common_multiple(fe[2], se[2]) / fe[2]) * i for i in fe]) - np.array(
                                [(least_common_multiple(fe[2], se[2]) / se[2]) * i for i in se])

        # 두번째 문자의 값을 구함.
        result[variables[1]] = equation[3] / equation[1]

        # 세번째 문자의 값을 구함.
        result[variables[2]] = (-1 * fe[1] * result[variables[1]] + fe[3]) / fe[2]

        # 첫번째 문자의 값을 구함.
        result[variables[0]] = ((-1 * first[1] * result[variables[1]]) + (-1 * first[2] * result[variables[2]]) + first[3]) / first[0]

    return result


# 연립방정식을 할 수 있도록 식의 형식을 바꿔주는 메소드.
def to_simultaneous(coefficients, variables, value):

    result = ""
    for c, v in zip(coefficients, variables):
        if int(c) > 0:
            result += f" +{c}{v}"

        elif int(c) < 0:
            result += f" {c}{v}"

        elif int(c) == 0:
            raise Exception("c must not be zero")

    result += f"={value}"
    return result


class Functions:
    def __init__(self, h):
        self.second_derivative = h  # 이계도함수
        self.increase = None  # 증가량 (t)
        self.func = []  # 함숫값을 담는 변수
        self.coefficient_t = 0  # t 의 계수

    # function 을 추가하는 메소드.
    def add_func(self, f):
        if len(self.func) == 0:
            self.func.append(f)

        elif self.func[-1][0] >= f[0]:
            raise Exception("f[0] not be smaller than functions[-1][0]")
        else:
            self.func.append(f)

    # 입력된 index 의 function 을 제거하는 메소드.
    def delete_func(self, index):
        del self.func[index]

    # increase (t) 를 구하는 메소드.
    def t(self):
        # f``(x) = h, f(n) = a, f(m) = b (m - n > 1) # 전제조건
        # t = (b - a - sigma(i=1, m-n-1, hi)) / (m - n)  # t = f(n+1) - f(n). 증가량

        if len(self.func) < 2:
            raise Exception("len(func) not be smaller than 2")

        n, a = self.func[0]
        m, b = self.func[1]

        self.increase = (b - a - sigma(i=1, k=m - n - 1, h=self.second_derivative)) / (m - n)
        return self.increase

    # 입력된 x 값의 함숫값을 구하는 메소드.
    def y(self, x):
        # f``(x) = h, f(n) = a, t = f(n+1) - f(n)  (전제조건)
        # f(x) = a + (x-n)t + sigma(i=1, |x-n-1|, hi)  (x > n)
        # f(x) = a + (x-n)t + sigma(i=1, |x-n|, hi)  (x < n)

        n, a = self.func[0]

        if n == x:
            return a

        self.coefficient_t = x - n
        c = 0

        if n > x:
            c = sigma(i=1, k=abs(x - n), h=self.second_derivative)

        elif n < x:
            c = sigma(i=1, k=abs(x - n - 1), h=self.second_derivative)

        result = a + (self.coefficient_t * self.increase) + c
        return result, f"{a} + ({self.coefficient_t}*{self.increase}) + ({c})"

    # 함수 그래프 구하는 메소드.
    def extract_f(self, ran):
        if self.increase is None:
            raise Exception("increase (t) must not be None.")

        x = list(ran)
        y = []

        n, a = self.func[0]

        for i in x:
            c = 0

            if i > n:
                c = sigma(i=1, k=abs(i - n - 1), h=self.second_derivative)

            elif i < n:
                c = sigma(i=1, k=abs(i - n), h=self.second_derivative)

            y.append(a + ((i - n) * self.increase) + c)

        plt.plot(x, y, linewidth=2.5, color='red')
        # plt.scatter(list(range(list(ran)[0], list(ran)[-1]+1, 5)), [2*(i**2) + 5*i for i in range(list(ran)[0],
        #             list(ran)[-1]+1, 5)], color='blue')
        plt.show()

    # 원래의 이차함수 식을 예측하는 메소드.
    # f(x) = ax^2 + bx + c
    def predict_func(self):
        mid = (self.func[1][0] + self.func[0][0]) // 2
        if mid == 0:
            mid = self.func[1][0] + 3
        mid_y = self.y(x=mid)[0]
        coefficients = []  # 계수값들
        values = []  # y값
        expressions = []

        for i in [self.func[0], (mid, mid_y), self.func[1]]:
            coefficients.append([i[0] ** 2, i[0], 1])
            values.append(i[1])

        for c, v in zip(coefficients, values):
            expressions.append(to_simultaneous(coefficients=c, variables=["a", "b", "c"], value=v))

        print(expressions)
        result = simultaneous_equation(expressions=expressions, variables=["a", "b", "c"])
        expression = f'{result["a"]}x^2 + {result["b"]}x + {result["c"]}'

        return result, expression


if __name__ == '__main__':
    # f(x) = 2x^2 + 5x
    f1 = (1, 7)  # (x, f(x))
    f2 = (5, 75)  # (x, f(x))

    functions = Functions(h=4)  # h=이계도함수
    functions.add_func(f1)  # 첫번째 함숫값 추가
    functions.add_func(f2)  # 두번째 함숫값 추가
    functions.t()  # f(n+1) - f(n) = t. 증가량 구하기
    print(functions.y(x=123942))  # f(123942) 의 값이 반환됨. 함수식도 반환됨.
    print(functions.predict_func())  # ((x^2 의 계수, x 의 계수, 상수), 예측된 함수식) 을 반환함.
    functions.extract_f(ran=range(-500, 501))  # x = -500 ~ 500 의 그래프를 반환함.
