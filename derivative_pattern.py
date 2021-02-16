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
    coefficients = []
    result = {}
    calc_expressions = []

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

    print(coefficients)

    # 첫번째 식과 두번째 식 연립.
    calc_expressions.append(list(first * (least_common_multiple(first[0], second[0]) / first[0])
                                 - second * (least_common_multiple(first[0], second[0]) / second[0])))

    # 첫번째 식과 세번째 식 연립.
    calc_expressions.append(list(first * (least_common_multiple(first[0], third[0]) / first[0])
                                 - third * (least_common_multiple(first[0], third[0]) / third[0])))

    fe = calc_expressions[0]  # fe = first equation
    se = calc_expressions[1]  # se = second equation

    print(fe, se)
    # 만일 두개의 문자의 계수가 0이라면.
    if fe[:3].count(0.0) == 2 or se[:3].count(0.0) == 2:
        nonzero_variable = 0

        if fe[:3].count(0.0) == 2:
            nonzero_variable = [i for i, d in enumerate(fe[:3]) if d != 0.0][0]
            # nonzero_variable = np.nonzero(fe[:3] != 0.0)[0][0]
            result[variables[nonzero_variable]] = fe[-1] / fe[nonzero_variable]

        elif se[:3].count(0.0) == 2:
            nonzero_variable = [i for i, d in enumerate(se[:3]) if d != 0.0][0]
            # nonzero_variable = np.nonzero(se[:3] != 0.0)[0][0]
            result[variables[nonzero_variable]] = se[-1] / se[nonzero_variable]

        print(nonzero_variable)

        first[3] += (-1 * first[nonzero_variable] * result[variables[nonzero_variable]])
        first = np.delete(first, nonzero_variable)

        second[3] += (-1 * second[nonzero_variable] * result[variables[nonzero_variable]])
        second = np.delete(second, nonzero_variable)

        print(first, second)

        equation1 = np.array([(least_common_multiple(first[0], second[0]) / first[0]) * i for i in first]) - np.array(
            [(least_common_multiple(first[0], second[0]) / second[0]) * i for i in second])

        print(equation1)

        if nonzero_variable == 0:
            pass
        elif nonzero_variable == 1:
            pass
        elif nonzero_variable == 2:
            pass

    else:
        b_equation = np.array([(least_common_multiple(fe[2], se[2]) / fe[2]) * i for i in fe]) - np.array(
                                [(least_common_multiple(fe[2], se[2]) / se[2]) * i for i in se])
        print(b_equation)

        result[variables[1]] = b_equation[3] / b_equation[1]
        result[variables[2]] = (-1 * fe[1] * result[variables[1]] + fe[3]) / fe[2]
        result[variables[0]] = ((-1 * first[1] * result[variables[1]]) + (-1 * first[2] * result[variables[2]]) + first[3]) / first[0]

    print(result)


# simultaneous_equation(expressions=["+1a +1b -1c=0", "+2a -1b +3c=9", "1a +2b +1c=8"], variables=["a", "b", "c"])
# simultaneous_equation(expressions=["+1a +2b -3c=0", "+2a -1b -6c=9", "1a +3b +1c=8"], variables=["a", "b", "c"])


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

    def predict_func(self):
        pass


if __name__ == '__main__':
    # f(x) = 2x^2 + 5x
    f1 = (1, 7)  # (x, f(x))
    f2 = (5, 75)  # (x, f(x))

    functions = Functions(h=4)  # h=이계도함수
    functions.add_func(f1)  # 첫번째 함숫값 추가
    functions.add_func(f2)  # 두번째 함숫값 추가
    functions.t()  # f(n+1) - f(n) = t. 증가량 구하기
    print(functions.y(x=123942))  # f(123942) 의 값이 반환됨. 함수식도 반환됨.
    functions.extract_f(ran=range(-500, 501))  # x = -100 ~ 100 의 그래프를 반환함.
