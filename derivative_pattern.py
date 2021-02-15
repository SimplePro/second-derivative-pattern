def sigma(i, k, h):
    result = 0

    for j in range(i, k+1):
        result += h * j

    return result


class Functions:
    def __init__(self, h):
        self.second_derivative = h  # 이계도함수
        self.increase = 0  # 증가량 (t)
        self.func = []  # 함숫값을 담는 변수
        self.coefficient_t = 0  # t 의 계수

    def add_func(self, f):
        if len(self.func) == 0:
            self.func.append(f)

        elif self.func[-1][0] >= f[0]:
            raise Exception("f[0] not be smaller than functions[-1][0]")
        else:
            self.func.append(f)

    def delete_func(self, index):
        del self.func[index]

    def t(self):
        # f``(x) = h, f(n) = a, f(m) = b (m - n > 1) # 전제조건
        # t = (b - a - sigma(i=1, m-n-1, hi)) / (m - n)  # t = f(n+1) - f(n). 증가량

        if len(self.func) < 2:
            raise Exception("len(func) not be smaller than 2")

        n, a = self.func[0]
        m, b = self.func[1]

        self.increase = (b - a - sigma(i=1, k=m-n-1, h=self.second_derivative)) / (m - n)
        return self.increase

    def y(self, x):
        # f``(x) = h, f(n) = a, t = f(n+1) - f(n)  (전제조건)
        # f(x) = a + (x-n)t + sigma(i=1, |m-n|, hi)  (x < n)
        # f(x) = a + (x-n)t + sigma(i=1, |m-n-1|, hi)  (x > n)

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


if __name__ == '__main__':
    # f(x) = 2x^2 + 5x
    f1 = (1, 7)  # (x, f(x))
    f2 = (5, 75)  # (x, f(x))

    functions = Functions(h=4)  # h=이계도함수
    functions.add_func(f1)  # 첫번째 함숫값 추가
    functions.add_func(f2)  # 두번째 함숫값 추가
    functions.t()  # f(n+1) - f(n) = t. 증가량 구하기
    print(functions.y(x=123942))
