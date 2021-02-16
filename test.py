from derivative_pattern import Functions

# x^2 + 3x - 2
# f``(x) = 2
# f(1) = 2
# f(-1) = -4
functions = Functions(h=2)
f1 = (-1, -4)
f2 = (1, 2)
functions.add_func(f1)
functions.add_func(f2)
functions.t()
print(functions.y(x=48124))
functions.extract_f(ran=range(-100, 101))
