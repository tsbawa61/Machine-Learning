def my_func(**kwargs):
    for i, j in kwargs.items():
        print(i, j)
my_func(name='tim', sport='football', roll=19, sex='M')

def my_three(a, b, c):
    print(a, b, c)
a = [1,2,3] ; my_three(*a) # here list is broken into three elements

def my_four(a, b, c,d):
    print(a, b, c, d)
a = {'a': "one", 'b': "two", 'c': "three", 'd':'four' } ; my_four(**a)

def func1(arg1, arg2=None):
    print(arg1)
    if (arg2==None):
        print("Arg2 not given")
        
func1(12)

