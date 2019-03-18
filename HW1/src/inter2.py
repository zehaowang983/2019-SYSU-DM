import numpy as np  
import math
from scipy.integrate import tplquad,dblquad,quad

def fun(y,x):
    return (y*y * math.exp(-y*y) + x*x*x*x * math.exp(-x*x)) / (x * math.exp(-x*x))

# print(fun(1,4))
#二重积分
val2,err2=dblquad(fun,2,4,lambda x:-1,lambda x:1)
print ('二重积分结果：',val2)
print('二重积分绝对误差的估计值',err2)