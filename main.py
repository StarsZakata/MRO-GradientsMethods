import sympy
import numpy as np
import matplotlib.pyplot as plt

from Classic import Gradient_Classic
from Momentum import Gradient_Momentum
from NAG import  Gradient_NAG
from RMSProp import Gradient_RMSProp
from AdaDelta import Gradient_AdaDelta
from Adam import Gradient_Adam

#Данное уравнение(Вариант №16)
def f(x1,x2):
    return 5*x1**2+2*x2**2-4*x1+5*x2
#Частные производные Уравнений
def df():
    x1 = sympy.Symbol('x1')
    x2 = sympy.Symbol('x2')
    Wx1x2=5*x1**2+2*x2**2-4*x1+5*x2
    return sympy.diff(Wx1x2, x1),sympy.diff(Wx1x2, x2)

array=df()
print(array)
n=100

#Количество итераций, для отображения "ящика с усами"
classicIteration = list()
momentumIteration = list()
nagIteration = list()
rmspropIteration = list()
adadeltaIteration = list()
adamIteration = list()

#Число, для расчета среднего числа итераций
SumClassicIter=0
SumMomentumIter=0
SumNAGIter=0
SumRMSPropIter=0
SumAdaDeltaIter=0
SumAdamIter=0

if __name__ == "__main__":
    for i in range(n):
        print("Расчет №",i + 1)
        classic = Gradient_Classic()
        print()
        print("______________________Расчет классическим методом__________________")
        print()
        x=classic.gradient_classic()
        print(x)
        print("Минимум заданной функции :", f(x[0],x[1]))
        Momentum = Gradient_Momentum()
        print()
        print("______________________Расчет методом Momentum__________________")
        print()
        x1= Momentum.gradient_Momentum()
        print(x1)
        print("Минимум заданной функции :", f(x1[0], x1[1]))
        NAG = Gradient_NAG()
        print()
        print("______________________Расчет методом NAG__________________")
        print()
        x2 = NAG.gradient_NAG()
        print(x2)
        print("Минимум заданной функции :", f(x2[0], x2[1]))
        RMSProp = Gradient_RMSProp()
        print()
        print("______________________Расчет методом RMSProp__________________")
        print()
        x3 = RMSProp.gradient_RMSProp()
        print(x3)
        print("Минимум заданной функции :", f(x3[0], x3[1]))
        AdaDelta = Gradient_AdaDelta()
        print()
        print("______________________Расчет методом AdaDelta__________________")
        print()
        x4 = AdaDelta.gradient_AdaDelta()
        print(x4)
        print("Минимум заданной функции :", f(x4[0], x4[1]))
        Adam = Gradient_Adam()
        print()
        print("______________________Расчет методом Adam__________________")
        print()
        x5 = Adam.gradient_Adam()
        print(x5)
        print("Минимум заданной функции :", f(x5[0], x5[1]))
        classicIteration.append(classic.iter)
        SumClassicIter+=classic.iter
        momentumIteration.append(Momentum.iter)
        SumMomentumIter += Momentum.iter
        nagIteration.append(NAG.iter)
        SumNAGIter += NAG.iter
        rmspropIteration.append(RMSProp.iter)
        SumRMSPropIter += RMSProp.iter
        adadeltaIteration.append(AdaDelta.iter)
        SumAdaDeltaIter += AdaDelta.iter
        adamIteration.append(Adam.iter)
        SumAdamIter += Adam.iter
print("Cреднее число итераций Классического метода: ",np.around(SumClassicIter/n))
print()
print("Cреднее число итераций Momentum метода: ",np.around(SumMomentumIter/n))
print()
print("Cреднее число итераций NAG метода: ",np.around(SumNAGIter/n))
print()
print("Cреднее число итераций RMSProp метода: ",np.around(SumRMSPropIter/n))
print()
print("Cреднее число итераций AdaDelta метода: ",np.around(SumAdaDeltaIter/n))
print()
print("Cреднее число итераций Adam метода: ",np.around(SumAdamIter/n))
data = [classicIteration, momentumIteration, nagIteration, rmspropIteration, adadeltaIteration, adamIteration]
ListTitle=['Classic-1','Momentum-2',' NAG-3','RMSProp-4','AdaDelta-5','Adam-6']
#plt.boxplot(data)
#plt.show()
# for i in range(6):
#     fig1, ax1 = plt.subplots()
#     ax1.set_title(ListTitle[i])
#     ax1.boxplot(data[i])
#     plt.show()

fig1, ax1 = plt.subplots()
ax1.set_title(ListTitle)
ax1.boxplot(data)
plt.show()




