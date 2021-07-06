import sympy
import numpy as np
from ParentsGradient import Gradient

class Gradient_Classic(Gradient):
    def gradient_classic(self):#задаемя начальными случайными значениями и каждый шаг спускаемся
        W=np.random.rand(2)
        W_prev = np.random.rand(2)#предыдущая точка
        for i in range(self.max_iter):
            if (np.absolute(np.linalg.norm(W - W_prev)) <= self.epsilon):#нормаль вектора по абсолютному значению
                print(f"Решение найдено,{i+1} итераций")
                break
            if (i == self.max_iter-1):#если не нашли решение
                print(f"Решение за {i+1} не найдено")
                print(f"Точность решения за {i+1} итераций = ", np.absolute(np.linalg.norm(W - W_prev)))
                break
            W_prev=W
            Grad_W = np.array([ 2*W[0]-4.8 , 2*W[1]+6.2 ])#вектор градиента
            W = W - self.rate*Grad_W#градиентный шаг
            self.iter=i+1
        return W