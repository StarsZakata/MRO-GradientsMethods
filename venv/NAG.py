import sympy
import numpy as np
from ParentsGradient import Gradient

class Gradient_NAG(Gradient):
    def gradient_NAG(self):
        W= np.random.rand(2)
        V = np.random.rand(2)
        W_prev = np.array([100, 100])
        gamma = 1 - self.lambd
        eta = (1 - gamma) * self.rate
        for i in range(self.max_iter):
            if (np.absolute(np.linalg.norm(W - W_prev)) <= self.epsilon):
                print(f"Решение найдено,{i+1} итераций")
                break
            if (i == self.max_iter-1):
                print(f"Решение за {i+1} не найдено")
                print(f"Точность решения за {i+1} итераций = ", np.absolute(np.linalg.norm(W - W_prev)))
                break
            W_prev=W
            Grad_W = np.array([ 2*(W[0]-gamma*V[0])-4.8 , 2*(W[1]-gamma*V[1])+6.2 ])
            V = gamma*V+eta*Grad_W
            W = W_prev-V
            self.iter=i
        return W