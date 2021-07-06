import sympy
import numpy as np
from ParentsGradient import Gradient

class Gradient_RMSProp(Gradient):
    def gradient_RMSProp(self):
        W=np.random.rand(2)
        G = np.random.rand(2)
        W_prev = np.array([100, 100])
        gamma = 1 - self.lambd
        for i in range(self.max_iter):
            if (np.absolute(np.linalg.norm(W - W_prev)) <= self.epsilon):
                print(f"Решение найдено,{i+1} итераций")
                break
            if (i == self.max_iter-1):
                print(f"Решение за {i+1} не найдено")
                print(f"Точность решения за {i+1} итераций = ", np.absolute(np.linalg.norm(W - W_prev)))
                break
            W_prev=W
            Grad_W = np.array([ 2*W[0]-4.8 , 2*W[1]+6.2 ])
            G=gamma*G+(1-gamma)*Grad_W*Grad_W
            W = W_prev-(1-gamma)*Grad_W/np.sqrt(G+self.epsilon)
            self.iter=i+1
        return W