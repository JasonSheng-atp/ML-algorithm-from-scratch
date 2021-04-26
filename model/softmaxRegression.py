from utils import softmax
import numpy as np
__all__ = ['softmax_Regression']
class softmax_Regression:
    def __init__(self,input_dim = 3072,output_dim=15):
        self.W = np.random.rand(output_dim,input_dim)
    def fit(self,train_x,train_v_y,iters = 1000,alpha=0.1,lambd=0.01):
        n_samples = train_x.shape[0]
        for _ in range(iters):
            scores = np.dot( train_x, self.W.T)
            probs = softmax(scores)
            dw = -(1.0 / n_samples) * np.dot((train_v_y - probs).T, train_x) + lambd * self.W
            dw[:,0] = dw[:,0] - lambd * self.W[:,0]
            self.W  = self.W - alpha * dw
        return self.W