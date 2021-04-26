import numpy as np
from utils import sigmoid_d,sigmoid,softmax,cross_entropy_d
__all__ =["net"]
class net:
    def __init__(self,input_dim = 3072,output_dim=15,hidden_neurons = 32):
        self.w1 = np.random.randn(input_dim, hidden_neurons)*np.sqrt(1/hidden_neurons)
        self.b1 = np.zeros((1, hidden_neurons))#1*128
        self.w2 = np.random.randn(hidden_neurons, output_dim)*np.sqrt(1/output_dim)
        self.b2 = np.zeros((1, output_dim))#1*15
    def forward(self):
        z1 = np.dot(self.x, self.w1) + self.b1
        self.a1 = sigmoid(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = softmax(z2)
    def backward(self,lambd):
        a2_delta = cross_entropy_d(self.a2, self.y) # w2
        z2_delta = np.dot(a2_delta, self.w2.T)
        a1_delta = z2_delta * sigmoid_d(self.a1) # w1

        self.w2 -= self.lr * (np.dot(self.a1.T, a2_delta)+1/self.batch_size*lambd*self.w2)
        self.b2 -= self.lr * np.sum(a2_delta, axis=0, keepdims=True)
        self.w1 -= self.lr * (np.dot(self.x.T, a1_delta)+1/self.batch_size*lambd*self.w1)
        self.b1 -= self.lr * np.sum(a1_delta, axis=0)
    def fit(self,x,y,epoch, batch_size,lr,lambd,cut_tail=True):
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.x_all = x
        self.y_all = y
        num_batch = self.x_all.shape[0]//batch_size
        for e in range(self.epoch):
            self.lr = lr*(1-e/self.epoch)
            for i in range(num_batch):
                self.x = self.x_all[i*batch_size:(i+1)*batch_size]
                self.y = self.y_all[i*batch_size:(i+1)*batch_size]
                self.forward()
                self.backward(lambd)
            if self.x_all.shape[0]>=num_batch*batch_size and not cut_tail:
                self.x = self.x_all[(num_batch+1)*batch_size:]
                self.y = self.y_all[(num_batch+1)*batch_size:]
                self.forward()
                self.backward(lambd)
    def predict(self, data):
        self.x = data
        self.forward()
        return self.a2.argmax()
if __name__ == "__main__":
    a = np.array([1,2,3]).reshape((-1,1))
    b = np.array([3,0,4])
    print(a)
    print(np.multiply(a,b))