# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 10:10:36 2021

@author: Vieira
"""

#%% Import modules
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

print(np.__name__ + " version = " + np.__version__)
print(mpl.__name__ + " version = " + mpl.__version__)

#%% Generate random data
np.random.seed(1)

xlim = (-10, 10)
ylim = (-10, 10)
step = 0.02

#%%
N = 50

c1 = np.random.randn(N,2) + np.array([-1, 0])
c2 = np.random.randn(N,2) + np.array([ 3, 5])

X = np.vstack((c1,c2))
y = np.vstack((-1*np.ones((N,1)), np.ones((N,1))))

#%%
def sgn(x):
    return (1 if x >= 0 else -1)

def predict(w, x):
    return sgn(np.dot(w.T, x))

def get_dec_boundary(w):
    xx, yy = np.meshgrid(np.arange(xlim[0], xlim[1], step), 
                         np.arange(ylim[0], ylim[1], step))
    samples = np.vstack([np.ones((xx.size)), xx.ravel(), yy.ravel()])
    z = np.dot(w.T,samples)
    z[z>=0] = 1
    z[z<0] = 0
    return xx, yy, z.reshape(xx.shape)

def get_line(w):
    x1 = np.linspace(xlim[0], xlim[1], 2)
    x2 = ( -w[0] - w[1]*x1 ) / w[2]
    return x1, x2

def plot_data(X, y, w, ind = 0):
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1, 1, 1)

    plt.plot(X[y.ravel() ==  1,0], X[y.ravel() ==  1,1], 'go')
    plt.plot(X[y.ravel() == -1,0], X[y.ravel() == -1,1], 'ro')

    # Plot the separation line
    x1, x2 = get_line(w)
    plt.plot(x1, x2, '-k')
    # Plot
    plt.plot(X[ind,0], X[ind,1], 'ko')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    xx, yy, zz = get_dec_boundary(w)
    plt.contourf(xx, yy, zz, cmap=plt.cm.RdYlGn)
    
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.show()

#%%
# for i in range(10):
i = 5
w = np.array([i, 1, 1])

plot_data(X, y, w, i)

#%%

# Initialize weigths
# w = np.zeros((3,1))# np.random.random((3,1))
w = 10. * np.random.random((3,1))
w = w/ np.linalg.norm(w)

epochs = range(200)
eta = .2

for epoch in epochs:
  for i in range(X.shape[0]):
    x = np.hstack([1, X[i,:]]).reshape((3,1))
    y_gt = y[i]
    y_hat = predict(w, x)
    w = w + eta * (y_gt - y_hat) * x
    plot_data(X, y, w, i)
plot_data(X, y, w, i)

#%%
w = np.random.random((3,1))
epochs = range(5) #%% hyperparameter
eta = 1           #%% hyperparameter

for epoch in epochs:
  for i in range(X.shape[0]):
    x = np.hstack([1, X[i,:]]).reshape((3,1))
    y_gt = y[i]
    y_hat = predict(w, x)
    w = w + eta * (y_gt - y_hat) * x
    plot_data(X, y, w, i)
plot_data(X, y, w, i)
