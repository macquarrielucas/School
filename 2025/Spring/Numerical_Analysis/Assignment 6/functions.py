import numpy as np

def h(f,x,y):
    return f(x,y)[0]**2 + f(x,y)[1]**2

''' This actually doesn't need to be calculated.
def Dh(Df,f,x,y):
    return np.array([[2*f(x,y)[0]*Df(x,y)[0][0], 2*f(x,y)[0]*Df(x,y)[0][1]],
                     [2*f(x,y)[1]*Df(x,y)[1][0], 2*f(x,y)[1]*Df(x,y)[1][1]]])
'''

def f2(x,y):
    return np.array([x**2 - 1, 
                     y**2 - 1])
    #return np.array([np.exp(x**2 + y**2)-3, 
    #                 x+y-np.sin(3*(x+y))])

def Df2(x,y):
    return np.array([[2*x, 0],
                    [0, 2*y]])

def f1(x,y):
    return np.array([np.exp(x**2 + y**2)-3, 
                    x+y-np.sin(3*(x+y))])

def Df1(x,y):
    return np.array([[2*x*np.exp(x**2 + y**2), 2*y*np.exp(x**2 + y**2)],
                     [1-np.cos(3*(x+y))*3, 1-np.cos(3*(x+y))*3]])
