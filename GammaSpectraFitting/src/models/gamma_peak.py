import numpy as np

def gaussian(x,A,mu,sigma):
    #Simple Gaussian function for fitting peaks
    return A*np.exp(-0.5*((x-mu)/sigma)**2 )

def background(x,c0,c1,c2):
    #Exponential plus constant background model
    return c0 + c1 * np.exp(-c2 * x)
def total_model(x,A,mu,sigma,c0,c1,c2):
    #Combined model of Gaussian peak and background
    return gaussian(x,A,mu,sigma) + background(x,c0,c1,c2)