from mpl_toolkits import mplot3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rnd
from matplotlib.collections import LineCollection as lc



def torus (n,c=7,a=3,pc=2):
    #torus
    mu=np.linspace(0,pc*np.pi,n)
    nu=np.linspace(0,pc*np.pi,n)
    mu,nu=np.meshgrid(mu,nu)

    x=(c+a*np.cos(nu))*np.cos(mu)
    y=(c+a*np.cos(nu))*np.sin(mu)
    z=a*np.sin(nu)

    x = x.ravel()
    y = y.ravel()
    z = z.ravel()
    return x,y,z

def met(iter,xbound=1,mu=0,sig=1,bcent=0.2,dim=1):
        #modified metropolis algorithm for multiple dimensions
        states=[]
        x=[]

        niter=(iter**dim//(1-bcent))
        burn_in=int(niter*bcent)
        #print(niter,burn_in,niter-burn_in,sep="\n\n")
        #if ((iter-burn_in )% dim!=0):
                #burn_in=burn_in-((iter-burn_in)%dim)

        normalDist= lambda x : (np.exp((-(x-mu)**2)/(2*sig**2)))/(sig * np.sqrt(2*np.pi))
        nextState= lambda x : rnd.uniform(-xbound*sig+mu,xbound*sig+mu)
        unif= lambda : rnd.uniform(0,1)
        c=nextState(xbound)
        for i in range(int(niter)):
            states.append(c)
            m=nextState(xbound)
            cProb=normalDist(c)
            mProb=normalDist(m)
            accept=min(mProb/cProb,1)
            if unif()<=accept:
                c=m
        x=states[burn_in:]
        sqDim=(iter*np.ones(dim,dtype=int))
        #print(sqDim)
        return np.array(x).reshape(sqDim)
def dsimN(m):
    '''This function creates a dissimilarity matrix using
     the Euclidean distance metric from a M*N point cloud matrix'''

    dray=[]
    for i in range(m.shape[0]):
        mt=m[i][:,None]
        sqRes=(m[i]-mt)**2
        if i == 0:
            dray=sqRes[:]
        else: 
            dray+= sqRes[:]
    return np.sqrt(dray)

l3c=mplot3d.art3d.Line3DCollection
n=16
x,y,z=torus(n,pc=1)
noise= lambda : met(n**2)

xn=x + noise()
yn=y + noise()
zn=z + noise()

r=np.array([xn,yn,zn])

rdsim=dsimN(r)
#print(rdsim,r,sep="\n\n")
rSort=np.unique(np.sort(rdsim.reshape(-1)))[1:]
rQ3=rSort[(len(rSort)-1)//16]

splitIndex=np.array(np.where((rdsim<=rQ3) & (rdsim!=0)), dtype='object')
#print(splitIndex)
izip=np.array(list(zip(splitIndex[0],splitIndex[1])))
izip_=np.unique(list(map(sorted, izip)),axis=0)
#print(r,izip_,sep="\n\n")
itop=list(map(lambda x: [tuple(r[:,x[0]]),tuple(r[:,x[1]])],izip_))


fig = plt.figure(facecolor='green')
ax = plt.axes(projection="3d")
ax.set_facecolor(color="green")
ax.add_collection(l3c(itop,color="black"))
ax.scatter3D(r[0],r[1],r[2],color="red")
ax.set_zlim(-11,11)
plt.axis("off")
plt.show()