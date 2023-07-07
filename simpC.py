#!/usr/bin/env python3

from mpl_toolkits import mplot3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection as lc
l3c=mplot3d.art3d.Line3DCollection


def met(iter, bcent=0.2,xbound=5,mu=0,sig=1):
        states=[]

        x=[]

        burn_in=int(iter*bcent)

        normalDist= lambda x : (np.exp((-(x-mu)**2)/(2*sig**2)))/(sig * np.sqrt(2*np.pi))
        nextState= lambda x : rnd.uniform(-xbound*sig+mu,xbound*sig+mu)
        unif= lambda : rnd.uniform(0,1)
        c=nextState(xbound)
        for i in range(iter):
            states.append(c)
            m=nextState(xbound)
            cProb=normalDist(c)
            mProb=normalDist(m)
            accept=min(mProb/cProb,1)
            if unif()<=accept:
                c=m
        x=states[burn_in:]
        return x

def dsim(m):
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

def eps(dm,perc=0.75):
    '''This function finds a strategic threshold value based on 
        the original dissimilarity matrix.
       It is set up with the inner quartile range in mind.'''
    ddsim=[]
    for i in range(dm.shape[0]):
        ddsim.append(np.delete(dm[i], np.s_[i::],0))
    ddsimArray=np.concatenate(ddsim[::],dtype='object')
    ddIns=np.sort(ddsimArray)
    ddQ=ddIns[np.int(np.floor((len(ddIns)-1)*perc))]
    #print(ddsimArray, ddIns,ddQ,sep="\n")
    return ddQ

def k1(o,dm,e):
    #r=np.random.rand(2,10)

    #rdsim=dsim(r)
    #print(rdsim)
    #oSort=np.unique(np.sort(dm.reshape(-1)))[1:]
    #rQ3=oSort[3*(len(oSort)-1)//4]
    #print(rQ3)
    #plt.plot(r[0],r[1], 'k.')
    splitIndex=np.array(np.where((dm<=e) & (dm!=0)), dtype='object')
    izip=np.array(list(zip(splitIndex[0],splitIndex[1])))
    izip_=np.unique(list(map(sorted, izip)),axis=0)
    itop=list(map(lambda x: [(o[0,x[0]],o[1,x[0]]),(o[0,x[1]],o[1,x[1]])],izip_))
    print(izip_[:20],"\n\n",o[:20],itop[:20],sep="\n")
    return itop




#ax1.autoscale()
#ax1.scatter(r[0],r[1],marker='.',c='black')


''' data is first read into a pandas dataframe and interesting columns 
    are taken out for further analysis.'''

covid=pd.read_excel("/Users/Launch/Downloads/ExcelSampleData 2/COVID-19_Daily_Testing with charts.xlsx","Descriptive Stats")
#print(len(covid["Date"]))
fig = plt.figure()
ax = plt.axes(projection="3d")


t=np.linspace(0,len(covid["Date"]),len(covid["Date"]))
icols=np.array(covid.loc[1:,["People Tested - Total","People Positive - Total"]])
y=covid["People Tested - Total"]
z=covid["People Positive - Total"]

'''Calculate dissimilarity matrix'''
edSim=dsim(icols.T)

'''Some nonrelevant data validation checks'''
maxdis1=max(edSim[0])
maxdin=np.where(edSim[0]==maxdis1)
#print(edSim,edSim[0],maxdis1,maxdin,maxdin[0].astype(float),sep="\n")
xs=(icols[0][0]-icols[133][0])**2
ys=(icols[0][1]-icols[133][1])**2
euc=np.sqrt(xs+ys)
#print("\n",euc,maxdis1,euc==maxdis1)
ax.scatter3D(t,y,z,marker=".",color="black")

'''Strategic epsilon threshold distance choosing'''
eps=eps(edSim)

'''Creating a line collection for graphing the k1 simplicial complex'''
#lcs=lc(k1(icols.T,edSim,eps))
#ax.add_collection(lcs)

plt.show()
