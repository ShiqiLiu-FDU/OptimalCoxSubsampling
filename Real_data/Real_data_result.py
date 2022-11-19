import os
import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.optimize import minimize



def fun(x, pis):
    return np.min([np.array(x) * pis, np.ones(len(pis))], axis=0).sum()

def find_position(nums, target):
    # bisection find the position of target
    left = 0
    right = len(nums) - 1
    mid = -1
    while left <= right:
        mid = left + (right - left) // 2
        temp = fun(1/nums[mid], nums)
        if temp < target:
            left = mid + 1
        elif temp > target:
            right = mid - 1
        else:
            return mid
    if fun(1/nums[mid], nums) < target:
        return mid + 1
    else:
        return mid

def find_c(pis, r):
    """
    pi = min(c|...|, 1)
    :param pis:  an array of sampling probability
    :param r: subsampling rate, E(pi) no more than r
    :return: c that reach the predetermined subsampling rate r
    """
    pis_temp = pis.copy()  
    pis_temp.sort()
    pis_temp = np.clip(pis_temp, 1e-10, None) # Need inverse
    n = len(pis_temp)
    c0 = (n * r) / sum(pis_temp)
    if c0 * pis_temp[-1] <= 1:
        c = c0
    else:
        pis_inverse = pis_temp.copy()
        pis_inverse = sorted(pis_inverse, reverse=True)
        m_prime = find_position(pis_inverse, n*r)
        m = n - m_prime
        c = (n * r - (n - m)) / sum(pis[:m])
    return c

class optimalSampling():
    def __init__(self, z, data, beta):
        self.N = data.shape[0]
        self.p = z.shape[1]
        self.zNames = z.columns
        data_concat = pd.concat([z, data], axis=1)
        data_concat.sort_values('time', inplace=True)
        self.z = data_concat.iloc[:,:self.p].values
        self.data = data_concat.iloc[:,self.p:]
        self.beta = beta
        [self.uft, self.uft_map, self.uft_ix, self.nuft, self.risk_enter] = self.indices()

    def indices(self):
        # All failure times
        ift = np.flatnonzero(self.data.event.values == 1)
        ft = self.data.time.values[ift]

        # Unique failure times
        uft = np.unique(ft)
        nuft = len(uft)

        # Indices of cases that fail at each unique failure time.
        uft_map = dict([(x, i) for i,x in enumerate(uft)])
        uft_ix = [[] for k in range(nuft)] 
        for ix,ti in zip(ift,ft):
            uft_ix[uft_map[ti]].append(ix)

        # Indices of cases (failed or censored) that enter the risk set at each unique failure time.
        risk_enter1 = [[] for k in range(nuft)]
        for i,t in enumerate(self.data.time.values):
            ix = np.searchsorted(uft, t, "right") - 1
            if ix >= 0:
                risk_enter1[ix].append(i)
        
        risk_enter = [np.asarray(x, dtype=np.int32)
                                    for x in risk_enter1]
        return [uft, uft_map, uft_ix, nuft, risk_enter]

    def pi(self):
        linpred = np.dot(self.z, self.beta)        
        linpred -= linpred.max()
        e_linpred = np.exp(linpred)

        xp0, xp1 = 0., 0.                
        A = np.zeros([self.p, self.p])        
        M = np.zeros_like(self.z)
        # Iterate backward through the unique failure times.
        for i in range(self.nuft)[::-1]:
            # Update for new cases entering the risk set.
            ix = self.risk_enter[i]
            if len(ix) > 0:
                ixf = self.uft_ix[i]
                v = self.z[ix,:]
                xp0 += e_linpred[ix].sum()
                xp1 += (e_linpred[ix][:,None] * v).sum(0)
                zmu = self.z[self.uft_ix[i],:] - xp1/xp0
                A += np.einsum('ki,kj->ij', zmu, zmu)

                # The jump component
                J = np.zeros([self.N, self.p]) 
                J[ixf,:] = zmu
                # The compensator component
                dLamb = len(ixf)/xp0
                C = ((self.data.time.values >= self.uft[i]) * e_linpred)[:,None]  * (self.z -  xp1/xp0) * dLamb                 
                # Update M
                M += J - C

        A /= self.N
        invA = inv(A)
        traceOfVar = np.einsum('ij,ij->i', M.dot(invA), M.dot(invA))
        result = np.sqrt(traceOfVar)
        return result
      
    def subsample(self, subsamplingRate):
        temp = self.pi()     
        # Optimal subsampling
        c1 = find_c(temp, subsamplingRate)
        pi = np.clip(c1 * np.array(temp), None, 1)
        
        subData = self.data.copy()
        result = np.random.binomial(n=1, p=pi, size=self.N)
        subData.samPro = pi
        subData.subInd = result

        subData = subData.loc[subData.subInd == 1]
        subZ = pd.DataFrame(self.z.copy()[result == 1], 
                            index = subData.index, 
                            columns = self.zNames
                            )
        return subZ, subData

class subSampleInference():  
    def __init__(self, z, data, n):
        self.N = data.shape[0]
        self.p = z.shape[1]
        self.n = n
        data_concat = pd.concat([z, data], axis=1)
        data_concat.sort_values('time', inplace=True)
        self.z = data_concat.iloc[:,:self.p].values
        self.data = data_concat.iloc[:,self.p:]
        [self.uft, self.uft_map, self.uft_ix, 
        self.nuft, self.risk_enter] = self.indices()

    def indices(self):
        # All failure times
        ift = np.flatnonzero(self.data.event.values == 1) # Event indicator.
        ft = self.data.time.values[ift]

        # Unique failure times
        uft = np.unique(ft) # Note that uft is sorted.
        nuft = len(uft)

        # Indices of cases (failed only) that fail at each unique failure time
        uft_map = dict([(x, i) for i,x in enumerate(uft)]) # Map failure time to its order in (sorted) uft sequence
        uft_ix = [[] for k in range(nuft)] # Map the order of (tied) failure time to index set in full sequence
        for ix,ti in zip(ift,ft):
            uft_ix[uft_map[ti]].append(ix)

        # Indices of cases (failed or censored) that enter the risk set at each unique failure time.         
        risk_enter1 = [[] for k in range(nuft)] # This break all cases into parts. 
        for i,t in enumerate(self.data.time.values):
            ix = np.searchsorted(uft, t, "right") - 1 
            if ix >= 0:
                risk_enter1[ix].append(i)
        
        risk_enter = [np.asarray(x, dtype=np.int32) for x in risk_enter1]
        return [uft, uft_map, uft_ix, nuft, risk_enter]

    def subLoss(self, beta):
        linpred = np.dot(self.z, beta)
        linpred -= linpred.max()
        e_linpred = np.exp(linpred)

        xp0 = 0.
        like = 0.
        # Iterate backward through the unique failure times.
        for i in range(self.nuft)[::-1]:
            # Update for new cases entering the risk set.
            ix = self.risk_enter[i]
            xp0 += (1/self.data.samPro.values[ix] * e_linpred[ix]).sum()

            # Account for all cases that fail at this point.
            ix = self.uft_ix[i]
            like += (1/self.data.samPro.values[ix] * (linpred[ix] - np.log(xp0))).sum()        
        return -like


    def subderiv(self, beta):
        grad = 0.
        
        linpred = np.dot(self.z, beta)        
        linpred -= linpred.max()
        e_linpred = np.exp(linpred)

        xp0, xp1 = 0., 0.

        # Iterate backward through the unique failure times.    
        for i in range(self.nuft)[::-1]:
            # Update for new cases entering the risk set.
            ix = self.risk_enter[i]
            if len(ix) > 0:
                v = self.z[ix,:]
                xp0 += (e_linpred[ix]/self.data.samPro.values[ix]).sum()
                xp1 += ((e_linpred[ix][:,None]/self.data.samPro.values[ix][:,None] * v)).sum(0)
                # Account for all cases that fail at this point.
                ix = self.uft_ix[i]
                grad += ((self.z[ix,:] - xp1 / xp0).T/self.data.samPro.values[ix]).T.sum(0) 

        return -grad
        
    def subResult(self, beta):  # can set pilot estimate as initial estimate
        res = minimize(self.subLoss, x0=beta, method='BFGS', jac=self.subderiv)  # , jac=self.deriv)
        return res.x

    def estSe(self, beta):
        linpred = np.dot(self.z, beta)
        linpred -= linpred.max()
        e_linpred = np.exp(linpred)

        xp0, xp1 = 0., 0.
        M = np.zeros([self.N, self.p]) # The martingale matrix.
        A = np.zeros([self.p, self.p]) # The A matrix.
        # Iterate backward through the unique failure times.
        for i in range(self.nuft)[::-1]:
            # Update for new cases entering the risk set.
            ix = self.risk_enter[i]
            if len(ix) > 0:                    
                ixf = self.uft_ix[i]
                v = self.z[ix,:]                
                xp0 += (e_linpred[ix]/self.data.samPro.values[ix]).sum()
                xp1 += ((e_linpred[ix][:,None]/self.data.samPro.values[ix][:,None] * v)).sum(0)
                # Update A
                zmu = self.z[ixf,:] - xp1/xp0  # z minus mu, mu = xp1/xp0                          
                pro = self.data.samPro.values[ixf]
                A += np.einsum('k, ki,kj->ij', 1/pro, zmu, zmu)
                # Update the jump component
                J = np.zeros([self.N, self.p]) # The jump matrix. 
                J[ixf,:] = zmu
                # Update compensator component
                dLamb = (1/self.data.samPro.values[ixf]).sum()/xp0
                C = ((self.data.time.values >= self.uft[i]) * e_linpred)[:,None]  * (self.z -  xp1/xp0) * dLamb 
                # Update M
                M += J - C
        Sigma = np.einsum('k, ki,kj->ij', 1/self.data.samPro.values**2, M, M)

        A /= self.n
        Sigma /= self.n
        invA = inv(A)
        return np.diag(invA.dot(Sigma).dot(invA)/self.n) ** 0.5

class fullSampleInference():
    def __init__(self, z, data):       
        self.N = data.shape[0]
        self.p = z.shape[1]
        data_concat = pd.concat([z, data], axis=1)
        data_concat.sort_values('time', inplace=True)
        self.z = data_concat.iloc[:,:self.p].values
        self.data = data_concat.iloc[:,self.p:]
        [self.uft, self.uft_map, self.uft_ix, 
        self.nuft, self.risk_enter] = self.indices()

    def indices(self):
        # All failure times
        ift = np.flatnonzero(self.data.event.values == 1)
        ft = self.data.time.values[ift]

        # Unique failure times
        uft = np.unique(ft)
        nuft = len(uft)

        # Indices of cases that fail at each unique failure time
        uft_map = dict([(x, i) for i,x in enumerate(uft)])
        uft_ix = [[] for k in range(nuft)]
        for ix,ti in zip(ift,ft):
            uft_ix[uft_map[ti]].append(ix)

        # Indices of cases (failed or censored) that enter the risk set at each unique failure time.
        risk_enter1 = [[] for k in range(nuft)]
        for i,t in enumerate(self.data.time.values):
            ix = np.searchsorted(uft, t, "right") - 1
            if ix >= 0:
                risk_enter1[ix].append(i)

        risk_enter = [np.asarray(x, dtype=np.int32)
                                    for x in risk_enter1]
        return [uft, uft_map, uft_ix, nuft, risk_enter]

    def fullLoss(self, beta): 
        linpred = np.dot(self.z, beta)        
        linpred -= linpred.max()
        e_linpred = np.exp(linpred)

        xp0 = 0.
        like = 0.
        for i in range(self.nuft)[::-1]:
            # Update for new cases entering the risk set.
            ix = self.risk_enter[i]
            xp0 += e_linpred[ix].sum()

            # Account for all cases that fail at this point.
            ix = self.uft_ix[i]
            like += (linpred[ix] - np.log(xp0)).sum()          
        return -like


    def fullderiv(self, beta):
        grad = 0.
        
        linpred = np.dot(self.z, beta)        
        linpred -= linpred.max()
        e_linpred = np.exp(linpred)

        xp0, xp1 = 0., 0.

        # Iterate backward through the unique failure times.
        for i in range(self.nuft)[::-1]:
            # Update for new cases entering the risk set.
            ix = self.risk_enter[i]
            if len(ix) > 0:
                v = self.z[ix,:]
                xp0 += e_linpred[ix].sum()
                xp1 += (e_linpred[ix][:,None] * v).sum(0)

            # Account for all cases that fail at this point.
            ix = self.uft_ix[i]
            grad += (self.z[ix,:] - xp1 / xp0).sum(0) 
        
        return -grad
        
    def fullResult(self):
        res = minimize(self.fullLoss, x0=np.zeros(self.p), method='BFGS', jac=self.fullderiv)
        return res.x
         
    def estSe(self,beta):     
        linpred = np.dot(self.z, beta)
        linpred -= linpred.max()
        e_linpred = np.exp(linpred)

        xp0, xp1 = 0., 0.                
        mu = np.zeros_like(self.z)
        # Iterate backward through the unique failure times.
        for i in range(self.nuft)[::-1]:
            # Update for new cases entering the risk set.
            ix = self.risk_enter[i]
            if len(ix) > 0:
                v = self.z[ix,:]
                xp0 += e_linpred[ix].sum()
                xp1 += (e_linpred[ix][:,None] * v).sum(0)
                mu[self.uft_ix[i],:] = xp1/xp0
        
        Sigma = np.einsum('k,ki,kj->ij',  self.data.event, self.z - mu, self.z - mu) / self.N
        return np.diag(inv(Sigma)/self.N) ** 0.5

#===================================================================================================================
import pickle
import multiprocessing
import os

class Realdata(multiprocessing.Process):
    def __init__(self,                  
                seed_list, 
                thresh,
                year,
                numPilotSample,
                subsamplingRate,               
                saveFolder = ''):
        multiprocessing.Process.__init__(self)        
        self.seed_list = seed_list        
        self.saveFolder = saveFolder
        self.thresh = thresh
        self.year = year
        self.numPilotSample = numPilotSample
        self.subsamplingRate = subsamplingRate

    def run(self):
        for seed in self.seed_list:
            saveFilename = self.saveFolder + str(seed) + '.pickle'
            with open(saveFilename, 'wb') as f:  
                data = pd.read_csv('Lung.csv')
                thresh = self.thresh
                year = self.year

                data.columns = ['Sex', 'Age', 'time', 'Year']
                data.loc[data.Sex == 'Female', 'Sex'] = 0
                data.loc[data.Sex == 'Male', 'Sex'] = 1
                data.Sex = pd.to_numeric(data.Sex)
                data.Age = pd.to_numeric(data.Age.apply(lambda x:x.split()[0]))
                data.Age = (data.Age - np.mean(data.Age))/np.std(data.Age)
                data['event'] = (data.time <= thresh).astype(int)
                data.time = np.clip(data.time, 0 ,thresh)
                data['subInd'] = np.ones(data.shape[0])
                data['samPro'] = np.ones(data.shape[0])
                data = data.loc[data.Year <= year,:]

                data.sort_values('time', inplace=True)
                z = data.iloc[:,range(2)]
                data.drop(columns=data.columns[range(2)], inplace=True)
                data.drop(columns='Year', inplace=True)                                    
                
                np.random.seed(seed)
                N = data.shape[0]
                numPilotSample = self.numPilotSample
                subsamplingRate = self.subsamplingRate
                
                Full = fullSampleInference(z, data)
                FullBeta = Full.fullResult()
                FullStd = Full.estSe(FullBeta)

                pilotInd = np.sort(np.random.choice(N, numPilotSample, replace = False))
                pilotData = data.iloc[pilotInd, :].copy()
                pilotZ = z.iloc[pilotInd, :].copy()
                pilotBeta = fullSampleInference(pilotZ, pilotData).fullResult()                
                optimalZ, optimalData = optimalSampling(data = data, z = z, beta = np.zeros(2)).subsample(subsamplingRate)

                Optimal = subSampleInference(optimalZ, optimalData, N)
                OptimalBeta = Optimal.subResult(pilotBeta)
                OptimalStd = Optimal.estSe(OptimalBeta)
                
                unifInd = np.random.binomial(n=1, p = subsamplingRate + numPilotSample/N, size = N).astype(bool)
                unifData = data.iloc[unifInd, :].copy()
                unifZ = z.iloc[unifInd, :].copy()
                Unif = fullSampleInference(unifZ, unifData)
                UnifBeta = Unif.fullResult()
                UnifStd = Unif.estSe(UnifBeta)
                
                pickle.dump(
                [pilotBeta, OptimalBeta, OptimalStd,
                FullBeta, FullStd, UnifBeta, UnifStd], 
                file = f)
            f.close()

if __name__ == '__main__':
    numPilotSample = 100
    subsamplingRate = 0.01
    thresh = 12
    year = 2015
    
    processlist = []
    folder = './Optimal_sampling_real_data/'
    if(not os.path.exists(folder)):
        os.makedirs(folder)

    for i in range(100):
        seed_list = [int(i)]
        processlist.append(Realdata(
            seed_list = seed_list,
            thresh = thresh,
            year = year,
            numPilotSample = numPilotSample,
            subsamplingRate = subsamplingRate,
            saveFolder = folder
        ))

    for process in processlist:
        process.start()
    for process in processlist:
        process.join() 

    # Print result
