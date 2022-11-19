import os
import time
import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.optimize import minimize
from scipy.stats import multivariate_normal

'''
Main codes
Reference: statsmodels.duration.hazard_regression.py
'''
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

class DataGeneration():
    def __init__(self, z, hazard):
        self.z = z
        self.hazard = hazard
        self.N = len(hazard)

    # True failure time
    def failureTime(self):
        failureTime = []
        for each in self.hazard:
            temp = list(np.random.exponential(size=1, scale= 1/each))
            failureTime.extend(temp)
        failureTime = np.array(failureTime)
        return failureTime

    # Censoring time
    def censoringTime(self, mu):
        temp = np.random.exponential(size=self.N, scale= 1/mu)
        result = np.clip(temp, None, 6)  # set the length of study as 6
        return result

    # Get observed follow-up time and number of controls
    def timeAndControls(self, failureTime, censoringTime):
        indicator = []
        count = 0
        result = []
        n = failureTime.shape[0]
        for i in range(n):
            if failureTime[i] >= censoringTime[i]:
                temp = censoringTime[i]
                indicator.append(0)
            else:
                temp = failureTime[i]
                count += 1
                indicator.append(1)
            result.append(temp)
        censoringRate = (n - count) / n
        return np.array(result), np.array(indicator), censoringRate

    def getData(self, censoringRate, scheme = 'n+n'):
        if (scheme == 'n+n'): # normal + normal
            if(censoringRate == 0.3):
                mu = 0.3
            if(censoringRate == 0.5):
                mu = 0.9

        if (scheme == 'e+e'): # exponential + exponential
            if(censoringRate == 0.3):
                mu = 0.3
            if(censoringRate == 0.5):
                mu = 1
        
        if (scheme == 'b+n'): # binary + normal            
            if(censoringRate == 0.3): 
                mu = 0.5
            if(censoringRate == 0.5):
                mu = 1.6        
        
        if (scheme == 'e+b'): # exponential + binary
            if(censoringRate == 0.3): 
                mu = 0.5
            if(censoringRate == 0.5):
                mu = 1.5

        if (scheme == 'e+n'): # exponential + normal
            if(censoringRate == 0.3): 
                mu = 0.8
            if(censoringRate == 0.5):
                mu = 2.5

        if (scheme == 'mvnorm'): # 5-multivariate normal
            if(censoringRate == 0.5):
                mu = 1
        
        a, b, c = self.timeAndControls(self.failureTime(), self.censoringTime(mu))
        delta = np.ones(self.N)
        pi = np.ones(self.N)
        temp = pd.DataFrame({'time': a, 'event': b, 'subInd': delta, 'samPro': pi}) # Use dataframe to represent survival data
        result = pd.concat([pd.DataFrame(self.z), temp], axis=1)
        return result


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
        # Surprise sampling
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


'''
Run simulation
'''
import pickle
import multiprocessing

class Simulation(multiprocessing.Process):
    def __init__(self, 
                N, 
                seed_list,
                trueBeta,                 
                numPilotSample=500, 
                censoringRate=0.5, 
                subsamplingRate=0.01,
                scheme = 'n+n',
                saveFolder = '',
                isEqualTime = False):
        multiprocessing.Process.__init__(self)
        self.N = N
        self.seed_list = seed_list
        self.trueBeta = trueBeta
        self.numPilotSample = numPilotSample
        self.censoringRate = censoringRate
        self.subsamplingRate = subsamplingRate
        self.scheme = scheme
        self.saveFolder = saveFolder
        self.isEqualTime = isEqualTime
        
    def run(self):
        for seed in self.seed_list:            
            np.random.seed(seed)

            print(self.scheme + ': ' + str(seed))            
            saveFilename = self.saveFolder + 'result_seed_'+'_'+str(seed)+'.pickle'
            
            if self.scheme == 'n+n':
                z = multivariate_normal.rvs(mean = np.zeros(2), cov = np.array([[1,0.5],[0.5,1]]), size = self.N) 
            
            if self.scheme == 'e+e':
                z = np.array([np.random.exponential(size = self.N), np.random.exponential(size = self.N)]).T
            
            if self.scheme == 'b+n':                
                z = np.array([np.random.binomial(n=1, p=0.5, size=self.N), np.random.randn(self.N)]).T

            if self.scheme == 'e+b':
                z = np.array([np.random.exponential(size = self.N), np.random.binomial(n=1, p=0.5, size=self.N)]).T
            
            if self.scheme == 'mvnorm':
                z = np.random.randn(self.N, 5)

            if self.scheme == 'e+n':
                z = np.array([np.random.exponential(size = self.N), np.random.randn(self.N)]).T                

            with open(saveFilename, 'wb') as file:
                # Data generation
                hazard = np.exp(z.dot(self.trueBeta))  #==lambda_0=1, T follows exponential distribution
                fullData = DataGeneration(z=z, hazard=hazard).getData(self.censoringRate, self.scheme)                
                z = pd.DataFrame(z)

                # Full sample inference.
                start = time.time()
                fullResult = fullSampleInference(z, fullData)
                fullBeta = fullResult.fullResult()
                fullSe = fullResult.estSe(fullBeta)
                end = time.time()
                fullTime = end - start               

                # Optimal subsampling results.            
                start = time.time()
                # Pilot data
                pilotInd = np.sort(np.random.choice(self.N, self.numPilotSample, replace=False))
                pilotData = fullData.iloc[pilotInd,:]
                pilotZ = z.loc[pilotInd,:]
                pilotBeta = fullSampleInference(pilotZ, pilotData).fullResult()
                # Optimal subsampling inference.
                optimalSubsample = optimalSampling(data=fullData, z=z, beta=pilotBeta)
                optimalZ,optimalData = optimalSubsample.subsample(self.subsamplingRate)                
                optimalResult = subSampleInference(z=optimalZ, data=optimalData, n=self.N)
                optimalBeta = optimalResult.subResult(beta=pilotBeta)
                optimalSe = optimalResult.estSe(beta=optimalBeta)
                end = time.time()
                optimalTime = end - start

                # Uniform Bernoulli subsampling results.
                start = time.time()                                                
                # Uniform Bernoulli sampling with equal computation time.
                if self.isEqualTime:
                    if self.scheme == 'e+e':
                        unifInd = np.random.binomial(n=1, p = self.subsamplingRate * 6.5, size = self.N).astype(bool)
                    if self.scheme == 'b+n':
                        unifInd = np.random.binomial(n=1, p = self.subsamplingRate * 8, size = self.N).astype(bool)
                    else:
                        unifInd = np.random.binomial(n=1, p = self.subsamplingRate * 6, size = self.N).astype(bool)
                else:
                    # Uniform Bernoulli sampling with equal sample size.
                    unifInd = np.random.binomial(n=1, p = self.subsamplingRate + self.numPilotSample/self.N, size = self.N).astype(bool)                 
                # Uniform subsampling inference.
                unifData = fullData.iloc[unifInd,:]
                unifZ = z.loc[unifInd,:]
                unifResult = fullSampleInference(unifZ, unifData)
                unifBeta = unifResult.fullResult()
                unifSe = unifResult.estSe(unifBeta)
                end = time.time()
                unifTime = end - start

                # Save results 
                pickle.dump([
                    fullBeta, unifBeta, optimalBeta, 
                    fullSe, unifSe, optimalSe, 
                    fullTime, unifTime, optimalTime],
                    file = file)
                file.close()
          
if __name__ == '__main__':
    N = 10000
    censoringRate = 0.5
    subsamplingRate = 0.1
    
    # Compare with uniform Bernoulli sampling with equal sample size.
    for numPilotSample in [100, 500, 1000]:
        for scheme in ['e+e', 'e+b', 'n+n', 'b+n', 'mvnorm']:

            saveFolder = './Optimal_sampling_correct/Equal_size_' + scheme + '_' +str(numPilotSample) + '/'
            if(not os.path.exists(saveFolder)):
                os.makedirs(saveFolder)

            # Specify dimension and ture parameters.
            if (scheme == 'mvnorm'):
                p = 5
                trueBeta = np.array([1, -1, 0.5, -0.5, 0])
            else:
                p = 2
                trueBeta = np.array([1, -1])

            # Parallelization using 100 cores.
            processlist = []                                
            for i in range(100):
                seed_list = [int(k) for k in (np.arange(10) + i * 10)] # Runs 10 seeds per job.
                processlist.append(Simulation(
                    N = N,
                    seed_list = seed_list,
                    trueBeta = trueBeta,
                    censoringRate = censoringRate,
                    numPilotSample = numPilotSample,
                    subsamplingRate = subsamplingRate,
                    scheme = scheme,
                    saveFolder = saveFolder,
                    isEqualTime = False))
            for process in processlist:
                process.start()
            for process in processlist:
                process.join()

    # Compare with uniform Bernoulli sampling with equal computation time.
    for numPilotSample in [100, 500, 1000]:
        for scheme in ['e+e', 'e+b', 'n+n', 'b+n', 'mvnorm']:

            saveFolder = './Optimal_sampling_correct/Equal_time_' + scheme + '_'+str(numPilotSample) + '/'
            if(not os.path.exists(saveFolder)):
                os.makedirs(saveFolder)
            
            # Specify dimension and ture parameters.
            if (scheme == 'mvnorm'):
                p = 5
                trueBeta = np.array([1, -1, 0.5, -0.5, 0])
            else:
                p = 2
                trueBeta = np.array([1, -1])
            
            # Parallelization using 100 cores.
            processlist = []
            for i in range(100):
                seed_list = [int(k) for k in (np.arange(10) + i * 10)] # Runs 10 seeds per job.
                processlist.append(Simulation(
                    N = N,
                    seed_list = seed_list,
                    trueBeta = trueBeta,
                    censoringRate = censoringRate,
                    numPilotSample = numPilotSample,
                    subsamplingRate = subsamplingRate,
                    scheme = scheme,
                    saveFolder = saveFolder,
                    isEqualTime = True))
            for process in processlist:
                process.start()
            for process in processlist:
                process.join()

'''
Print results
'''