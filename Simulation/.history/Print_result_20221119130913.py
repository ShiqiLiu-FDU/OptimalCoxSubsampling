import numpy as np
import pickle
from collections.abc import Iterable
def round2(num):
    if isinstance(num,Iterable):
        return [round(n * 10**4)/10**2 for n in num]
    else:
        return round(num * 10**4)/10**2        
def round3(num):
    if isinstance(num,Iterable):
        return [round(n * 10**5)/10**2 for n in num]
    else:
        return round(num * 10**5)/10**2

def print_cox_results(scheme='n+n', numPilotSample=100, isEqualTime=False):
    if isEqualTime:
        folder = './Optimal_sampling_correct/' + 'Equal_time_' + scheme + '_'+ str(numPilotSample) + '/'
    else:
        folder = './Optimal_sampling_correct/' + 'Equal_size_' + scheme + '_'+ str(numPilotSample) + '/'
        
    if (scheme == 'mvnorm'):
        trueBeta = np.array([1, -1, 0.5, -0.5, 0])
    else:
        trueBeta = np.array([1, -1])

    fullBetaList = []
    unifBetaList = []
    optimalBetaList = []    
    fullSeList = []
    unifSeList = []
    optimalSeList = []
    timeList = []
    for i in range(1000):
        with open(folder + '/result_seed_'+'_'+str(i)+'.pickle', 'rb') as f:
            temp = pickle.load(f)            
            fullBetaList.append(temp[0])
            unifBetaList.append(temp[1])           
            optimalBetaList.append(temp[2])
            fullSeList.append(temp[3])
            unifSeList.append(temp[4])
            optimalSeList.append(temp[5])
            timeList.append(temp[6:])
            f.close()

    fullBetaResult = np.array(fullBetaList)
    unifBetaResult = np.array(unifBetaList)
    optimalBetaResult = np.array(optimalBetaList)
    fullSe = np.array(fullSeList)
    unifSe = np.array(unifSeList)
    optimalSe = np.array(optimalSeList)
    
    print('=================')
    print('Bias, SE, SEE, CP')
    print('=================')
    temp = unifBetaResult - trueBeta        
    unifmse = np.mean((np.einsum('ij,ij->i', temp, temp)))
    print('Uniform subsampling MSE: '+ str(round3(unifmse)))    
    print('Uniform subsampling RE: '+ str(1))
    times = [timeList[i][1] for i in range(1000)]   
    print('Time unif: ' + str(np.sum(times)))         

    temp = optimalBetaResult - trueBeta
    mse = np.mean((np.einsum('ij,ij->i', temp, temp)))
    print('Optimal subsampling MSE: '+ str(round3(mse)))
    print('Optimal subsampling RE: '+ str(round(unifmse/mse, 2)))
    times = [timeList[i][2] for i in range(1000)]
    print('Time optimal: ' + str(np.sum(times)))    

    temp = fullBetaResult - trueBeta
    mse = np.mean((np.einsum('ij,ij->i', temp, temp)))
    print('Full MSE: '+ str(round3(mse))) 
    print('Full RE: '+ str(round(unifmse/mse, 2)))
    times = [timeList[i][0] for i in range(1000)]
    print('Time full: ' + str(np.sum(times)))        

    print('==============')
    print('MSE, RE, Time')
    print('==============')
    print('Bias unif: ' + str(round3(np.mean((unifBetaResult - trueBeta), axis= 0))))
    print('SE unif: ' + str(round2(np.sqrt(np.mean((unifBetaResult - np.mean(unifBetaResult, axis=0))**2, axis= 0)))))
    print('SEE unif: ' + str(round2(np.mean(unifSe, axis = 0))))
    print('CP unif: ' + str(np.sum((unifBetaResult - trueBeta - 1.96 * unifSe <= 0) * (unifBetaResult - trueBeta + 1.96 * unifSe >= 0) , axis = 0)/1000))

    print('\n')
    print('Bias optimal: ' + str(round3(np.mean((optimalBetaResult - trueBeta), axis= 0))))        
    print('SE optimal: ' + str(round2(np.sqrt(np.mean((optimalBetaResult - np.mean(optimalBetaResult, axis=0))**2, axis= 0)))))        
    print('SEE optimal: ' + str(round2(np.mean(optimalSe, axis = 0))))        
    print('CP optimal: ' + str(np.sum((optimalBetaResult - trueBeta - 1.96 * optimalSe <= 0) * (optimalBetaResult - trueBeta + 1.96 * optimalSe >= 0) , axis = 0)/1000))

    print('\n')
    print('Bias full: ' + str(round3(np.mean((fullBetaResult - trueBeta), axis= 0))))
    print('SE full: ' + str(round2(np.sqrt(np.mean((fullBetaResult - np.mean(fullBetaResult, axis=0))**2, axis= 0)))))
    print('SEE full: ' + str(round2(np.mean(fullSe, axis = 0))))
    print('CP full: ' + str(np.sum((fullBetaResult - trueBeta - 1.96 * fullSe <= 0) * (fullBetaResult - trueBeta + 1.96 * fullSe >= 0) , axis = 0)/1000))

def print_misspecified_results(model_type='Cox', numPilotSample=100):
    folder = './Optimal_sampling_misspecified/' + model_type + '_'  + str(numPilotSample) + '/' 
        
    if (model_type == 'TM'):
        trueBeta = np.zeros(5)
    else:
        trueBeta = np.array([1, -1])

    fullBetaList = []
    unifBetaList = []
    optimalBetaList = []    
    fullSeList = []
    unifSeList = []
    optimalSeList = []
    timeList = []
    for i in range(1000):
        with open(folder + '/result_seed_'+'_'+str(i)+'.pickle', 'rb') as f:
            temp = pickle.load(f)            
            fullBetaList.append(temp[0])
            unifBetaList.append(temp[1])           
            optimalBetaList.append(temp[2])
            fullSeList.append(temp[3])
            unifSeList.append(temp[4])
            optimalSeList.append(temp[5])
            timeList.append(temp[6:])
            f.close()

    fullBetaResult = np.array(fullBetaList)
    unifBetaResult = np.array(unifBetaList)
    optimalBetaResult = np.array(optimalBetaList)
    fullSe = np.array(fullSeList)
    unifSe = np.array(unifSeList)
    optimalSe = np.array(optimalSeList)
    
    print('=================')
    print('Bias, SE, SEE, CP')
    print('=================')
    temp = unifBetaResult - trueBeta        
    unifmse = np.mean((np.einsum('ij,ij->i', temp, temp)))
    print('Uniform subsampling MSE: '+ str(round3(unifmse)))    
    print('Uniform subsampling RE: '+ str(1))
    times = [timeList[i][1] for i in range(1000)]   
    print('Time unif: ' + str(np.sum(times)))         

    temp = optimalBetaResult - trueBeta
    mse = np.mean((np.einsum('ij,ij->i', temp, temp)))
    print('Optimal subsampling MSE: '+ str(round3(mse)))
    print('Optimal subsampling RE: '+ str(round(unifmse/mse, 2)))
    times = [timeList[i][2] for i in range(1000)]
    print('Time optimal: ' + str(np.sum(times)))    

    temp = fullBetaResult - trueBeta
    mse = np.mean((np.einsum('ij,ij->i', temp, temp)))
    print('Full MSE: '+ str(round3(mse))) 
    print('Full RE: '+ str(round(unifmse/mse, 2)))
    times = [timeList[i][0] for i in range(1000)]
    print('Time full: ' + str(np.sum(times)))        

    print('==============')
    print('MSE, RE, Time')
    print('==============')
    print('Bias unif: ' + str(round3(np.mean((unifBetaResult - trueBeta), axis= 0))))
    print('SE unif: ' + str(round2(np.sqrt(np.mean((unifBetaResult - np.mean(unifBetaResult, axis=0))**2, axis= 0)))))
    print('SEE unif: ' + str(round2(np.mean(unifSe, axis = 0))))
    print('CP unif: ' + str(np.sum((unifBetaResult - trueBeta - 1.96 * unifSe <= 0) * (unifBetaResult - trueBeta + 1.96 * unifSe >= 0) , axis = 0)/1000))

    print('\n')
    print('Bias optimal: ' + str(round3(np.mean((optimalBetaResult - trueBeta), axis= 0))))        
    print('SE optimal: ' + str(round2(np.sqrt(np.mean((optimalBetaResult - np.mean(optimalBetaResult, axis=0))**2, axis= 0)))))        
    print('SEE optimal: ' + str(round2(np.mean(optimalSe, axis = 0))))        
    print('CP optimal: ' + str(np.sum((optimalBetaResult - trueBeta - 1.96 * optimalSe <= 0) * (optimalBetaResult - trueBeta + 1.96 * optimalSe >= 0) , axis = 0)/1000))

    print('\n')
    print('Bias full: ' + str(round3(np.mean((fullBetaResult - trueBeta), axis= 0))))
    print('SE full: ' + str(round2(np.sqrt(np.mean((fullBetaResult - np.mean(fullBetaResult, axis=0))**2, axis= 0)))))
    print('SEE full: ' + str(round2(np.mean(fullSe, axis = 0))))
    print('CP full: ' + str(np.sum((fullBetaResult - trueBeta - 1.96 * fullSe <= 0) * (fullBetaResult - trueBeta + 1.96 * fullSe >= 0) , axis = 0)/1000))

#################
# Print results #
#################
'''
scheme = ['n+n', 'e+e', 'e+b', 'b+n', 'mvnorm']
numPilotSample = [100, 500, 1000]
isEqualTime = [False, True]
'''
# Example
print_cox_results(scheme = 'n+n', numPilotSample = 100, isEqualTime = False)


