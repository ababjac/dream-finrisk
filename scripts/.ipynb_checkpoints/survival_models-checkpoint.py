import lifelines
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.svm import FastKernelSurvivalSVM
from sksurv.linear_model import CoxnetSurvivalAnalysis
from cox_nnet import *
from scipy.stats import chi2
from sklearn.model_selection import GridSearchCV

import pandas as pd

#--------------------------------------------------------------------------------------------------#

# #https://stackoverflow.com/questions/40327399/hosmer-lemeshow-goodness-of-fit-test-in-python
# def calculate_HLStat(y_outcomes, y_probs, g=10):
#     pihat = y_probs
#     pihatcat = pd.cut(pihat, np.percentile(pihat,list(range(0, 101, (100/g))),labels=False,include_lowest=True)
    
#     meanprobs = [0]*g
#     expevents = [0]*g
#     obsevents = [0]*g
#     meanprobs2 = [0]*g
#     expevents2 = [0]*g
#     obsevents2 = [0]*g 
    
#     for i in range(g):
#        meanprobs[i]=np.mean(pihat[pihatcat==i])
#        expevents[i]=np.sum(pihatcat==i)*np.array(meanprobs[i])
#        obsevents[i]=np.sum(Y[pihatcat==i])
#        meanprobs2[i]=np.mean(1-pihat[pihatcat==i])
#        expevents2[i]=np.sum(pihatcat==i)*np.array(meanprobs2[i])
#        obsevents2[i]=np.sum(1-Y[pihatcat==i]) 

                
#     data1={'meanprobs':meanprobs,'meanprobs2':meanprobs2}
#     data2={'expevents':expevents,'expevents2':expevents2}
#     data3={'obsevents':obsevents,'obsevents2':obsevents2}
#     m=pd.DataFrame(data1)
#     e=pd.DataFrame(data2)
#     o=pd.DataFrame(data3)
    
#     #calculate test statistic
#     tt = sum(sum((np.array(o)-np.array(e))**2/np.array(e)))      
                      
#     #use chi2 distribution and p-value can test HL > C2
#     pvalue = 1 - chi2.cdf(tt,g-2)
        
#     return (round(tt, 4), round(pvalue, 4))

#--------------------------------------------------------------------------------------------------#
                      
def run_CoxPH(df):
    cph = lifelines.CoxPHFitter() #there is l1, l2 regularization and penalty parameters here
    cph.fit(df, duration_col='Event_time', event_col='Event')
    cph.print_summary()
    
    return cph
                      
#--------------------------------------------------------------------------------------------------#
                      
def run_WeibullAFT(df):
    aft = lifelines.WeibullAFTFitter()
    aft.fit(df, duration_col='Event_time', event_col='Event')
    aft.print_summary()
    
    return aft

#--------------------------------------------------------------------------------------------------#

def run_ElasticNetCox(X_train, X_test, y_train, y_test, random_state=762538):
    cox_elastic_net = CoxnetSurvivalAnalysis(alpha_min_ratio=0.01, max_iter=500)
    cox_elastic_net.fit(X_train, y_train)
    
    grid = GridSearchCV(cox_elastic_net, 
                        param_grid={
                            'alphas' : [[v] for v in cox_elastic_net.alphas_],
                            'l1_ratio' : [0.001, 0.01, 0.1, 1]
                        },
                        cv=5,
                        verbose=3
                       )
    
    grid.fit(X_train, y_train)
    cindex = grid.score(X_test, y_test)
    
    return cindex, grid
     
#--------------------------------------------------------------------------------------------------#
    
def run_RandomSurvival(X_train, X_test, y_train, y_test, n_estimators=500, random_state=1009563):
    rsf = RandomSurvivalForest(n_estimators=n_estimators, random_state=random_state)
    
    grid = GridSearchCV(rsf, 
                        param_grid={
                            'max_features' : ['sqrt', 'log2'],
                            'max_depth' : [6,7,8],
                        },
                        cv=5,
                        verbose=3
                       )
    
    grid.fit(X_train, y_train)
    cindex = grid.score(X_test, y_test)
    
    return cindex, grid
    
#--------------------------------------------------------------------------------------------------#
    
def run_SurvivalSVM(X_train, X_test, y_train, y_test, random_state=56281):
    kssvm = FastKernelSurvivalSVM(random_state=random_state)
    
    grid = GridSearchCV(kssvm, 
                        param_grid={
                            'kernel' : ['linear', 'rbf', 'sigmoid'],
                            'alpha' : [0.01, 0.1, 1, 10],
                            'gamma' : [1, 0.1, 0.01]
                        },
                        cv=5,
                        verbose=3
                       )

    grid.fit(X_train, y_train)
    cindex = grid.score(X_test, y_test)
    
    return cindex, grid

#--------------------------------------------------------------------------------------------------#

def run_SurvivalGB(X_train, X_test, y_train, y_test, n_estimators=500, random_state=10852):
    est_cph_tree = GradientBoostingSurvivalAnalysis(n_estimators=n_estimators, random_state=random_state)
    
    grid = GridSearchCV(est_cph_tree, 
                        param_grid={
                            'loss' : ['coxph', 'squared', 'ipcwls'],
                            'max_depth' : [5,6,7],
                            'max_features' : ['sqrt', 'log2']
                        },
                        cv=5,
                        verbose=3
                       )
    
    grid.fit(X_train, y_train)
    cindex = grid.score(X_test, y_test)
    
    return cindex, grid

#--------------------------------------------------------------------------------------------------#

#http://traversc.github.io/cox-nnet/docs/
def run_SurvivalNN(X_train, X_test, ytime_train, ystatus_train, ytime_test, ystatus_test):
    model_params = dict(node_map = None, input_split = None)
    search_params = dict(
        method = "nesterov", 
        learning_rate=0.01, 
        momentum=0.9,
        max_iter=2000, 
        stop_threshold=0.995, 
        patience=1000, 
        patience_incr=2, 
        rand_seed = 123,
        eval_step=23, 
        lr_decay = 0.9, 
        lr_growth = 1.0
    )
    cv_params = dict(cv_seed=1, n_folds=5, cv_metric = "loglikelihood",
    L2_range = numpy.arange(-4.5,1,0.5))
    
    cv_likelihoods, L2_reg_params, mean_cvpl = L2CVProfile(
        x_train,
        ytime_train,
        ystatus_train,
        model_params,
        search_params,
        cv_params, 
        verbose=False
    )
    
    L2_reg = L2_reg_params[numpy.argmax(mean_cvpl)]
    model_params = dict(node_map = None, input_split = None, L2_reg=numpy.exp(L2_reg))
    model, cost_iter = trainCoxMlp(
        x_train, 
        ytime_train, 
        ystatus_train, 
        model_params, 
        search_params, 
        verbose=True
    )
    theta = model.predictNewData(x_test)
    
    return theta
