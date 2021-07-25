######################################################
###                                                ###
###                import packages                 ###
###                                                ###
######################################################
import collections
import itertools
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import shap

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

import time
import xgboost as xgb

######################################################
###                                                ###
###            defining functions                  ###
###                                                ###
######################################################

def var_drop_models(train_DF,
                    train_target,
                    test_DF,
                    loc,
                    filename,
                    varcols,
                    drpd_l,
                    perf_df = pd.DataFrame(),
                    earlier_drops = [],
                    num_drop = 5,
                    modl = xgb.XGBClassifier,
                    params_dict_l = [{"subsample":0.9, "colsample_bytree":0.9, "max_depth":6, "reg_alpha":0.3},
                                     {"reg_alpha":0.2, "n_estimators":50}],
                    ):
  """
  builds models on given train set with given hyperpars & tests them on given test set
  with every combination of variables defined by number of variables given
  records performance in DF

  vars:
    train_DF: DF with data
    train_target:
    test_DF:
    loc: string of location to save output
    filename: name of output file without extension
    varcols: list of names of columns in biggst model
    drpd_l: list of lists of dropped variables in models already checked (to avoid double checking)
    earlier_drops: list of names of vars dropped earlier from full model
    perf_df: initial perf_df if some of the models where fitted earlier
    num_drop: number of vars to drop for iteration (1: only largest model, 2: largest model + 1 model/each vars dropped...)
    params_dict_l: list of dictionaries of model parameters to fit
  """
  for num_vars in range(len(varcols), len(varcols)-num_drop,-1): #iterates through all number of vars from high to low
    print('- '*50)
    print(num_vars, time.ctime())
    num_iters = 0
    for i in itertools.combinations(varcols,num_vars):
      num_iters += 1
      vcols = list(i) #variables in local model
      drpd = [f for f in varcols if f not in vcols] + earlier_drops
      if set(drpd) not in drpd_l:

        train_df = train_DF[vcols]
        test_df = test_DF[vcols]

        for params in params_dict_l:
          mdl = modl(**params)

          perf = {}
          perf['variables'] = str(vcols)
          perf['dropped_vars'] = str(drpd)
          perf['num_vars'] = len(vcols)
          perf['model'] = str(mdl)
          perf['parameters'] = str(params)

          mdl.fit(train_df, train_target)

          perf['feature_imp'] = str(mdl.feature_importances_)
          train_pred = mdl.predict(train_df)
          test_pred = mdl.predict(test_df)

          perf['classification_report'] = classification_report(test_target, test_pred)
          perf['test_accuracy'] = accuracy_score(test_target, test_pred)
          perf['train_accuracy'] = accuracy_score(train_target, train_pred)
          perf['test_f1'] = f1_score(test_target, test_pred)
          perf['train_f1'] = f1_score(train_target, train_pred)

          perf_df = perf_df.append(perf, ignore_index = True)

        if num_iters%100 == 0:
          perf_df.to_excel(loc+filename+'_backup.xlsx')
          print(num_iters, drpd, time.ctime())

    
    perf_df.to_excel(loc+filename+'.xlsx')

  perf_df.to_excel(loc+filename+'.xlsx')

  return perf_df

#############################################################################################################################
def select_topn_dropped_bymodel(n,
                                break_acc,
                                perf_df,
                                params_dict_l = [{"subsample":0.9, "colsample_bytree":0.9, "max_depth":6, "reg_alpha":0.3},
                                                 {"reg_alpha":0.2, "n_estimators":50}]):
  """
  lists how many times each vars were dropped in top n models built/hyperpar set
  variables:
    n: how many models to look at per hyperpar set
    break_acc: only look at models with higher accuracy
    perf_df: DF with model perf data from var_drop_models function
    params_dict_l: list of hyperpar dictionaries
  """
  drpds_l = []
  sztr = {} #keeping count of number of models considered/híperpar set
  for params in params_dict_l:
    sztr[str(params)] = 0
  
  for i, drps in enumerate(perf_df.dropped_vars):
    sztr[perf_df.parameters.iloc[i]] += 1
    if sztr[perf_df.parameters.iloc[i]] <=n:
      drpds_l += [drps]
      fimp = [float(i) for i in perf_df.feature_imp.iloc[i][1:-1].split()] #feat importances
      mindx = np.where(np.array(fimp) == min(fimp)) #index of feature with smallest feat imp
      varis = [i.strip() for i in perf_df.variables.iloc[i][1:-1].split(',')] #variables in model
      trdp = '' #string to print features with smallest imp
      for ix in mindx[0]:
        trdp += str(ix) + ' ' + varis[int(ix)] + '\n'
      print(i, len(perf_df.variables.iloc[i].split()),
            drps, '\nacc:',
            perf_df.test_accuracy.iloc[i],
            'F1:', perf_df.test_f1.iloc[i],
            perf_df.parameters.iloc[i], '\n',
            perf_df.feature_imp.iloc[i], '\n',
            min(fimp), trdp)
      
    if perf_df.test_accuracy.iloc[i] < break_acc: #break loop if accuracy below break point
      break
  
  #count & print how many times each vars were dropped in models considered
  drpds_l = [re.sub(r'[\s\']', '', l[2:-2]).split(',') for l in drpds_l]
  print(len(drpds_l))
  drpds = [e for l in drpds_l for e in l]
  print(collections.Counter(drpds).most_common())

#############################################################################################################################
def select_topn_dropped_bynumval(n,
                                 perf_df):
  """
  lists how many times each vars were dropped in top n models built/ number of variables
  variables:
    n: how many models to look at per hyperpar set
    perf_df: DF with model perf data from var_drop_models function
  """
  drpds_l = []
  for n in set(perf_df.num_vars):
    for p in set(perf_df.parameters):
      drpds_l += list(perf_df[(perf_df.num_vars == n) &
                              (perf_df.parameters == p)][:n].dropped_vars)
  #count & print how many times each vars were dropped in models considered
  drpds_l = [re.sub(r'[\s\']', '', l[2:-2]).split(',') for l in drpds_l]
  print(len(drpds_l))
  drpds = [e for l in drpds_l for e in l]
  collections.Counter(drpds).most_common()

#############################################################################################################################
def test_vars_model(vars_df,
                    target_df,
                    pars,
                    n_main_cv,
                    n_cv,
                    scoring,
                    refit,
                    dropped_var = [],
                    mdl = xgb.XGBClassifier(),
                    verbose = 1,
                    randCV = False,
                    rand_n_iter = 0,
                    rnd_fld = True):
  """
  CV fitting & testing models on train data with CV & eturns prformance DF
  vars:
    - vars_df: train data
    - target_df: target var df or lost
    - pars: hyperpar space
    - n_main_cv: number of CV rounds for model fitting & testing with CV on splits
    - scoring: scoring to eval model by & record in output DF
    - refit: score to select best model
    - dropped_var: list of dropped vars (only to record in output DF)
    - mdl: model
    - verbose: verbose val of CV
    - randCV: bool - whether to only test a random sample of hyperpar space
    - rand_n_iter: if randCV=TRUE: how many pooints to test
    - rnd_fld: TRUE or list. if TRUE: random CV of train data else the fold of data is provided in the list
  """
  test_acc_l = []
  test_f1_l = []
  feat_imp_l = []
  perf = {}
  perf['variables'] = str(list(vars_df))
  perf['dropped_vars'] = str(dropped_var)
  perf['num_vars'] = len(list(vars_df))
  perf['model'] = str(mdl)
  perf['par_space'] = str(pars)
  perf['refit'] = refit
  perf['num_sub_cv'] = n_cv

  if type(rnd_fld) == bool:
    if rnd_fld: #ha nem: kívűlről hoz
      fld = np.random.choice(n_main_cv, len(target_df))
  else:
    fld = rnd_fld

  for f in range(n_main_cv):
    perf['CV_'+str(f)+'_train_size'] = sum(fld != f)
    perf['CV_'+str(f)+'_test_size'] = sum(fld == f)
    perf['CV_'+str(f)+'_train_ratio_roma'] = sum(target_df[fld!=f].roma)/sum(fld != f)
    perf['CV_'+str(f)+'_test_ratio_roma'] = sum(target_df[fld==f].roma)/sum(fld == f)

    train_df = vars_df[fld != f]
    train_target = target_df[fld != f]
    test_df = vars_df[fld == f]
    test_target = target_df[fld == f]
    
    clf = mdl
    if randCV:
      clf_cv = RandomizedSearchCV(clf,
                                  param_distributions=pars,
                                  n_iter = rand_n_iter,
                                  scoring=scoring,
                                  refit = refit,
                                  return_train_score=True,
                                  cv=n_cv,
                                  verbose=verbose,
                                  n_jobs = -1
          
      )
      perf['num_par_space_check'] = rand_n_iter
    else:
      clf_cv = GridSearchCV(clf,
                            param_grid=pars,
                            scoring=scoring,
                            refit = refit,
                            return_train_score=True,
                            cv=n_cv,
                            verbose=verbose,
                            n_jobs = -1)
    clf_cv.fit(train_df, train_target.iloc[:,0])

    best_ind = np.where(clf_cv.cv_results_['mean_test_'+refit] == max(clf_cv.cv_results_['mean_test_'+refit]))[0][0]
    for key in scoring:
      perf['CV_'+str(f)+'_mean_train_CV_'+key] = clf_cv.cv_results_['mean_test_'+key][best_ind]
    perf['CV_'+str(f)+'_best_parms'] = str(clf_cv.best_params_)
    
    preds = clf_cv.best_estimator_.predict(test_df)

    test_acc = accuracy_score(test_target.iloc[:,0], preds)
    test_acc_l.append(test_acc)
    perf['CV_'+str(f)+'_test_accuracy'] = test_acc

    test_f1 = f1_score(test_target.iloc[:,0], preds)
    test_f1_l.append(test_f1)
    perf['CV_'+str(f)+'_test_f1'] = test_f1

    perf['CV_'+str(f)+'_classification_report'] = classification_report(test_target.iloc[:,0], preds)

    feat_imp_l.append(clf_cv.best_estimator_.feature_importances_)
    perf['CV_'+str(f)+'_feature_imp'] = str(clf_cv.best_estimator_.feature_importances_)

  perf['mean_test_accuracy'] = np.mean(test_acc_l)
  perf['mean_test_f1'] = np.mean(test_f1_l)
  perf['mean_feature_imp'] = np.mean(feat_imp_l, axis=0)

  return perf

#############################################################################################################################
def test_vars_simple(vars_df,
                    target_df,
                    pars,
                    n_cv,
                    scoring,
                    refit,
                    dropped_var = [],
                    mdl = xgb.XGBClassifier(),
                    verbose = 1,
                     tst = False,
                     test_df = None,
                     test_target = None):
  """
  fits a model with CV on vars_df & returns performance DF
  variables:
    - vars_df: tarin data
    - target_df: DF or list of train targets
    - pars: hyperpar space
    - scoring: scoring to eval model by & record in output DF
    - refit: score to select best model
    - dropped_var: list of dropped vars (only to record in output DF)
    - mdl: model
    - tst: bool - whether to test best model on a test set
    - test_df: test data
    - test_target
  """
  
  perf = {}
  perf['variables'] = str(list(vars_df))
  perf['dropped_vars'] = str(dropped_var)
  perf['num_vars'] = len(list(vars_df))
  perf['model'] = str(mdl)
  perf['par_space'] = str(pars)
  perf['refit'] = refit
  perf['num_sub_cv'] = n_cv

  clf = mdl
  clf_cv = GridSearchCV(clf,
                        param_grid=pars,
                        scoring=scoring,
                        refit = refit,
                        return_train_score=True,
                        cv=n_cv,
                        verbose=verbose,
                        n_jobs = -1)
  clf_cv.fit(vars_df, list(target_df))
  # print(32)
  best_ind = np.where(clf_cv.cv_results_['mean_test_'+refit] == max(clf_cv.cv_results_['mean_test_'+refit]))[0][0]
  # print(34)
  for key in scoring:
    perf['mean_train_CV_'+key] = clf_cv.cv_results_['mean_test_'+key][best_ind]
  # print(37)
  perf['best_parms'] = str(clf_cv.best_params_)  
  perf['CV_feature_imp'] = str(clf_cv.best_estimator_.feature_importances_)

  if tst:
    preds = clf_cv.predict(test_df)
    perf['test_accuracy'] = accuracy_score(list(test_target), preds)
    perf['test_f1'] = f1_score(list(test_target), preds)
    perf['test_classificaton_report'] = classification_report(list(test_target), preds)

  return perf

#############################################################################################################################
def iter_test_vars(perf_df,
                   train_DF,
                   targetcol,
                   typ,
                   varcols,
                   fld_col,
                   loc, 
                   filename,
                   params,
                   scoring,
                   refit = 'Accuracy',
                   n_main_cv = 4,
                   n_cv = 4,
                   num_drop = 2,
                   tst = False,
                   test_DF = None,
                   drpd_l = [],
                   dropped_var = [],
                   mdl = xgb.XGBClassifier(),
                   verbose = 4
                   ):
  """
  vars:
    -perf_df: performance DF to write results in
    -train_DF: train data with varcols & target & fold col
    -tagetcol: colname of target var
    -typ: "simple" or "model"
    -varcols: list of column names in models
    -fld_col: fold col for test_vars_model
    -loc: string of location to save output
    -filename: name of output file without extension
    -params: hyperpar space
    -scoring: scoring used for model eval
    -refit: scoring to select best model
    -n_main_cv: test_vars_model par: how many mian CV fols
    -n_cv: teat_vars par: # of model fitting CV folds
    -num_drop: number of vars to drop for iteration
        (1: only largest model, 2: largest model + 1 model/each vars dropped...)
    -tst: bool wheteher to test best models on separ test set
    -test_DF: test_data
    -drpd_l: list of lists of dropped variables in models already checked (to avoid double checking)
    -dropped_var: list of dropped vars (only to record in output DF)
    -mdl: model
    -verbose
  """
  for num_vars in range(len(varcols), len(varcols)-num_drop,-1):
    p = 0 #counter
    for i in itertools.combinations(varcols,num_vars):
      p += 1
      vcols = list(i)
      drpd = [f for f in varcols if f not in vcols] + dropped_var
      if drpd not in drpd_l:
        print('-'*100)
        print(len(vcols), '/', p, time.ctime(), vcols)
        if typ == "model":
          perf = test_vars_model(vars_df = train_DF[vcols],
                                  target_df = train_DF[[targetcol]],
                                  pars = params,
                                  n_main_cv = n_main_cv,
                                  n_cv = n_cv,
                                  scoring = scoring,
                                  refit = refit,
                                  dropped_var = drpd,
                                  mdl = mdl,
                                  verbose = verbose,
                                  rnd_fld = train_DF[fld_col])
        elif typ == "simple":
          perf = test_vars_simple(vars_df = train_DF[vcols],
                                  target_df = train_DF[[targetcol]],
                                  pars = params,
                                  n_cv = n_cv,
                                  scoring = scoring,
                                  refit = refit,
                                  dropped_var = drpd,
                                  tst = tst,
                                  test_df = test_DF[vcols],
                                  test_target = test_DF[[targetcol]])
          
        perf_df = perf_df.append(perf, ignore_index = True)
        perf_df.to_excel(loc+filename+'_backup.xlsx')
  if typ == "model":
    perf_df.sort_values(by = 'mean_test_'+refit.lower(), inplace=True, ascending=False)
  elif typ = "simple":
    perf_df.sort_values(by = 'test_'+refit.lower(), inplace=True, ascending=False)
  perf_df.to_excel(loc+filename+'_full.xlsx')

  return perf_df

############################################
############################################
###                                      ###
###                                      ###
###    load data & define parameters     ###
###                                      ###
###                                      ###
############################################
############################################

DF = pd.read_excel('data/processed/stat_model_df_filtered_withfolds.xlsx', index_col=0)
print(DF.shape)
DF.head()

train_DF = DF[DF.test_fold == 0] #930
test_DF = DF[DF.test_fold == 1] #200

print(test_DF.shape, train_DF.shape)
train_target = train_DF['roma']
test_target = test_DF['roma']

f1_scorer = make_scorer(f1_score, average='weighted')
acc_scorer = make_scorer(accuracy_score)
scoring = {'F1': f1_scorer, 'Accuracy': acc_scorer}

############################################
############################################
###                                      ###
###                                      ###
###      feature selection round 1       ###
###                                      ###
###                                      ###
############################################
############################################

############################################
###
###     start with 29 features
###

varcols = ['num_words', 'var_punct', 'ratio_cim_token', 'ratio_dátum_token',
           'ratio_email_token', 'ratio_idő_token', 'ratio_link_token',
           'ratio_name_token', 'ratio_szám_token', 'ratio_telefonszám_token',
           'ratio_település_m_token', 'ratio_település_token', 'MATTR',
           'avg_length_sent', 'avg_length_word', 'ratio_stopw', 'ratio_punct',
           'freq_punct', 'ratio_VERB', 'ratio_PROPN', 'ratio_CONJ',
           'ratio_PRON', 'ratio_PART', 'ratio_NOUN', 'ratio_DET', 'ratio_AUX',
           'ratio_ADV', 'ratio_ADJ', 'ratio_ADP']
print(len(varcols))

default_perf_df = pd.DataFrame(columns=['num_vars',
                                        'dropped_vars',
                                        'variables',
                                        'model',
                                        'parameters',
                                        'feature_imp',
                                        'classification_report',
                                        'test_accuracy',
                                        'test_f1',
                                        'train_accuracy',
                                        'train_f1'])

#building models #29-25
default29_perf_df = var_drop_models(train_DF = train_DF,
                                    train_target = train_target,
                                    test_DF = test_DF,
                                    test_target = test_target,
                                    loc = "analysis/",
                                    filename = "stat_model_default29_perf_df",
                                    varcols = varcols,
                                    drpd_l = [],
                                    perf_df = default_perf_df,
                                    earlier_drops = [],
                                    num_drop = 5,
                                    modl = xgb.XGBClassifier,
                                    params_dict_l = [{"subsample":0.9, "colsample_bytree":0.9, "max_depth":6, "reg_alpha":0.3},
                                                    {"reg_alpha":0.2, "n_estimators":50}],
                                    ):

#############
###
###   looking for most frequently dropped var (#29-25)

default29_perf_df = pd.read_excel('analysis/stat_model_default29_perf_df.xlsx', index_col=0)
print(default29_perf_df.shape)
default29_perf_df.sort_values(by = 'test_accuracy', inplace=True, ascending=False)
default29_perf_df.head()

for tn in range(4):
  select_topn_dropped_bynumval(n = tn+1,
                              perf_df = default29_perf_df)
print("- "*50)

for tn in range(4)
  select_topn_dropped_bymodel(n=tn+1,
                              break_acc=0.57,
                              perf_df = default29_perf_df)

############################################
###
###     continue with 28 features ('ratio_ADJ' dropped)
###

default29_perf_df = pd.read_excel('analysis/stat_model_default29_perf_df.xlsx', index_col = 0)

varcols.remove("ratio_ADJ")

default28_perf_df = default29_perf_df[default29_perf_df.dropped_vars.str.contains('ratio_ADJ')]
default28_perf_df.shape

drpd_l = [set(re.sub(r'[\s\']', '', l[2:-2]).split(',')) for l in default28_perf_df.dropped_vars]

# building models #28-24
default28_perf_df = var_drop_models(train_DF = train_DF,
                                    train_target = train_target,
                                    test_DF = test_DF,
                                    test_target = test_target,
                                    loc = "analysis/",
                                    filename = "stat_model_default28_perf_df",
                                    varcols = varcols,
                                    drpd_l = drpd_l,
                                    perf_df = default28_perf_df,
                                    earlier_drops = ['ratio_ADJ'],
                                    num_drop = 5,
                                    modl = xgb.XGBClassifier,
                                    params_dict_l = [{"subsample":0.9, "colsample_bytree":0.9, "max_depth":6, "reg_alpha":0.3},
                                                    {"reg_alpha":0.2, "n_estimators":50}],
                                    ):

#############
###
###   looking for most frequently dropped var (#29-25)

default28_perf_df = pd.read_excel('analysis/stat_model_default28_perf_df.xlsx', index_col=0)
print(default28_perf_df.shape)
default28_perf_df.sort_values(by = 'test_accuracy', inplace=True, ascending=False)
default28_perf_df.head()

for tn in range(4):
  select_topn_dropped_bynumval(n = tn+1,
                              perf_df = default28_perf_df)
print("- "*50)

for tn in range(3)
  select_topn_dropped_bymodel(n=tn+1,
                              break_acc=0.57,
                              perf_df = default28_perf_df)

############################################
###
###     continue with 28 features ('ratio_ADV' dropped)
###

default28_perf_df = pd.read_excel('analysis/stat_model_default28_perf_df.xlsx', index_col = 0)

varcols.remove("ratio_ADV")
len(varcols)

default27_perf_df = default28_perf_df[default28_perf_df.dropped_vars.str.contains('ratio_ADV')]
default27_perf_df.shape

drpd_l = [set(re.sub(r'[\s\']', '', l[2:-2]).split(',')) for l in default27_perf_df.dropped_vars]

# building models #27-23
default27_perf_df = var_drop_models(train_DF = train_DF,
                                    train_target = train_target,
                                    test_DF = test_DF,
                                    test_target = test_target,
                                    loc = "analysis/",
                                    filename = "stat_model_default27_perf_df",
                                    varcols = varcols,
                                    drpd_l = drpd_l,
                                    perf_df = default27_perf_df,
                                    earlier_drops = ['ratio_ADJ', 'ratio_ADV'],
                                    num_drop = 5,
                                    modl = xgb.XGBClassifier,
                                    params_dict_l = [{"subsample":0.9, "colsample_bytree":0.9, "max_depth":6, "reg_alpha":0.3},
                                                    {"reg_alpha":0.2, "n_estimators":50}],
                                    ):

#############
###
###   looking for most frequently dropped var (#27-23)

default27_perf_df = pd.read_excel('analysis/stat_model_default27_perf_df.xlsx', index_col=0)
print(default27_perf_df.shape)
default27_perf_df.sort_values(by = 'test_accuracy', inplace=True, ascending=False)
default27_perf_df.head()

for tn in range(4):
  select_topn_dropped_bynumval(n = tn+1,
                              perf_df = default27_perf_df)
print("- "*50)

for tn in range(4)
  select_topn_dropped_bymodel(n=tn+1,
                              break_acc=0.57,
                              perf_df = default27_perf_df)

############################################
###
###     continue with 26 features ('ratio_idő_token' dropped)
###

default27_perf_df = pd.read_excel('analysis/stat_model_default27_perf_df.xlsx', index_col = 0)

varcols.remove('ratio_idő_token')
len(varcols)

default26_perf_df = default27_perf_df[default27_perf_df.dropped_vars.str.contains('ratio_idő_token')]

drpd_l = [set(re.sub(r'[\s\']', '', l[2:-2]).split(',')) for l in default26_perf_df.dropped_vars]
len(drpd_l)

# building models #26-22
default26_perf_df = var_drop_models(train_DF = train_DF,
                                    train_target = train_target,
                                    test_DF = test_DF,
                                    test_target = test_target,
                                    loc = "analysis/",
                                    filename = "stat_model_default26_perf_df",
                                    varcols = varcols,
                                    drpd_l = drpd_l,
                                    perf_df = default26_perf_df,
                                    earlier_drops = ['ratio_ADJ', 'ratio_ADV', 'ratio_idő_token'],
                                    num_drop = 5,
                                    modl = xgb.XGBClassifier,
                                    params_dict_l = [{"subsample":0.9, "colsample_bytree":0.9, "max_depth":6, "reg_alpha":0.3},
                                                    {"reg_alpha":0.2, "n_estimators":50}],
                                    ):



#############
###
###   looking for most frequently dropped var (#26-22)

default26_perf_df = pd.read_excel('analysis/stat_model_default26_perf_df.xlsx', index_col=0)
print(default26_perf_df.shape)
default26_perf_df.sort_values(by = 'test_accuracy', inplace=True, ascending=False)
default26_perf_df.head()

for tn in range(4):
  select_topn_dropped_bynumval(n = tn+1,
                              perf_df = default26_perf_df)
print("- "*50)

for tn in range(4)
  select_topn_dropped_bymodel(n=tn+1,
                              break_acc=0.57,
                              perf_df = default26_perf_df)

############################################
###
###     continue with 25 features ('ratio_dátum_token' dropped)
###

default26_perf_df = pd.read_excel('analysis/stat_model_default26_perf_df.xlsx', index_col = 0)

varcols.remove('ratio_dátum_token')
len(varcols)

default25_perf_df = default26_perf_df[default26_perf_df.dropped_vars.str.contains('ratio_dátum_token')]
default25_perf_df.shape

drpd_l = [set(re.sub(r'[\s\']', '', l[2:-2]).split(',')) for l in default25_perf_df.dropped_vars]

# building models #25-21
default25_perf_df = var_drop_models(train_DF = train_DF,
                                    train_target = train_target,
                                    test_DF = test_DF,
                                    test_target = test_target,
                                    loc = "analysis/",
                                    filename = "stat_model_default25_perf_df",
                                    varcols = varcols,
                                    drpd_l = drpd_l,
                                    perf_df = default25_perf_df,
                                    earlier_drops = ['ratio_ADJ', 'ratio_ADV', 'ratio_idő_token', 'ratio_dátum_token'],
                                    num_drop = 5,
                                    modl = xgb.XGBClassifier,
                                    params_dict_l = [{"subsample":0.9, "colsample_bytree":0.9, "max_depth":6, "reg_alpha":0.3},
                                                    {"reg_alpha":0.2, "n_estimators":50}],
                                    ):



#############
###
###   looking for most frequently dropped var (#25-21)

default25_perf_df = pd.read_excel('analysis/stat_model_default25_perf_df.xlsx', index_col=0)
print(default25_perf_df.shape)
default25_perf_df.sort_values(by = 'test_accuracy', inplace=True, ascending=False)
default25_perf_df.head()

for tn in range(4):
  select_topn_dropped_bynumval(n = tn+1,
                              perf_df = default25_perf_df)
print("- "*50)

for tn in range(3)
  select_topn_dropped_bymodel(n=tn+1,
                              break_acc=0.57,
                              perf_df = default25_perf_df)

############################################
###
###     continue with 24 features ('ratio_stopw' dropped)
###

default25_perf_df = pd.read_excel('analysis/stat_model_default25_perf_df.xlsx', index_col = 0)

varcols.remove('ratio_stopw')
print(len(varcols))

default24_perf_df = default25_perf_df[default25_perf_df.dropped_vars.str.contains('ratio_stopw')]

drpd_l = [set(re.sub(r'[\s\']', '', l[2:-2]).split(',')) for l in default24_perf_df.dropped_vars]
print(len(drpd_l))

# building models #24-20
default24_perf_df = var_drop_models(train_DF = train_DF,
                                    train_target = train_target,
                                    test_DF = test_DF,
                                    test_target = test_target,
                                    loc = "analysis/",
                                    filename = "stat_model_default24_perf_df",
                                    varcols = varcols,
                                    drpd_l = drpd_l,
                                    perf_df = default24_perf_df,
                                    earlier_drops = ['ratio_ADJ', 'ratio_ADV', 'ratio_idő_token', 'ratio_dátum_token', 'ratio_stopw'],
                                    num_drop = 5,
                                    modl = xgb.XGBClassifier,
                                    params_dict_l = [{"subsample":0.9, "colsample_bytree":0.9, "max_depth":6, "reg_alpha":0.3},
                                                    {"reg_alpha":0.2, "n_estimators":50}],
                                    ):

#############
###
###   looking for most frequently dropped var (#24-20)

default24_perf_df = pd.read_excel('analysis/stat_model_default24_perf_df.xlsx', index_col=0)
print(default24_perf_df.shape)
default24_perf_df.sort_values(by = 'test_accuracy', inplace=True, ascending=False)
default24_perf_df.head()

for tn in range(4):
  select_topn_dropped_bynumval(n = tn+1,
                              perf_df = default24_perf_df)
print("- "*50)

for tn in range(3)
  select_topn_dropped_bymodel(n=tn+1,
                              break_acc=0.57,
                              perf_df = default24_perf_df)

############################################
###
###     continue with 23 features ('ratio_Noun' dropped)
###

default24_perf_df = pd.read_excel('analysis/stat_model_default24_perf_df.xlsx', index_col = 0)

varcols.remove('ratio_NOUN')
print(len(varcols))

default23_perf_df = default24_perf_df[default24_perf_df.dropped_vars.str.contains('ratio_NOUN')]

drpd_l = [set(re.sub(r'[\s\']', '', l[2:-2]).split(',')) for l in default23_perf_df.dropped_vars]
print(len(drpd_l))

# building models #23-19
default23_perf_df = var_drop_models(train_DF = train_DF,
                                    train_target = train_target,
                                    test_DF = test_DF,
                                    test_target = test_target,
                                    loc = "analysis/",
                                    filename = "stat_model_default23_perf_df",
                                    varcols = varcols,
                                    drpd_l = drpd_l,
                                    perf_df = default23_perf_df,
                                    earlier_drops = ['ratio_ADJ', 'ratio_ADV', 'ratio_idő_token', 'ratio_dátum_token', 'ratio_stopw', 'ratio_NOUN'],
                                    num_drop = 5,
                                    modl = xgb.XGBClassifier,
                                    params_dict_l = [{"subsample":0.9, "colsample_bytree":0.9, "max_depth":6, "reg_alpha":0.3},
                                                    {"reg_alpha":0.2, "n_estimators":50}],
                                    ):



#############
###
###   looking for most frequently dropped var (#23-19)

default23_perf_df = pd.read_excel('analysis/stat_model_default23_perf_df.xlsx', index_col=0)
print(default23_perf_df.shape)
default23_perf_df.sort_values(by = 'test_accuracy', inplace=True, ascending=False)
default23_perf_df.head()

for tn in range(4):
  select_topn_dropped_bynumval(n = tn+1,
                              perf_df = default23_perf_df)
print("- "*50)

for tn in range(3)
  select_topn_dropped_bymodel(n=tn+1,
                              break_acc=0.57,
                              perf_df = default23_perf_df)

############################################
###
###     continue with 22 features ('avg_length_word' dropped)
###

default23_perf_df = pd.read_excel('analysis/stat_model_default23_perf_df.xlsx', index_col = 0)

varcols.remove('avg_length_word')
print(len(varcols))

default22_perf_df = default23_perf_df[default23_perf_df.dropped_vars.str.contains('avg_length_word')]

drpd_l = [set(re.sub(r'[\s\']', '', l[2:-2]).split(',')) for l in default22_perf_df.dropped_vars]
print(len(drpd_l))

# building models #22-18
default22_perf_df = var_drop_models(train_DF = train_DF,
                                    train_target = train_target,
                                    test_DF = test_DF,
                                    test_target = test_target,
                                    loc = "analysis/",
                                    filename = "stat_model_default22_perf_df",
                                    varcols = varcols,
                                    drpd_l = drpd_l,
                                    perf_df = default22_perf_df,
                                    earlier_drops = ['ratio_ADJ', 'ratio_ADV', 'ratio_idő_token', 'ratio_dátum_token',
                                                      'ratio_stopw', 'ratio_NOUN', 'avg_length_word'],
                                    num_drop = 5,
                                    modl = xgb.XGBClassifier,
                                    params_dict_l = [{"subsample":0.9, "colsample_bytree":0.9, "max_depth":6, "reg_alpha":0.3},
                                                    {"reg_alpha":0.2, "n_estimators":50}],
                                    ):



#############
###
###   looking for most frequently dropped var (#22-18)

default22_perf_df = pd.read_excel('analysis/stat_model_default22_perf_df.xlsx', index_col=0)
print(default22_perf_df.shape)
default22_perf_df.sort_values(by = 'test_accuracy', inplace=True, ascending=False)
default22_perf_df.head()

for tn in range(4):
  select_topn_dropped_bynumval(n = tn+1,
                              perf_df = default22_perf_df)
print("- "*50)

for tn in range(3)
  select_topn_dropped_bymodel(n=tn+1,
                              break_acc=0.57,
                              perf_df = default22_perf_df)

############################################
###
###     continue with 21 features ('afreq_punct' dropped)
###

default22_perf_df = pd.read_excel('analysis/stat_model_default22_perf_df.xlsx', index_col = 0)

varcols.remove('freq_punct')
print(len(varcols))

default21_perf_df = default22_perf_df[default22_perf_df.dropped_vars.str.contains('freq_punct')]

drpd_l = [set(re.sub(r'[\s\']', '', l[2:-2]).split(',')) for l in default21_perf_df.dropped_vars]
print(len(drpd_l))

# building models #21-17
default21_perf_df = var_drop_models(train_DF = train_DF,
                                    train_target = train_target,
                                    test_DF = test_DF,
                                    test_target = test_target,
                                    loc = "analysis/",
                                    filename = "stat_model_default21_perf_df",
                                    varcols = varcols,
                                    drpd_l = drpd_l,
                                    perf_df = default21_perf_df,
                                    earlier_drops = ['ratio_ADJ', 'ratio_ADV', 'ratio_idő_token', 'ratio_dátum_token',
                                                      'ratio_stopw', 'ratio_NOUN', 'avg_length_word', 'freq_punct'],
                                    num_drop = 5,
                                    modl = xgb.XGBClassifier,
                                    params_dict_l = [{"subsample":0.9, "colsample_bytree":0.9, "max_depth":6, "reg_alpha":0.3},
                                                    {"reg_alpha":0.2, "n_estimators":50}],
                                    ):

#############
###
###   looking for most frequently dropped var (#21-17)

default21_perf_df = pd.read_excel('analysis/stat_model_default21_perf_df.xlsx', index_col=0)
print(default21_perf_df.shape)
default21_perf_df.sort_values(by = 'test_accuracy', inplace=True, ascending=False)
default21_perf_df.head()

for tn in range(4):
  select_topn_dropped_bynumval(n = tn+1,
                              perf_df = default21_perf_df)
print("- "*50)

for tn in range(3)
  select_topn_dropped_bymodel(n=tn+1,
                              break_acc=0.57,
                              perf_df = default21_perf_df)

############################################
###
###     continue with 20 features ('ratio_email_token' dropped)
###

default21_perf_df = pd.read_excel('analysis/stat_model_default21_perf_df.xlsx', index_col = 0)

varcols.remove('ratio_email_token')
print(len(varcols))

default20_perf_df = default21_perf_df[default21_perf_df.dropped_vars.str.contains('ratio_email_token')]

drpd_l = [set(re.sub(r'[\s\']', '', l[2:-2]).split(',')) for l in default20_perf_df.dropped_vars]
print(len(drpd_l))

# building models #20-16
default20_perf_df = var_drop_models(train_DF = train_DF,
                                    train_target = train_target,
                                    test_DF = test_DF,
                                    test_target = test_target,
                                    loc = "analysis/",
                                    filename = "stat_model_default20_perf_df",
                                    varcols = varcols,
                                    drpd_l = drpd_l,
                                    perf_df = default20_perf_df,
                                    earlier_drops = ['ratio_ADJ', 'ratio_ADV', 'ratio_idő_token', 'ratio_dátum_token',
                                                      'ratio_stopw', 'ratio_NOUN', 'avg_length_word', 'freq_punct', 'ratio_email_token'],
                                    num_drop = 5,
                                    modl = xgb.XGBClassifier,
                                    params_dict_l = [{"subsample":0.9, "colsample_bytree":0.9, "max_depth":6, "reg_alpha":0.3},
                                                    {"reg_alpha":0.2, "n_estimators":50}],
                                    ):

#############
###
###   looking for most frequently dropped var (#20-16)

default20_perf_df = pd.read_excel('analysis/stat_model_default20_perf_df.xlsx', index_col=0)
print(default20_perf_df.shape)
default20_perf_df.sort_values(by = 'test_accuracy', inplace=True, ascending=False)
default20_perf_df.head()

for tn in range(4):
  select_topn_dropped_bynumval(n = tn+1,
                              perf_df = default20_perf_df)
print("- "*50)

for tn in range(3)
  select_topn_dropped_bymodel(n=tn+1,
                              break_acc=0.57,
                              perf_df = default20_perf_df)

############################################
###
###     continue with 19 features ('ratio_AUX' dropped)
###

default20_perf_df = pd.read_excel('analysis/stat_model_default20_perf_df.xlsx', index_col = 0)

varcols.remove('ratio_AUX')
print(len(varcols))

default19_perf_df = default20_perf_df[default20_perf_df.dropped_vars.str.contains('ratio_AUX')]

drpd_l = [set(re.sub(r'[\s\']', '', l[2:-2]).split(',')) for l in default19_perf_df.dropped_vars]
print(len(drpd_l))

# building models #19-15
default19_perf_df = var_drop_models(train_DF = train_DF,
                                    train_target = train_target,
                                    test_DF = test_DF,
                                    test_target = test_target,
                                    loc = "analysis/",
                                    filename = "stat_model_default19_perf_df",
                                    varcols = varcols,
                                    drpd_l = drpd_l,
                                    perf_df = default19_perf_df,
                                    earlier_drops = ['ratio_ADJ', 'ratio_ADV', 'ratio_idő_token', 'ratio_dátum_token',
                                                      'ratio_stopw', 'ratio_NOUN', 'avg_length_word', 'freq_punct',
                                                      'ratio_email_token', 'ratio_AUX'],
                                    num_drop = 5,
                                    modl = xgb.XGBClassifier,
                                    params_dict_l = [{"subsample":0.9, "colsample_bytree":0.9, "max_depth":6, "reg_alpha":0.3},
                                                    {"reg_alpha":0.2, "n_estimators":50}],
                                    ):

#############
###
###   looking for most frequently dropped var (#19-15)

default19_perf_df = pd.read_excel('analysis/stat_model_default19_perf_df.xlsx', index_col=0)
print(default19_perf_df.shape)
default19_perf_df.sort_values(by = 'test_accuracy', inplace=True, ascending=False)
default19_perf_df.head()

############################################
############################################
###                                      ###
###          feature sel round 2         ###
###                                      ###
############################################
############################################

############################################
###
###     continue with 16 features (16-15)
###
# list of best featurset so far
varcols = ['num_words',
            'var_punct',
            'ratio_link_token',
            'ratio_name_token',
            'ratio_szám_token',
            'ratio_telefonszám_token',
            'ratio_település_m_token',
            'ratio_település_token',
            'avg_length_sent',
            'ratio_punct',
            'ratio_VERB',
            'ratio_PROPN',
            'ratio_PRON',
            'ratio_PART',
            'ratio_DET',
            'ratio_ADP']
print(len(varcols))

perf_df = pd.DataFrame()
drpd_l = []

#defining search space

params = {
        'min_child_weight': [1, 2, 4],
        'subsample': [0.8, 0.9, 1],
        'colsample_bytree': [0.9, 1],
        'colsample_bynode': [0.9, 1],
        'max_depth': [3, 6, 9],
        'learning_rate' : [0.3, 0.1],
        'reg_alpha':[0.1, 0.2, 0.3],
        'n_estimators' : [50, 100]
        }

#redifining train_DF to whole train set 
train_DF = DF[DF.test_fold != 2]

print(test_DF.shape, train_DF.shape)
train_target = train_DF['roma']
test_target = test_DF['roma']

perf_df = iter_test_vars(perf_df = perf_df,
                          train_DF = train_DF,
                          targetcol = "roma",
                          typ = "model",
                          varcols = varcols,
                          fld_col = "CV_all_4f",
                          loc = "analysis/", 
                          filename = "stat_model_best_vars_CV_16_15_fixedfold",
                          params = params,
                          scoring = scoring,
                          refit = 'Accuracy',
                          n_main_cv = 4,
                          n_cv = 4,
                          num_drop = 2,
                          drpd_l = drpd_l,
                          mdl = xgb.XGBClassifier(),
                          verbose = 4
                          )
perf_df.head()

############################################
###
###     continue with 15 features (15-14-13)
###
varcols.remove("ratio_település_m_token")
len(varcols)

perf_df =pd.DataFrame()
drpd_l = []

perf_df = iter_test_vars(perf_df = perf_df,
                          train_DF = train_DF,
                          targetcol = "roma",
                          typ = "model",
                          varcols = varcols,
                          fld_col = "CV_all_4f",
                          loc = "analysis/", 
                          filename = "stat_model_best_vars_CV_15_14_13_fixedfold",
                          params = params,
                          scoring = scoring,
                          refit = 'Accuracy',
                          n_main_cv = 4,
                          n_cv = 4,
                          num_drop = 3,
                          drpd_l = drpd_l,
                          mdl = xgb.XGBClassifier(),
                          verbose = 4
                          )
perf_df.head()

############################################
###
###     continue with 13 features (13-12-11)
###
varcols.remove('ratio_link_token')
varcols.remove('avg_length_sent')
len(varcols)

perf_df =pd.DataFrame()
drpd_l = []

perf_df = iter_test_vars(perf_df = perf_df,
                          train_DF = train_DF,
                          targetcol = "roma",
                          typ = "model",
                          varcols = varcols,
                          fld_col = "CV_all_4f",
                          loc = "analysis/", 
                          filename = "stat_model_best_vars_CV_13_12_11_fixedfold",
                          params = params,
                          scoring = scoring,
                          refit = 'Accuracy',
                          n_main_cv = 4,
                          n_cv = 4,
                          num_drop = 3,
                          drpd_l = drpd_l,
                          mdl = xgb.XGBClassifier(),
                          verbose = 4
                          )

perf_df.head()

############################################
###
###     continue with 11 features (11-10-9)
###
varcols.remove("ratio_telefonszám_token")
varcols.remove("ratio_település_token")
len(varcols)

perf_df =pd.DataFrame()
drpd_l = []

perf_df = iter_test_vars(perf_df = perf_df,
                          train_DF = train_DF,
                          targetcol = "roma",
                          typ = "model",
                          varcols = varcols,
                          fld_col = "CV_all_4f",
                          loc = "analysis/", 
                          filename = "stat_model_best_vars_CV_11_10_9_fixedfold",
                          params = params,
                          scoring = scoring,
                          refit = 'Accuracy',
                          n_main_cv = 4,
                          n_cv = 4,
                          num_drop = 2,
                          drpd_l = drpd_l,
                          mdl = xgb.XGBClassifier(),
                          verbose = 4
                          )

############################################
###
###     continue with 9 features (9-8-7)
###
varcols.remove("ratio_name_token")
varcols.remove("ratio_PART")
len(varcols)

perf_df =pd.DataFrame()
drpd_l = []

perf_df = iter_test_vars(perf_df = perf_df,
                          train_DF = train_DF,
                          targetcol = "roma",
                          typ = "model",
                          varcols = varcols,
                          fld_col = "CV_all_4f",
                          loc = "analysis/", 
                          filename = "stat_model_best_vars_CV_9_8_7_fixedfold",
                          params = params,
                          scoring = scoring,
                          refit = 'Accuracy',
                          n_main_cv = 4,
                          n_cv = 4,
                          num_drop = 3,
                          drpd_l = drpd_l,
                          mdl = xgb.XGBClassifier(),
                          verbose = 4
                          )

perf_df.head()

##############################################
##############################################
###                                        ###
###    fit model on selected feat set      ###
###                                        ###
##############################################
##############################################
test_DF = DF[DF.test_fold == 2] #redifining test DF to model performance test set (not seen previously for feature sel & model training)
#best feat set
varcols = ['num_words',
           'var_punct',
           'ratio_szám_token',
           'ratio_punct',
           'ratio_VERB',
           'ratio_PROPN',
           'ratio_PRON',
           'ratio_DET',
           'ratio_ADP']
len(varcols)


clf = xgb.XGBClassifier()
clf_cv = GridSearchCV(clf,
                      param_grid=params,
                      scoring=scoring,
                      refit = "Accuracy",
                      return_train_score=True,
                      cv=5,
                      verbose=verbose,
                      n_jobs = -1)
clf_cv.fit(train_DF[varcols], train_DF['roma'])

print(clf_cv.best_params_)  
print(clf_cv.best_estimator_.feature_importances_)

mdl9 = clf_cv.best_estimator_
joblib.dump(mdl9, "models/stat_model_9vars")

##############################################
##############################################
###                                        ###
###           model performance            ###
###                                        ###
##############################################
##############################################

############################################
###
###       overall
###
preds = mdl9.predict(test_DF[varcols])
print('acc:', accuracy_score(test_DF['roma'], preds))
print('f1:', f1_score(test_DF['roma'], preds))
print(classification_report(test_DF['roma'], preds))

##################################
###
###  feature importance
###

imps = ['weight', 'gain', 'total_gain', 'cover', 'total_cover']
labels = []
vals = {}
vals_n = {}
for impname in imps:
  imp = mdl9.get_booster().get_score(importance_type= impname)
  if len(labels) == 0:
    labels = list(imp.keys())
  vals[impname] = list(imp.values())
  vals_n[impname] = np.divide(list(imp.values()),sum(imp.values()))
  print(impname, imp)

#########
###
###   feature imp plots
###
#########

### all

x = np.arange(len(labels))
width = 0.15
mins = [-2,-1,0,1,2]

rects_l = []
fig, ax = plt.subplots(figsize=(10, 5))

rects0 = ax.bar(x + width*mins[0], vals_n[imps[0]], width, label=imps[0])
rects1 = ax.bar(x + width*mins[1], vals_n[imps[1]], width, label=imps[1])
rects2 = ax.bar(x + width*mins[2], vals_n[imps[2]], width, label=imps[2])
rects3 = ax.bar(x + width*mins[3], vals_n[imps[3]], width, label=imps[3])
rects4 = ax.bar(x + width*mins[4], vals_n[imps[4]], width, label=imps[4])

ax.set_ylabel('normalised scores')
ax.set_title('feature importances')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()



fig.tight_layout()

plt.show()

### separately

for impname in imps:
  fig, ax = plt.subplots(figsize=(10,10))
  xgb.plot_importance(mdl9, max_num_features=9, height=0.5, ax=ax,importance_type= impname, title='feature importance: '+impname)
  plt.show()

###########
###
###  shap
###
###########
#imp
shap_values = shap.TreeExplainer(mdl9).shap_values(train_DF[varcols])
shap.summary_plot(shap_values, train_DF[varcols], plot_type="bar")
print(shap_values.shape)
for ix, v in enumerate(np.mean(abs(shap_values), axis = 0)):
  print(varcols[ix], v)

#direction
shap.summary_plot(shap_values, train_DF[varcols])

###########
###
###  feature depletion
###
###########

perf_df =pd.DataFrame()
drpd_l = []

# drop each feature & build model
perf_df = iter_test_vars(perf_df = perf_df,
                          train_DF = train_DF,
                          targetcol = "roma",
                          typ = "simple",
                          varcols = varcols,
                          fld_col = None,
                          loc = "analysis/", 
                          filename = "stat_model_var_drop_CV_performance_9_8",
                          params = params,
                          scoring = scoring,
                          refit = 'Accuracy',
                          n_cv = 5,
                          num_drop = 2,
                          tst = True,
                          test_df = test_DF,
                          drpd_l = drpd_l,
                          dropped_var = [],
                          mdl = xgb.XGBClassifier(),
                          verbose = 4
                          )

# drop each 'type' of feture
varcol_l = [['ratio_punct', 'ratio_VERB', 'ratio_PROPN', 'ratio_PRON', 'ratio_DET', 'ratio_ADP'],
            ['num_words', 'var_punct', 'ratio_szám_token']]
for n in range(len(varcol_l)):
  perf_df = iter_test_vars(perf_df = perf_df,
                          train_DF = train_DF,
                          targetcol = "roma",
                          typ = "simple",
                          varcols = varcol_l[n],
                          fld_col = None,
                          loc = "analysis/", 
                          filename = "stat_model_var_drop_CV_performance_9_8",
                          params = params,
                          scoring = scoring,
                          refit = 'Accuracy',
                          n_cv = 5,
                          num_drop = 2,
                          tst = True,
                          test_df = test_DF,
                          drpd_l = drpd_l,
                          dropped_var = varcol_l[-1-n], #other element of varcol_l is dropped
                          mdl = xgb.XGBClassifier(),
                          verbose = 4
                          )  
  
perf_df_sm = perf_df[['dropped_vars', 'test_accuracy',	'test_f1',	'mean_train_CV_Accuracy',	'mean_train_CV_F1']]
perf_df_sm.to_excel('analysis/stat_model_var_drop_CV_performance_9_8_type_small.xlsx')

############################################
###
###       model performance by gender
###

##
## females
##
print('acc:', accuracy_score(test_DF['roma'][test_DF.female == 1], preds[test_DF.female == 1]))
print('f1:', f1_score(test_DF['roma'][test_DF.female == 1], preds[test_DF.female == 1]))
print(classification_report(test_DF['roma'][test_DF.female == 1], preds[test_DF.female == 1]))

##
## males
##
print('acc:', accuracy_score(test_DF['roma'][test_DF.female == 0], preds[test_DF.female == 0]))
print('f1:', f1_score(test_DF['roma'][test_DF.female == 0], preds[test_DF.female == 0]))
print(classification_report(test_DF['roma'][test_DF.female == 0], preds[test_DF.female == 0]))

#######
##
##   separate model by gender (same hyperpars)
##
#######

train_rows_fem = train_DF.female == 1
test_rows_fem = test_DF.female == 1

##
## females
##

mdl9_fem = xgb.XGBClassifier(colsample_bynode= 0.9,
                            colsample_bytree= 1,
                            learning_rate= 0.1,
                            max_depth= 6,
                            min_child_weight= 1,
                            n_estimators= 50,
                            reg_alpha= 0.3,
                            subsample= 0.8)
mdl9_fem.fit(train_DF[varcols][train_rows_fem], train_DF['roma'][train_rows_fem])

preds_fem = mdl9_fem.predict(test_DF[varcols][test_rows_fem])
print('acc:', accuracy_score(test_DF['roma'][test_rows_fem], preds_fem))
print('f1:', f1_score(test_DF['roma'][test_rows_fem], preds_fem))
print(classification_report(test_DF['roma'][test_rows_fem], preds_fem))

##
## males
##

train_rows_male = train_DF.female == 0
test_rows_male = test_DF.female == 0
mdl9_male = xgb.XGBClassifier(colsample_bynode= 0.9,
                         colsample_bytree= 1,
                         learning_rate= 0.1,
                         max_depth= 6,
                         min_child_weight= 1,
                         n_estimators= 50,
                         reg_alpha= 0.3,
                         subsample= 0.8)
mdl9_male.fit(train_DF[varcols][train_rows_male], train_DF['roma'][train_rows_male])

preds_male = mdl9_male.predict(test_DF[varcols][test_rows_male])
print('acc:', accuracy_score(test_DF['roma'][test_rows_male], preds_male))
print('f1:', f1_score(test_DF['roma'][test_rows_male], preds_male))
print(classification_report(test_DF['roma'][test_rows_male], preds_male))