import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

import shap
import time
import xgboost as xgb

######################################################
###                                                ###
###        load data & define parameters           ###
###                                                ###
######################################################
RAW = pd.read_excel('data/processed/cleaned_v5_withfolds.xlsx', index_col=0)

DF = RAW[RAW.no_real_resp == 0][['town',
                                 'wave',
                                 'treated',
                                 'roma',
                                 'female',
                                 'response_cleaned_lower',
                                 'response_cleaned_lemma',
                                 'response_cleaned_withoutstop',
                                 'response_cleaned_withoutstop_lemma',
                                 'response_cleaned_pos',
                                 'response_cleaned_lower_nopunct',
                                 'response_cleaned_lemma_nopunct',
                                 'response_cleaned_withoutstop_nopunct',
                                 'response_cleaned_withoutstop_lemma_nopunct',
                                 'test_fold',
                                 'CV_all_4f']]
DF.head()

DF.to_excel('data/processed/ngram_df_filtered.xlsx')

train_DF = DF[DF.test_fold != 2]
test_DF = DF[DF.test_fold == 2]

print(test_DF.shape, train_DF.shape)
train_target = train_DF['roma']
test_target = test_DF['roma']

f1_scorer = make_scorer(f1_score, average='weighted')
acc_scorer = make_scorer(accuracy_score)
scoring = {'F1': f1_scorer, 'Accuracy': acc_scorer}

######################################################
###                                                ###
###            defining functions                  ###
###                                                ###
######################################################
def combined_vocab_maker(df, textcol, classcol, maxfeatures,
                         mindf, maxdf, ngramrange, tokenpattern, szorz,
                         plusboth = True, return_separate_vocabs = False):
  """
  makes a vocabulary of the most distinctive n-grams of a corpus between groups
  variables:
    - df: DF with at leats classcol & taextcol
    - textcol: name  of column with text
    - classcol: name of column with class of datapoints
    - maxfeatures: number of most frequent features to consider by class
      (drops from these those that are frequent in other class(es) too)
    - mindf: min_df par of TfidfVectorizer
    - maxdf: max_df par of TfidfVectorizer
    - ngramrange: ngram_range par of TfidfVectorizer
    - tokenpattern: token_pattern par of TfidfVectorizer
    - szorz: multiplicator to use when checking whether a class specific token
      is also frequent in the other classes (max_features=maxfeatures*szorz)
    - plusboth: bool - if True the most frequent tokens on the whole corpus
      are also included in the output
    - return_separate_vocabs: bool - if True also outputs separate vocabs for each class 
  """

  vocab = []

  if return_separate_vocabs:
    vocabs = {}

  for val in set(df[classcol]):
    vect_t = TfidfVectorizer(min_df=mindf,
                            max_df=maxdf,
                            ngram_range=ngramrange,
                            max_features=maxfeatures,
                            token_pattern=tokenpattern,
                            use_idf=True,
                            smooth_idf=True,
                            sublinear_tf=True)
    t_X=vect_t.fit_transform(df[df[classcol]==val][textcol])

    vect_c = TfidfVectorizer(min_df=mindf,
                            max_df=maxdf,
                            ngram_range=ngramrange,
                            max_features=int(maxfeatures*szorz),
                            token_pattern=tokenpattern,
                            use_idf=True,
                            smooth_idf=True,
                            sublinear_tf=True)
    c_X=vect_c.fit_transform(df[df[classcol]!=val][textcol])

    vocab += [s for s in vect_t.get_feature_names() if s not in vect_c.get_feature_names()]
    
    del(t_X)
    del(c_X)

    if return_separate_vocabs:
      vocabs[val] = [s for s in vect_t.get_feature_names() if s not in vect_c.get_feature_names()]
    
  if plusboth:
    vect_all = TfidfVectorizer(min_df=mindf,
                              max_df=maxdf,
                              ngram_range=ngramrange,
                              max_features=maxfeatures,
                              token_pattern=tokenpattern,
                              use_idf=True,
                              smooth_idf=True,
                              sublinear_tf=True)
    all_X=vect_all.fit_transform(df[textcol])

    vocab += vect_all.get_feature_names()
    del(all_X)
    if return_separate_vocabs:
      vocabs['all'] = vect_all.get_feature_names()

  vocab = set(vocab)

  if return_separate_vocabs:
    return vocab, vocabs
  else:
    return vocab

def CV_combinedvocab(train_DF,
                     train_target,
                     loc,
                     szorz_l,
                     textcol_l,
                     mindf_l,
                     maxdf_l,
                     ngramrange_l,
                     maxfeat_l,
                     file_n,
                     tokenp_l,
                     subsam_l,
                     colsamt_l,
                     colsamn_l,
                     depth_l,
                     alph_l,
                     estims_l,
                     classcol_n = 'roma',
                     fold_col = 'CV_all_4f',
                     return_last = False,
                     onlycount = False,
                     save = True,
                     randratio = 1
                     ):
  """
  CV search for best fitting model with combined_vocab_maker & XGB
  saves performance dataframes to given location (in multiple chunks for easier paralellization)
  variables:
    - train_DF: DF with training data
    - train_target: corresponding target vector
    - loc: loaction to save outputs
    - szorz_l: list of szorz hyperpars in combined_vocab_maker
    - textcol_l: list of textcol hyperpars in combined_vocab_maker
    - mindf_l: list of mindf hyperpars in combined_vocab_maker
    - maxdf_l: list of maxdf hyperpars in combined_vocab_maker
    - ngramrange_l: list of ngramrange hyperpars in combined_vocab_maker
    - maxfeat_l: list of maxfeatures hyperpars in combined_vocab_maker
    - file_n: output files name start
    - tokenp_l: list of tokenpatterns
    - subsam_l: list of subasmple hyperpars
    - colsamt_l: colsample by tree hyperpars
    - colsamn_l: coldsample by node híperpars
    - depth_l: maxdepth hyperpars
    - alph_l: reg_alpha hyperpars
    - estims_l: n_estimators hyperpars
    - classcol_n: name of class column
    - fold_col: name of fold column
    - return_last: bool-whether to return last performance DF
    - onlycount: bool-whether to only count the # of DF-s & hyperpar points
    - save: bool- wheter to save outpus
    - randratio: ratio of hyperpar gridpoint to randomly test
  """
  locs = locals()
  db_ossz = 1
  for key in locs:
    if type(locs[key]) == list:
      # print(key, locs[key], len(locs[key]))
      db_ossz = db_ossz*len(locs[key])
  db_df = len(szorz_l)*len(textcol_l)*len(mindf_l)*len(maxdf_l)*len(ngramrange_l)*len(maxfeat_l)

  print('\n', '- '*50, '\nÖsszes:', db_ossz*randratio, 'DF-ek:', db_df*randratio, 'db/df:', db_ossz/db_df, '\n', '- '*50)
  
  if not onlycount:

    
    s_df = 0
    for szorz in szorz_l:
      for textcol in textcol_l:
        for mindf in mindf_l:
          for maxdf in maxdf_l:          
            for ngramrange in ngramrange_l:
              for maxfeat in maxfeat_l:

                files = os.listdir(loc)
                s_df += 1
                perf_df = pd.DataFrame(columns= ['mean_test_acc',
                                                'std_test_acc',
                                                'mean_test_f1',
                                                'std_test_f1',
                                                 'min_test_acc',
                                                 'min_test_f1',
                                                'test_accs',
                                                'test_f1s',
                                                'mean_train_acc',
                                                'std_train_acc',
                                                'mean_train_f1',
                                                'std_train_f1',
                                                'train_accs',
                                                'trainf1s',
                                                'szorz',
                                                'textcol',
                                                'mindf',
                                                'maxdf',
                                                'ngram_range',
                                                'max_features',
                                                'token_pattern',
                                                'subsample',
                                                'colsample_bytree',
                                                'colsample_bynode',
                                                'max_depth',
                                                'reg_alpha',
                                                'n_estimators'])
                
                s = 0
                print('\n', '#'*100,'\n ##  ', s_df, '/', db_df, time.ctime(), 'maxdf:', maxdf,
                      'mindf:', mindf, 'max featurs:', maxfeat, 'ngram range:',
                      ngramrange, textcol.split('_')[-1], '\n', '#'*100)
                
                startt = time.time()
                filename = (file_n+
                            str(textcol)+'_'+str(szorz)+
                            "_maxfeat"+str(maxfeat)+
                            '_ngramrange'+str(ngramrange)+
                            '_mindf'+str(mindf)+
                            '_maxdf'+str(maxdf)+
                            '.xlsx')
                if filename not in files:

                  if 'backup_'+filename in files:
                    perf_df = pd.read_excel(loc+'/backup_'+filename)
                    print('load backup', len(perf_df))

                  for tokenp in tokenp_l:
                    for subsam in subsam_l:
                      for colsamt in colsamt_l:
                        for colsamn in colsamn_l:
                          for depth in depth_l:
                            for alph in alph_l:
                              for estims in estims_l:
                                s += 1
                                if s%100==0:
                                  elaps = time.time()-startt
                                  print(s, time.ctime(), 'elapsed:', elaps, '|s/db:', elaps/s)
                                  if save:
                                    perf_df.to_excel(loc+'/backup_'+filename, index=False)

                                if np.random.random() <= randratio:
                                  if s*randratio > len(perf_df):
                                    loc_df = pd.DataFrame()
                                    loc_df.loc[0, 'szorz'] = szorz
                                    loc_df.loc[0, 'textcol'] = textcol
                                    loc_df.loc[0, 'mindf'] = mindf
                                    loc_df.loc[0, 'maxdf'] = maxdf
                                    loc_df.loc[0, 'ngram_range'] = str(ngramrange)
                                    loc_df.loc[0, 'max_features'] = maxfeat
                                    loc_df.loc[0, 'token_pattern'] = tokenp
                                    loc_df.loc[0, 'subsample'] = subsam
                                    loc_df.loc[0, 'colsample_bytree'] = colsamt
                                    loc_df.loc[0, 'colsample_bynode'] = colsamn
                                    loc_df.loc[0, 'max_depth'] = depth
                                    loc_df.loc[0, 'reg_alpha'] = alph
                                    loc_df.loc[0, 'n_estimators'] = estims

                                    test_acc_l = []
                                    train_acc_l = []
                                    test_f1_l = []
                                    train_f1_l = []

                                    for i in range(set(fold_col)):
                                      loc_test_index = train_DF[fold_col] == i
                                      loc_train_DF = train_DF[~loc_test_index]
                                      loc_test_DF = train_DF[loc_test_index]
                                      loc_train_target = train_target[~loc_test_index]
                                      loc_test_target = train_target[loc_test_index]

                                      vocab = combined_vocab_maker(df = loc_train_DF,
                                                                  textcol = textcol,
                                                                  classcol = classcol_n,
                                                                  maxfeatures = maxfeat,
                                                                  mindf = mindf,
                                                                  maxdf = maxdf,
                                                                  ngramrange = ngramrange,
                                                                  tokenpattern = tokenp,
                                                                  szorz = szorz,
                                                                  plusboth = True)
                                      
                                      vect_xgb = TfidfVectorizer(vocabulary=vocab,
                                                                token_pattern=tokenp,
                                                                use_idf=True,
                                                                smooth_idf=True,
                                                                sublinear_tf=True)
                                      xgb_X=vect_xgb.fit_transform(loc_train_DF[textcol])

                                      xgb_en = xgb.XGBClassifier(subsample= subsam,
                                                                colsample_bytree= colsamt,
                                                                colsample_bynode= colsamn,
                                                                reg_alpha= alph,
                                                                max_depth= depth,
                                                                n_estimators= estims
                                                                )
                                      xgb_en.fit(xgb_X,loc_train_target)

                                      train_preds = xgb_en.predict(xgb_X)
                                      train_acc_l.append(accuracy_score(loc_train_target, train_preds))
                                      train_f1_l.append(f1_score(loc_train_target, train_preds))

                                      xgb_X = vect_xgb.transform(loc_test_DF[textcol])
                                      test_preds = xgb_en.predict(xgb_X)
                                      test_acc_l.append(accuracy_score(loc_test_target, test_preds))
                                      test_f1_l.append(f1_score(loc_test_target, test_preds))

                                    loc_df.loc[0, 'mean_test_acc'] = np.mean(test_acc_l)
                                    loc_df.loc[0, 'std_test_acc'] = np.std(test_acc_l)
                                    loc_df.loc[0, 'mean_test_f1'] = np.mean(test_f1_l)
                                    loc_df.loc[0, 'std_test_f1'] = np.std(test_f1_l)
                                    loc_df.loc[0, 'test_accs'] = str(test_acc_l)
                                    loc_df.loc[0, 'test_f1s'] = str(test_f1_l)
                                    loc_df.loc[0, 'min_test_acc'] = min(test_acc_l)
                                    loc_df.loc[0, 'min_test_f1'] = min(test_f1_l)
                                    loc_df.loc[0, 'mean_train_acc'] = np.mean(train_acc_l)
                                    loc_df.loc[0, 'std_train_acc'] = np.std(train_acc_l)
                                    loc_df.loc[0, 'mean_train_f1'] = np.mean(train_f1_l)
                                    loc_df.loc[0, 'std_train_f1'] = np.std(train_f1_l)
                                    loc_df.loc[0, 'train_accs'] = str(train_acc_l)
                                    loc_df.loc[0, 'trainf1s'] = str(train_f1_l)

                                    perf_df = perf_df.append(loc_df, ignore_index = True)

                  if save:
                    perf_df.to_excel(loc+'/'+filename, index=False)
                  elaps = time.time()-startt
                  print('-'*25, 'done', len(perf_df), time.ctime(), 'elapsed:', elaps, '|s/db:', elaps/len(perf_df))

    if return_last:
      return perf_df

######################################################
###                                                ###
###        CV search for best hyperpars            ###
###                                                ###
######################################################

CV_combinedvocab(train_DF = train_DF,
                  train_target = train_target,
                  loc = 'analysis',
                 szorz_l = [1, 1.2, 1.5],
                  textcol_l = ['response_cleaned_lower', 'response_cleaned_withoutstop', 'response_cleaned_withoutstop_lemma'],
                  mindf_l = [0.005],
                  maxdf_l = [0.99],
                  ngramrange_l = [(1,2), (1,3)],
                  maxfeat_l = [100, 150, 250, 400],
                  file_n = "separ_ngram_cv_",
                  tokenp_l = [r'(?u)\b\w\w+\b', r'(?u)\b\w+\b', r'(?u)\b\w+\b|[^\w\s]'],
                  subsam_l = [0.8, 0.9],
                  colsamt_l = [0.8, 0.9, 1],
                colsamn_l = [0.8, 0.9, 1],
                depth_l = [3, 5, 6, 9],
                alph_l = [0.1, 0.15, 0.3],
                estims_l = [100, 50],
                classcol_n = 'roma',
                fold_col = 'CV_all_4f',
                 onlycount = False
                  )

#################
###
###  concatenating output perf_df-s
###

files = os.listdir('analysis')
#selecting only final output files of CV_combinedvocab run
files = [f for f in files if (('separ_ngram' in f)& ('backup' not in f))]
print(len(files))

#concat DFs
for sorsz, fl in enumerate(files):
  act_df = pd.read_excel('analysis/'+fl)
  
  if sorsz == 0:
    CV_DF = act_df.copy()
  else:
    CV_DF = pd.concat([CV_DF, act_df], ignore_index=True)

  if sorsz%25 == 0:
    print(sorsz)

CV_DF.sort_values(by='mean_test_acc', ascending=False, inplace=True)
CV_DF.to_excel('analysis/separ_ngram_cv_full.xlsx', index=False)

######################################################
###                                                ###
###           training final model                 ###
###                                                ###
######################################################
#setting hyperpars
textcol = CV_DF.textcol.iloc[0]
szorz = float(CV_DF.szorz.iloc[0])
mindf = float(CV_DF.mindf.iloc[0])
maxdf = float(CV_DF.maxdf.iloc[0])
ngramrange = tuple([int (s) for s in CV_DF.ngram_range.iloc[0][1:-1].split(', ')])
maxfeat = int(CV_DF.max_features.iloc[0])
tokenp = CV_DF.token_pattern.iloc[0]
subsam = float(CV_DF.subsample.iloc[0])
colsamt = float(CV_DF.colsample_bytree.iloc[0])
colsamn = float(CV_DF.colsample_bynode.iloc[0])
depth = int(CV_DF.max_depth.iloc[0])
alph = float(CV_DF.reg_alpha.iloc[0])
estims = int(CV_DF.n_estimators.iloc[0])

#creating vocab
vocab, vocabs = combined_vocab_maker(df = train_DF,
                                     textcol = textcol,
                                     classcol = 'roma',
                                     maxfeatures = maxfeat,
                                     szorz = szorz,
                                     mindf = mindf,
                                     maxdf = maxdf,
                                     ngramrange = ngramrange,
                                     tokenpattern = tokenp,
                                     plusboth = True,
                                     return_separate_vocabs = True)
#vectorize corpus
vect_xgb = TfidfVectorizer(vocabulary=vocab,
                          token_pattern=tokenp,
                          use_idf=True,
                          smooth_idf=True,
                          sublinear_tf=True)
xgb_X_train=vect_xgb.fit_transform(train_DF[textcol])

#training model
xgb_en = xgb.XGBClassifier(subsample= subsam,
                          colsample_bytree= colsamt,
                          colsample_bynode= colsamn,
                          reg_alpha= alph,
                          max_depth= depth,
                          n_estimators= estims
                          )
xgb_en.fit(xgb_X_train,train_target)

#train set performance
train_preds = xgb_en.predict(xgb_X_train)
print('train_acc', accuracy_score(train_target, train_preds))
print('train_f1', f1_score(train_target, train_preds))

#test set performance
xgb_X_test = vect_xgb.transform(test_DF[textcol])
test_preds = xgb_en.predict(xgb_X_test)
print('test_acc', accuracy_score(test_target, test_preds))
print('test_f1', f1_score(test_target, test_preds))
print(classification_report(test_target, test_preds))

# save models
pickle.dump(vect_xgb, open('models/ngram_vect.pickle', 'wb'))
joblib.dump(xgb_en, "models/ngram_model")

######################################################
###                                                ###
###           feature importance                   ###
###                                                ###
######################################################
#constructing label dictionary for feature numbers
label_dict = {v:k for k, v in vect_xgb.vocabulary_.items()}
imps = ['weight', 'gain', 'total_gain', 'cover', 'total_cover']
vals = {}
vals_n = {}
labels = []
for i in imps:
  imp = xgb_en.get_booster().get_score(importance_type= i)
  if len(labels) == 0:
    labels = list(imp.keys())
    labels_ngrams = [label_dict[int(l[1:])] for l in labels]
  vals[i] = list(imp.values())
  vals_n[i] = np.divide(list(imp.values()),sum(imp.values()))

#select features in top 20 percent by gain or cover
perc = 80
indexek = np.logical_or(vals_n[imps[1]] > np.percentile(vals_n[imps[1]],perc),
                        vals_n[imps[3]] > np.percentile(vals_n[imps[3]],perc))

#plot noramlized feature importances
x = np.arange(len(np.array(labels_ngrams)[indexek]))
width = 0.15
mins = [-2,-1,0,1,2]

#creating feature importance DF with cols feature name & importances for selected features
imp_df = pd.DataFrame({'feature': np.array(labels_ngrams)[indexek],
                       imps[0]: vals_n[imps[0]][indexek],
                       imps[1]: vals_n[imps[1]][indexek],
                       imps[2]: vals_n[imps[2]][indexek],
                       imps[3]: vals_n[imps[3]][indexek],
                       imps[4]: vals_n[imps[4]][indexek]})

#sorting by combined gain & cover
imp_df['sum'] = imp_df[imps[1]]+imp_df[imps[3]]
imp_df.sort_values(by='sum', ascending=False, inplace=True)

#setting plot size
fig, ax = plt.subplots(figsize=(20, 5))

#plotting bars for each importance value
rects0 = ax.bar(x + width*mins[0], imp_df[imps[0]], width, label=imps[0])
rects1 = ax.bar(x + width*mins[1], imp_df[imps[1]], width, label=imps[1])
rects2 = ax.bar(x + width*mins[2], imp_df[imps[2]], width, label=imps[2])
rects3 = ax.bar(x + width*mins[3], imp_df[imps[3]], width, label=imps[3])
rects4 = ax.bar(x + width*mins[4], imp_df[imps[4]], width, label=imps[4])

ax.set_ylabel('normalised scores')
ax.set_title('feature importances of features in top ' + str(100-perc) + ' percentile by cover and gain (' + str(sum(indexek))+ ' feature)')
ax.set_xticks(x)
ax.set_xticklabels(imp_df.feature, rotation = 45, ha = 'right', fontsize=14)
ax.legend()

fig.tight_layout()
plt.show()

imp_df.drop(['sum'], axis = 1, inplace=True)

#inserting columns for each feature with source vocabulary
for i in range(len(imp_df)):
  szo = imp_df.feature[i]
  # lis = []
  for tip, voc in vocabs.items():
    if szo in voc:
      if tip == 0:
        imp_df.loc[i, 'notroma_vocab'] = 1
      elif tip == 1:
        imp_df.loc[i, 'roma_vocab'] = 1
      elif tip == 'all':
        imp_df.loc[i, 'both_vocab'] = 1
      else:
        print(szo, tip)

#saving to excel
imp_df.to_excel('analysis/ngram_feature_imp.xlsx', index = False)

#plot only noramlized gain & cover
width = 0.3
mins = [-0.5,0.5]

fig, ax = plt.subplots(figsize=(20, 5))

imp_df = pd.DataFrame({'feature': np.array(labels_ngrams)[indexek],
                       imps[1]: vals_n[imps[1]][indexek],
                       imps[3]: vals_n[imps[3]][indexek]})
imp_df['sum'] = imp_df[imps[1]]+imp_df[imps[3]]
imp_df.sort_values(by='sum', ascending=False, inplace=True)

rects1 = ax.bar(x + width*mins[0], imp_df[imps[1]], width, label=imps[1])
rects1 = ax.bar(x + width*mins[1], imp_df[imps[3]], width, label=imps[3])

ax.set_ylabel('normalised scores')
ax.set_title('feature importances of features in top ' + str(100-perc) + ' percentile by cover and gain (' + str(sum(indexek))+ ' feature)')
ax.set_xticks(x)
ax.set_xticklabels(imp_df.feature, rotation = 45, ha = 'right', fontsize = 14)
ax.legend()

fig.tight_layout()
plt.show()

#plotting feature importance plots/value type
num = 40 #for top 40 fatures/value
for tipus in imps:
  imp = xgb_en.get_booster().get_score(importance_type= tipus)
  mapped = {label_dict[int(k[1:])]: v for k, v in imp.items()}
  fig, ax = plt.subplots(figsize=(10,15))
  xgb.plot_importance(mapped, max_num_features=num, height=0.5, ax=ax,importance_type=tipus, title='feature importance: '+tipus)
  plt.show()

#plotting top shap values
num = 48
shap_values = shap.TreeExplainer(xgb_en).shap_values(xgb_X_test)
shap.summary_plot(shap_values, xgb_X_test.A, plot_type="bar", feature_names=list(vect_xgb.vocabulary_), max_display=num)
print(shap_values.shape)
szot = {}
for ix, v in enumerate(np.mean(abs(shap_values), axis = 0)):
  szot[v] = [label_dict[ix], ix]
db = 0
for key in sorted(szot.keys(), reverse=True):
  db += 1
  print(szot[key][0], key, szot[key][1])

#feature effect direction
shap.summary_plot(shap_values, xgb_X_test.A, feature_names=list(vect_xgb.vocabulary_), max_display=30)

######################################################
###                                                ###
###        model performance by gender             ###
###                                                ###
######################################################

rows = test_DF.female == 1
print('acc:', accuracy_score(test_DF['roma'][rows], test_preds[rows]))
print('f1:', f1_score(test_DF['roma'][rows], test_preds[rows]))
print(classification_report(test_DF['roma'][rows], test_preds[rows]))

rows = test_DF.female == 0
print('acc:', accuracy_score(test_DF['roma'][rows], test_preds[rows]))
print('f1:', f1_score(test_DF['roma'][rows], test_preds[rows]))
print(classification_report(test_DF['roma'][rows], test_preds[rows]))

########################################
###
###   train separate models by gender
###

###
### female
###

train_rows_fem = train_DF.female == 1
test_rows_fem = test_DF.female == 1
print('train size:', sum(train_rows_fem), 'test size:', sum(test_rows_fem))

textcol = CV_DF.textcol.iloc[0]
szorz = float(CV_DF.szorz.iloc[0])
mindf = float(CV_DF.mindf.iloc[0])
maxdf = float(CV_DF.maxdf.iloc[0])
ngramrange = tuple([int (s) for s in CV_DF.ngram_range.iloc[0][1:-1].split(', ')])
maxfeat = int(CV_DF.max_features.iloc[0])
tokenp = CV_DF.token_pattern.iloc[0]
subsam = float(CV_DF.subsample.iloc[0])
colsamt = float(CV_DF.colsample_bytree.iloc[0])
colsamn = float(CV_DF.colsample_bynode.iloc[0])
depth = int(CV_DF.max_depth.iloc[0])
alph = float(CV_DF.reg_alpha.iloc[0])
estims = int(CV_DF.n_estimators.iloc[0])


vocab_fem = combined_vocab_maker(df = train_DF[train_rows_fem],
                            textcol = textcol,
                            classcol = 'roma',
                            maxfeatures = maxfeat,
                             szorz = szorz,
                            mindf = mindf,
                            maxdf = maxdf,
                            ngramrange = ngramrange,
                            tokenpattern = tokenp,
                            plusboth = True)

vect_xgb_fem = TfidfVectorizer(vocabulary=vocab_fem,
                          token_pattern=tokenp,
                          use_idf=True,
                          smooth_idf=True,
                          sublinear_tf=True)
xgb_X=vect_xgb_fem.fit_transform(train_DF[train_rows_fem][textcol])

xgb_en_fem = xgb.XGBClassifier(subsample= subsam,
                          colsample_bytree= colsamt,
                          colsample_bynode= colsamn,
                          reg_alpha= alph,
                          max_depth= depth,
                          n_estimators= estims
                          )
xgb_en_fem.fit(xgb_X,train_target[train_rows_fem])

train_preds_fem = xgb_en_fem.predict(xgb_X)
print('train_acc', accuracy_score(train_target[train_rows_fem], train_preds_fem))
print('train_f1', f1_score(train_target[train_rows_fem], train_preds_fem))
print()
xgb_X = vect_xgb_fem.transform(test_DF[test_rows_fem][textcol])
test_preds_fem = xgb_en_fem.predict(xgb_X)
print('test_acc', accuracy_score(test_target[test_rows_fem], test_preds_fem))
print('test_f1', f1_score(test_target[test_rows_fem], test_preds_fem))
print(classification_report(test_target[test_rows_fem], test_preds_fem))

###
###   male
###

train_rows_male = train_DF.female == 0
test_rows_male = test_DF.female == 0
print('train size:', sum(train_rows_male), 'test size:', sum(test_rows_male))

textcol = CV_DF.textcol.iloc[0]
szorz = float(CV_DF.szorz.iloc[0])
mindf = float(CV_DF.mindf.iloc[0])
maxdf = float(CV_DF.maxdf.iloc[0])
ngramrange = tuple([int (s) for s in CV_DF.ngram_range.iloc[0][1:-1].split(', ')])
maxfeat = int(CV_DF.max_features.iloc[0])
tokenp = CV_DF.token_pattern.iloc[0]
subsam = float(CV_DF.subsample.iloc[0])
colsamt = float(CV_DF.colsample_bytree.iloc[0])
colsamn = float(CV_DF.colsample_bynode.iloc[0])
depth = int(CV_DF.max_depth.iloc[0])
alph = float(CV_DF.reg_alpha.iloc[0])
estims = int(CV_DF.n_estimators.iloc[0])


vocab_male = combined_vocab_maker(df = train_DF[train_rows_male],
                            textcol = textcol,
                            classcol = 'roma',
                            maxfeatures = maxfeat,
                             szorz = szorz,
                            mindf = mindf,
                            maxdf = maxdf,
                            ngramrange = ngramrange,
                            tokenpattern = tokenp,
                            plusboth = True)

vect_xgb_male = TfidfVectorizer(vocabulary=vocab_male,
                          token_pattern=tokenp,
                          use_idf=True,
                          smooth_idf=True,
                          sublinear_tf=True)
xgb_X=vect_xgb_male.fit_transform(train_DF[train_rows_male][textcol])

xgb_en_male = xgb.XGBClassifier(subsample= subsam,
                          colsample_bytree= colsamt,
                          colsample_bynode= colsamn,
                          reg_alpha= alph,
                          max_depth= depth,
                          n_estimators= estims
                          )
xgb_en_male.fit(xgb_X,train_target[train_rows_male])

train_preds_male = xgb_en_male.predict(xgb_X)
print('train_acc', accuracy_score(train_target[train_rows_male], train_preds_male))
print('train_f1', f1_score(train_target[train_rows_male], train_preds_male))
print()
xgb_X = vect_xgb_male.transform(test_DF[test_rows_male][textcol])
test_preds_male = xgb_en_male.predict(xgb_X)
print('test_acc', accuracy_score(test_target[test_rows_male], test_preds_male))
print('test_f1', f1_score(test_target[test_rows_male], test_preds_male))
print(classification_report(test_target[test_rows_male], test_preds_male))