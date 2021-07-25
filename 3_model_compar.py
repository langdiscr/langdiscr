import joblib

from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

import xgboost as xgb

def print_head(szov, h = 50, fo = False):
  """
  prints out header for easier overview of outputs
  szov: text to print
  h: width of header
  fo: bool - main header or nor
  """
  h_sp = (h-10-len(szov))/2
  if h_sp != int(h_sp):
    h_sp1 = int(h_sp-0.5)
    h_sp2 = int(h_sp+0.5)
  else:
    h_sp1 = h_sp2 = int(h_sp)
  if fo:
    print()
  print('#' * h)
  if fo:
    print('#####', ' '*(h-12),'#####')
  print('###',  ' '*h_sp1, szov, ' '*h_sp2, '###')
  if fo:
    print('#####', ' '*(h-12),'#####')
  print('#' * h)

######################################################
###                                                ###
###                importing data                  ###
###                                                ###
######################################################
RAW = pd.read_excel('data/processed/cleaned_v5_withfolds.xlsx', index_col=0)
RAW = RAW[RAW.no_real_resp == 0]
print(RAW.shape)
RAW.head()

#DF with variables of interest
hatter_DF = RAW[['town', 'pop', 'wave', 'treated', 'roma', 'request', 'ethnicity',
                 'gender', 'treated_roma', 'response_cleaned', 'test_fold']]
print(hatter_DF.shape)
hatter_DF.head()

stat_DF = pd.read_excel('data/processed/stat_model_df_filtered_withfold.xlsx', index_col=0)
print(stat_DF.shape)
stat_DF.head()

ngram_DF = pd.read_excel('data/processed/ngram_df_filtered.xlsx', index_col=0)
print(ngram_DF.shape)
ngram_DF.head()

#train-test split
ngram_train_DF = ngram_DF[ngram_DF.test_fold != 2]
ngram_test_DF = ngram_DF[ngram_DF.test_fold == 2]

ngram_train_target = ngram_train_DF['roma']
ngram_test_target = ngram_test_DF['roma']

stat_train_DF = stat_DF[stat_DF.test_fold != 2]
stat_test_DF = stat_DF[stat_DF.test_fold == 2]

stat_train_target = stat_train_DF['roma']
stat_test_target = stat_test_DF['roma']

hatter_train_DF = hatter_DF[hatter_DF.test_fold != 2]
hatter_test_DF = hatter_DF[hatter_DF.test_fold == 2]

hatter_train_target = hatter_train_DF['roma']
hatter_test_target = hatter_test_DF['roma']

#trained models
mdl9 = joblib.load("models/stat_model_9vars")
vect_xgb = pickle.load(open('models/ngram_vect.pickle', 'rb'))
xgb_en = joblib.load("models/ngram_model")

#stat model performance
varcols = ['num_words',
           'var_punct',
           'ratio_szám_token',
           'ratio_punct',
           'ratio_VERB',
           'ratio_PROPN',
           'ratio_PRON',
           'ratio_DET',
           'ratio_ADP']

stat_preds = mdl9.predict(stat_test_DF[varcols])
stat_preds_proba = mdl9.predict_proba(stat_test_DF[varcols])
print('acc:', accuracy_score(stat_test_DF['roma'], stat_preds))
print('f1:', f1_score(stat_test_DF['roma'], stat_preds))
print(classification_report(stat_test_DF['roma'], stat_preds))

#n-gram model performance
textcol = 'response_cleaned_withoutstop'
xgb_X_test = vect_xgb.transform(ngram_test_DF[textcol])
ngram_preds = xgb_en.predict(xgb_X_test)
ngram_preds_proba = xgb_en.predict_proba(xgb_X_test)
print('test_acc', accuracy_score(ngram_test_target, ngram_preds))
print('test_f1', f1_score(ngram_test_target, ngram_preds))
print(classification_report(ngram_test_target, ngram_preds))

#ratio of agreement between the 2 models
np.mean(stat_preds == ngram_preds)

#test prediction to DF
hatter_test_DF['ngram_preds'] = ngram_preds
hatter_test_DF['ngram_preds_proba'] = ngram_preds_proba[:,1]
hatter_test_DF['stat_preds'] = stat_preds
hatter_test_DF['stat_preds_proba'] = stat_preds_proba[:,1]

######################################################
###                                                ###
###        agreements between the 2 models         ###
###                                                ###
######################################################
#both pred is roma
hatter_test_DF_2r = hatter_test_DF[(hatter_test_DF.ngram_preds == 1) & (hatter_test_DF.stat_preds == 1)]
print('number of rows:', hatter_test_DF_2r.shape[0])
print('ratio of correct predictions:', np.mean(hatter_test_DF_2r.roma))

#both pred is not roma
hatter_test_DF_2nr = hatter_test_DF[(hatter_test_DF.ngram_preds == 0) & (hatter_test_DF.stat_preds == 0)]
print('number of rows:', hatter_test_DF_2nr.shape[0])
print('ratio of correct predictions:', np.mean(hatter_test_DF_2nr.roma == 0))

#ngram roma, stat not roma
hatter_test_DF_ngrstnr = hatter_test_DF[(hatter_test_DF.ngram_preds == 1) & (hatter_test_DF.stat_preds == 0)]
print('number of rows:', hatter_test_DF_ngrstnr.shape[0])
print('ratio of romas:', np.mean(hatter_test_DF_ngrstnr.roma))

#ngram not roma, stat roma
hatter_test_DF_ngnrstr = hatter_test_DF[(hatter_test_DF.ngram_preds == 0) & (hatter_test_DF.stat_preds == 1)]
print('number of rows:', hatter_test_DF_ngnrstr.shape[0])
print('ratio of romas:', np.mean(hatter_test_DF_ngnrstr.roma))

# descriptive stats of test set
print('átlag pop\t\t\t',
      np.mean(hatter_test_DF['pop']))
print('alsó 10ed pop\t\t', sum(hatter_test_DF['pop'] < np.percentile(hatter_DF['pop'], 10)), '\t',
      np.mean(hatter_test_DF['pop'] < np.percentile(hatter_DF['pop'], 10)))
print('felső 10ed pop\t\t', sum(hatter_test_DF['pop'] > np.percentile(hatter_DF['pop'], 90)), '\t',
      np.mean(hatter_test_DF['pop'] > np.percentile(hatter_DF['pop'], 90)))
print('nők aránya\t\t', sum(hatter_test_DF['gender'] == 'female'), '\t',
      np.mean(hatter_test_DF['gender'] == 'female'))

#descriptive stats of those who were classified as not roma by both model incorrectly
teves2roma_DF = hatter_test_DF[(hatter_test_DF.ngram_preds == 0) &
                              (hatter_test_DF.stat_preds == 0) &
                              (hatter_test_DF.roma == 1)]
print('number of rows:', len(teves2roma_DF))
print('átlag pop\t\t\t',
      np.mean(teves2roma_DF['pop']))
print('alsó 10ed pop\t\t', sum(teves2roma_DF['pop'] < np.percentile(hatter_DF['pop'], 10)), '\t',
      np.mean(teves2roma_DF['pop'] < np.percentile(hatter_DF['pop'], 10)))
print('felső 10ed pop\t\t', sum(teves2roma_DF['pop'] > np.percentile(hatter_DF['pop'], 90)), '\t',
      np.mean(teves2roma_DF['pop'] > np.percentile(hatter_DF['pop'], 90)))
print('nők aránya\t\t', sum(teves2roma_DF['gender'] == 'female'), '\t',
      np.mean(teves2roma_DF['gender'] == 'female'))

#descriptive stats of those who were classified as roma by both model incorrectly
teves2nroma_DF = hatter_test_DF[(hatter_test_DF.ngram_preds == 1) &
                              (hatter_test_DF.stat_preds == 1) &
                              (hatter_test_DF.roma == 0)]
print('number of rows:', len(teves2nroma_DF))

print('átlag pop\t\t\t',
      np.mean(teves2nroma_DF['pop']))
print('alsó 10ed pop\t\t', sum(teves2nroma_DF['pop'] < np.percentile(hatter_DF['pop'], 10)), '\t',
      np.mean(teves2nroma_DF['pop'] < np.percentile(hatter_DF['pop'], 10)))
print('felső 10ed pop\t\t', sum(teves2nroma_DF['pop'] > np.percentile(hatter_DF['pop'], 90)), '\t',
      np.mean(teves2nroma_DF['pop'] > np.percentile(hatter_DF['pop'], 90)))
print('nők aránya\t\t', sum(teves2nroma_DF['gender'] == 'female'), '\t',
      np.mean(teves2nroma_DF['gender'] == 'female'))

#descriptive stats of those who were classified as not roma by both model correctly
jo2nroma_DF = hatter_test_DF[(hatter_test_DF.ngram_preds == 0) &
                              (hatter_test_DF.stat_preds == 0) &
                              (hatter_test_DF.roma == 0)]
print('number of rows:', len(jo2nroma_DF))
print('átlag pop\t\t\t',
      np.mean(jo2nroma_DF['pop']))
print('alsó 10ed pop\t\t', sum(jo2nroma_DF['pop'] < np.percentile(hatter_DF['pop'], 10)), '\t',
      np.mean(jo2nroma_DF['pop'] < np.percentile(hatter_DF['pop'], 10)))
print('felső 10ed pop\t\t', sum(jo2nroma_DF['pop'] > np.percentile(hatter_DF['pop'], 90)), '\t',
      np.mean(jo2nroma_DF['pop'] > np.percentile(hatter_DF['pop'], 90)))
print('nők aránya\t\t', sum(jo2nroma_DF['gender'] == 'female'), '\t',
      np.mean(jo2nroma_DF['gender'] == 'female'))

#descriptive stats of those who were classified as roma by both model correctly
jo2roma_DF = hatter_test_DF[(hatter_test_DF.ngram_preds == 1) &
                              (hatter_test_DF.stat_preds == 1) &
                              (hatter_test_DF.roma == 1)]
print('number of rows:', len(jo2roma_DF))
print('átlag pop\t\t\t',
      np.mean(jo2roma_DF['pop']))
print('alsó 10ed pop\t\t', sum(jo2roma_DF['pop'] < np.percentile(hatter_DF['pop'], 10)), '\t',
      np.mean(jo2roma_DF['pop'] < np.percentile(hatter_DF['pop'], 10)))
print('felső 10ed pop\t\t', sum(jo2roma_DF['pop'] > np.percentile(hatter_DF['pop'], 90)), '\t',
      np.mean(jo2roma_DF['pop'] > np.percentile(hatter_DF['pop'], 90)))
print('nők aránya\t\t', sum(jo2roma_DF['gender'] == 'female'), '\t',
      np.mean(jo2roma_DF['gender'] == 'female'))

hatter_test_DF.to_excel('analysis/test_DF_withpreds.xlsx', index=False)

"""### településnagyság"""

######################################################
###                                                ###
###        model performance by population         ###
###                                                ###
######################################################

#binary pop size
hatter_test_DF['pop_bin'] = pd.qcut(hatter_test_DF['pop'], 2, [0,1])
#three class pop size
hatter_test_DF['pop_tri'] = pd.qcut(hatter_test_DF['pop'], 3, [0,1,2])
#absolute errors of model predictions (of probabilities)
hatter_test_DF['stat_abs_err'] = abs(hatter_test_DF['roma'] - hatter_test_DF['stat_preds_proba'])
hatter_test_DF['ngram_abs_err'] = abs(hatter_test_DF['roma'] - hatter_test_DF['ngram_preds_proba'])

#performaces of models by binary pop size on whole test set
print_head('kis teleps', 60, True)
rows = (hatter_test_DF.pop_bin == 0)
print_head('stat')
preds_col = "stat_preds"
print('acc:', accuracy_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print('f1:', f1_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print(classification_report(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print_head('ngram')
preds_col = "ngram_preds"
print('acc:', accuracy_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print('f1:', f1_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print(classification_report(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print_head('nagy teleps', 60, True)
rows = (hatter_test_DF.pop_bin == 1)
print_head('stat')
preds_col = "stat_preds"
print('acc:', accuracy_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print('f1:', f1_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print(classification_report(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print_head('ngram')
preds_col = "ngram_preds"
print('acc:', accuracy_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print('f1:', f1_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print(classification_report(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))

#performaces of models by binary pop size on males
print_head('kis teleps ffi', 60, True)
rows = (hatter_test_DF.pop_bin == 0) & (hatter_test_DF.gender == 'male')
print_head('stat')
preds_col = "stat_preds"
print('acc:', accuracy_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print('f1:', f1_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print(classification_report(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print_head('ngram')
preds_col = "ngram_preds"
print('acc:', accuracy_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print('f1:', f1_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print(classification_report(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print_head('nagy teleps ffi', 60, True)
rows = (hatter_test_DF.pop_bin == 1) & (hatter_test_DF.gender == 'male')
print_head('stat')
preds_col = "stat_preds"
print('acc:', accuracy_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print('f1:', f1_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print(classification_report(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print_head('ngram')
preds_col = "ngram_preds"
print('acc:', accuracy_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print('f1:', f1_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print(classification_report(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))

#performaces of models by binary pop size on females
print_head('kis teleps no', 60, True)
rows = (hatter_test_DF.pop_bin == 0) & (hatter_test_DF.gender == 'female')
print_head('stat')
preds_col = "stat_preds"
print('acc:', accuracy_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print('f1:', f1_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print(classification_report(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print_head('ngram')
preds_col = "ngram_preds"
print('acc:', accuracy_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print('f1:', f1_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print(classification_report(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print_head('nagy teleps no', 60, True)
rows = (hatter_test_DF.pop_bin == 1) & (hatter_test_DF.gender == 'female')
print_head('stat')
preds_col = "stat_preds"
print('acc:', accuracy_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print('f1:', f1_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print(classification_report(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print_head('ngram')
preds_col = "ngram_preds"
print('acc:', accuracy_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print('f1:', f1_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print(classification_report(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))

#performaces of models by three pop size class on whole test set
print_head('kis teleps', 60, True)
rows = (hatter_test_DF.pop_tri == 0)
print_head('stat')
preds_col = "stat_preds"
print('acc:', accuracy_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print('f1:', f1_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print(classification_report(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print_head('ngram')
preds_col = "ngram_preds"
print('acc:', accuracy_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print('f1:', f1_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print(classification_report(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))

print_head('kp teleps', 60, True)
rows = (hatter_test_DF.pop_tri == 1)
print_head('stat')
preds_col = "stat_preds"
print('acc:', accuracy_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print('f1:', f1_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print(classification_report(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print_head('ngram')
preds_col = "ngram_preds"
print('acc:', accuracy_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print('f1:', f1_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print(classification_report(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))

print_head('nagy teleps', 60, True)
rows = (hatter_test_DF.pop_tri == 2)
print_head('stat')
preds_col = "stat_preds"
print('acc:', accuracy_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print('f1:', f1_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print(classification_report(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print_head('ngram')
preds_col = "ngram_preds"
print('acc:', accuracy_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print('f1:', f1_score(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))
print(classification_report(hatter_test_DF['roma'][rows], hatter_test_DF[preds_col][rows]))

#plot n-gram model absolute error and pop size by ethinicty and gender
fig, ax = plt.subplots(figsize=(10, 6))
collist = ['g', 'b']
marker_dict = {'male': '+', 'female': 'o'}
for rm in [0,1]:
  for gdr in ['male', 'female']:
    rows = (hatter_test_DF['gender'] == gdr) & (hatter_test_DF['roma'] == rm)
    ax.scatter(hatter_test_DF['pop'][rows].apply(np.log),
                hatter_test_DF['ngram_abs_err'][rows],
                # c=hatter_test_DF['roma'][rows],
                c = collist[rm],
                marker = marker_dict[gdr],
                s = 100)
# ax.legend(loc="upper left", title="roma")
green_patch = mpatches.Patch(color='green', label='not roma')
blue_patch = mpatches.Patch(color='blue', label='roma')
l1 = ax.legend(handles=[green_patch, blue_patch], loc="lower left", title="roma")
ax.add_artist(l1)


legend_elements = [Line2D([], [], marker='o', color='k', label='female',
                          markerfacecolor='k', markersize=10, linestyle = 'None'),
                   Line2D([], [], marker='+', color='k', label='male',
                          markerfacecolor='k', markersize=10, linestyle = 'None')]
l2 = ax.legend(handles=legend_elements, loc="lower right", title="gender")
ax.add_artist(l2)

plt.suptitle('log of pop and ngram model abs error', fontsize = 'xx-large')
ax.set_ylabel('absolute error of ngram model')
ax.set_xlabel('log of population')

plt.plot( range(4,13),[0.5]*9, '--', c = 'r', linewidth=0.5)
plt.plot( [np.log(np.median(hatter_test_DF['pop']))]*2,[0.2,0.8], '--', c = 'c', linewidth=0.5)
# ax.add_artist(legend1)
plt.show()

#plot stat model absolute error and pop size by ethinicty and gender
fig, ax = plt.subplots(figsize=(10, 6))
collist = ['g', 'b']
marker_dict = {'male': '+', 'female': 'o'}
for rm in [0,1]:
  for gdr in ['male', 'female']:
    rows = (hatter_test_DF['gender'] == gdr) & (hatter_test_DF['roma'] == rm)
    ax.scatter(hatter_test_DF['pop'][rows].apply(np.log),
                hatter_test_DF['stat_abs_err'][rows],
                # c=hatter_test_DF['roma'][rows],
                c = collist[rm],
                marker = marker_dict[gdr],
                s = 100)
# ax.legend(loc="upper left", title="roma")
green_patch = mpatches.Patch(color='green', label='not roma')
blue_patch = mpatches.Patch(color='blue', label='roma')
l1 = ax.legend(handles=[green_patch, blue_patch], loc="lower left", title="roma")
ax.add_artist(l1)


legend_elements = [Line2D([], [], marker='o', color='k', label='female',
                          markerfacecolor='k', markersize=10, linestyle = 'None'),
                   Line2D([], [], marker='+', color='k', label='male',
                          markerfacecolor='k', markersize=10, linestyle = 'None')]
l2 = ax.legend(handles=legend_elements, loc="lower right", title="gender")
ax.add_artist(l2)

plt.suptitle('log of pop and stat model abs error', fontsize = 'xx-large')
ax.set_ylabel('absolute error of stat model')
ax.set_xlabel('log of population')
ax.plot( range(4,13),[0.5]*9, '--', c = 'r', linewidth=0.5)
plt.plot( [np.log(np.median(hatter_test_DF['pop']))]*2,[0.2,0.8], '--', c = 'c', linewidth=0.5)
# ax.add_artist(legend1)
plt.show()

#correlation of absolute errors & pop size on whole test set
print('pop - stat abs err corr:', '\t', hatter_test_DF['pop'].corr(hatter_test_DF['stat_abs_err']))
print('pop - ngram abs err corr:', '\t', hatter_test_DF['pop'].corr(hatter_test_DF['ngram_abs_err']))
print()

#correlation of absolute errors & pop size by gender
rows = hatter_test_DF['gender'] == 'male'
print('male: pop - stat abs err corr:', '\t', hatter_test_DF['pop'][rows].corr(hatter_test_DF['stat_abs_err'][rows]))
print('male: pop - ngram abs err corr:', '\t', hatter_test_DF['pop'][rows].corr(hatter_test_DF['ngram_abs_err'][rows]))
print('')
rows = hatter_test_DF['gender'] == 'female'
print('female: pop - stat abs err corr:', '\t', hatter_test_DF['pop'][rows].corr(hatter_test_DF['stat_abs_err'][rows]))
print('female: pop - ngram abs err corr:', '\t', hatter_test_DF['pop'][rows].corr(hatter_test_DF['ngram_abs_err'][rows]))

######################################################
###                                                ###
###            training stacking model             ###
###                                                ###
######################################################
#(only for crude info, for precise results it should have been done on separate train set)
xgb_X_train = vect_xgb.transform(ngram_train_DF[textcol])
hatter_train_DF['ngram_preds_proba'] = xgb_en.predict_proba(xgb_X_train)[:,1]
hatter_train_DF['stat_preds_proba'] = mdl9.predict_proba(stat_train_DF[varcols])[:,1]

f1_scorer = make_scorer(f1_score, average='weighted')
acc_scorer = make_scorer(accuracy_score)
scoring = {'F1': f1_scorer, 'Accuracy': acc_scorer}

params = {'solver' : ['saga'],
        'penalty' : ['elasticnet'],
        'C' : [0, 0.1, 0.2, 0.4, 0.7, 0.9, 1, 1.2, 1.5, 2, 3, 5],
        'l1_ratio' : [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}


#CV train staking model
logreg_clf = LogisticRegression()
logreg = GridSearchCV(logreg_clf,
                      param_grid=params,
                      scoring=scoring,
                      refit = "Accuracy",
                      return_train_score=True,
                      cv=5,
                      verbose=1,
                      n_jobs = -1)
logreg.fit(hatter_train_DF[['ngram_preds_proba', 'stat_preds_proba']], hatter_train_DF["roma"])

print()
#performance metrics and hyperpars
print(logreg.best_score_, logreg.best_params_, "\n",
      logreg.best_estimator_, "\n",
      logreg.best_estimator_.coef_, logreg.best_estimator_.intercept_, '\n')

#save model
joblib.dump(logreg, 'analysis/stacking_logreg')

###############################
###
###     test performance
###
###############################
stacking_preds = logreg.predict(hatter_test_DF[['ngram_preds_proba', 'stat_preds_proba']])
print(accuracy_score(hatter_test_DF.roma, stacking_preds))
print(f1_score(hatter_test_DF.roma, stacking_preds))
print(classification_report(hatter_test_DF.roma, stacking_preds))

###
###   performance by gender
###
rows = hatter_test_DF.gender == 'female'
print('acc:', accuracy_score(hatter_test_DF['roma'][rows], stacking_preds[rows]))
print('f1:', f1_score(hatter_test_DF['roma'][rows], stacking_preds[rows]))
print(classification_report(hatter_test_DF['roma'][rows], stacking_preds[rows]))

rows = hatter_test_DF.gender != 'female'
print('acc:', accuracy_score(hatter_test_DF['roma'][rows], stacking_preds[rows]))
print('f1:', f1_score(hatter_test_DF['roma'][rows], stacking_preds[rows]))
print(classification_report(hatter_test_DF['roma'][rows], stacking_preds[rows]))