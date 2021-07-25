#############################################
##                                         ##
##             importing packages          ##
##                                         ##
#############################################
from lexical_diversity import lex_div as ld
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import re
import seaborn as sns
import spacy
from spacy import displacy

#############################################
##                                         ##
##             importing data              ##
##                                         ##
#############################################
RAW = pd.read_excel('data/processed/cleaned_v5.xlsx', index_col=0)

#selecting rows with response
DF = RAW[RAW.no_real_resp == 0][['town',
                                 'wave',
                                 'treated',
                                 'cemetary',
                                 'nursery',
                                 'wedding',
                                 'roma',
                                 'white',
                                 'female',
                                 'response_cleaned',
                                 'response_cleaned_1line',
                                 'response_cleaned_lower',
                                 'response_cleaned_lemma',
                                 'response_cleaned_withoutstop',
                                 'response_cleaned_withoutstop_lemma',
                                 'response_cleaned_pos']]
DF.reset_index(inplace=True)

#############################################
##                                         ##
##           feature engineering           ##
##                                         ##
#############################################
#number of words
DF['num_words'] = DF.response_cleaned_lower.apply(lambda x: len(x.split())-len(re.findall(r'[^\w\s]',x)))
#number of punctuation marks
DF['num_punct'] = DF.response_cleaned_lower.apply(lambda x: len(re.findall(r'[^\w\s]',x)))
#number of different punctuation marks
DF['var_punct'] = DF.response_cleaned_lower.apply(lambda x: len(set(re.findall(r'[^\w\s]',x))))
#number of charachters
DF['num_chars'] = DF.response_cleaned_lower.apply(lambda x: len(re.findall(r'[^\W_]',x)))
#number of tokens
for tk in ['cim_token',
           'dátum_token',
           'email_token',
           'idő_token',
           'link_token',
           'name_token',
           'szám_token',
           'telefonszám_token',
           'település_m_token',
           'település_token']:
  DF['num_'+ tk] = DF.response_cleaned_lower.apply(lambda x: sum([1 for tok in x.split() if tk in tok]))

#ratio of diff tokens compared to number of words
for tk in ['cim_token',
           'dátum_token',
           'email_token',
           'idő_token',
           'link_token',
           'name_token',
           'szám_token',
           'telefonszám_token',
           'település_m_token',
           'település_token']:
  DF['ratio_' + tk] =  DF['num_'+ tk]/DF['num_words']

#number of sentences
DF['num_sent'] = DF.response_cleaned_lower.apply(lambda x: len(re.findall(r'[.!?•]',x)))
#moving-average type–token ratio - window: 21 (only 10% of letters are shorter)
#keeping only tokens with only word charachters
DF['MATTR'] = DF.response_cleaned_lemma.apply(lambda x: ld.mattr([w2 for w2 in [w for w in x.split() if len(re.findall(r'\W',w)) == 0] if w2 != '_'], window_length=21)) #21:10. percentil
#number of stopword (length diff of 2 different clening result)
DF['num_stopw'] = DF.response_cleaned_lemma.apply(lambda x: len(x.split())) - DF.response_cleaned_withoutstop_lemma.apply(lambda x: len(x.split()))

#nonlinear transformation of variables
DF['avg_length_sent'] = DF['num_words']/DF['num_sent']
DF['avg_length_word'] = DF['num_chars']/DF['num_words']
DF['ratio_stopw'] = DF['num_stopw']/DF['num_words']
DF['ratio_punct'] = DF['num_punct']/DF['num_chars']
DF['freq_punct'] =  DF['num_words']/DF['num_punct']

#number and ratio of POS tags
for tg in ['VERB','PROPN','NUM','CONJ','PRON','PART','NOUN', 'DET','AUX','ADV','ADJ','ADP']: #'INTJ': 12 db össz
  DF['num_'+tg] = DF.response_cleaned_pos.apply(lambda x: len([k for k in [p for p in x.split() if p not in ['X', 'SPACE', 'PUNCT']] if tg in k]))
  DF['ratio_'+tg] = (DF.response_cleaned_pos.apply(lambda x: len([k for k in [p for p in x.split() if p not in ['X', 'SPACE', 'PUNCT']] if tg in k]))/
                     DF.response_cleaned_pos.apply(lambda x: len([p for p in x.split() if p not in ['X', 'SPACE', 'PUNCT']])))
  print(tg, sum(DF['num_'+tg]))

#replace nans etc. with 0
DF.fillna(0, inplace=True)
DF.replace(np.inf, 0, inplace=True)

DF.to_excel('data/processed/stat_model_df.xlsx')

#############################################
##                                         ##
##        descr stats of features          ##
##                                         ##
#############################################

#####################################
###
###   histograms & boxplots

DF = pd.read_excel('data/processed/stat_model_df.xlsx', index_col=0)

hiba = []
for cl in ['num_words', 'num_punct',
       'var_punct', 'num_chars', 'num_cim_token', 'num_dátum_token',
       'num_email_token', 'num_idő_token', 'num_link_token', 'num_name_token',
       'num_szám_token', 'num_telefonszám_token', 'num_település_m_token',
       'num_település_token', 'ratio_cim_token', 'ratio_dátum_token',
       'ratio_email_token', 'ratio_idő_token', 'ratio_link_token',
       'ratio_name_token', 'ratio_szám_token', 'ratio_telefonszám_token',
       'ratio_település_m_token', 'ratio_település_token', 'num_sent', 'MATTR',
       'num_stopw', 'avg_length_sent', 'avg_length_word', 'ratio_stopw',
       'ratio_punct', 'freq_punct', 'num_VERB', 'ratio_VERB', 'num_PROPN',
       'ratio_PROPN', 'num_NUM', 'ratio_NUM', 'num_CONJ', 'ratio_CONJ',
       'num_PRON', 'ratio_PRON', 'num_PART', 'ratio_PART', 'num_NOUN',
       'ratio_NOUN', 'num_DET', 'ratio_DET', 'num_AUX', 'ratio_AUX', 'num_ADV',
       'ratio_ADV', 'num_ADJ', 'ratio_ADJ', 'num_ADP', 'ratio_ADP']:
  print(cl, 'össz:', sum(DF[cl]), 'szórás:', np.std(DF[cl]), 'átlag:', np.mean(DF[cl]))
  try:
    #histogram
    DF[[cl]].plot.hist(figsize = (10,5))
    plt.show()
  except:
    pass
  print('# '*100)
  #boxplot
  DF.boxplot(column = cl,by='roma', figsize = (10,10))
  plt.show()

  try:
    #groupwise hist
    sns.displot(DF, x=cl, hue="roma", stat="density", common_norm=False)
    #groupwise estim distrib
    sns.displot(DF, x=cl, hue="roma", kind="kde", fill=True)
  except:
    hiba.append(cl)
  print('##'*100)
  print('##'*100)
print(hiba)

######################################
###
###   pairwise correlations

#correlation heatmap
cls = ['num_words', 'num_punct',
       'var_punct', 'num_chars', 'num_cim_token', 'num_dátum_token',
       'num_email_token', 'num_idő_token', 'num_link_token', 'num_name_token',
       'num_szám_token', 'num_telefonszám_token', 'num_település_m_token',
       'num_település_token', 'ratio_cim_token', 'ratio_dátum_token',
       'ratio_email_token', 'ratio_idő_token', 'ratio_link_token',
       'ratio_name_token', 'ratio_szám_token', 'ratio_telefonszám_token',
       'ratio_település_m_token', 'ratio_település_token', 'num_sent', 'MATTR',
       'num_stopw', 'avg_length_sent', 'avg_length_word', 'ratio_stopw',
       'ratio_punct', 'freq_punct', 'num_VERB', 'ratio_VERB', 'num_PROPN',
       'ratio_PROPN', 'num_NUM', 'ratio_NUM', 'num_CONJ', 'ratio_CONJ',
       'num_PRON', 'ratio_PRON', 'num_PART', 'ratio_PART', 'num_NOUN',
       'ratio_NOUN', 'num_DET', 'ratio_DET', 'num_AUX', 'ratio_AUX', 'num_ADV',
       'ratio_ADV', 'num_ADJ', 'ratio_ADJ', 'num_ADP', 'ratio_ADP']
cors = DF[cls].corr()
# cors
sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(15, 15)
ax = sns.heatmap(
    cors, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

#print pairwise corr values
cls = ['num_words', 'num_punct',
       'var_punct', 'num_chars', 'num_cim_token', 'num_dátum_token',
       'num_email_token', 'num_idő_token', 'num_link_token', 'num_name_token',
       'num_szám_token', 'num_telefonszám_token', 'num_település_m_token',
       'num_település_token', 'ratio_cim_token', 'ratio_dátum_token',
       'ratio_email_token', 'ratio_idő_token', 'ratio_link_token',
       'ratio_name_token', 'ratio_szám_token', 'ratio_telefonszám_token',
       'ratio_település_m_token', 'ratio_település_token', 'num_sent', 'MATTR',
       'num_stopw', 'avg_length_sent', 'avg_length_word', 'ratio_stopw',
       'ratio_punct', 'freq_punct', 'num_VERB', 'ratio_VERB', 'num_PROPN',
       'ratio_PROPN', 'num_NUM', 'ratio_NUM', 'num_CONJ', 'ratio_CONJ',
       'num_PRON', 'ratio_PRON', 'num_PART', 'ratio_PART', 'num_NOUN',
       'ratio_NOUN', 'num_DET', 'ratio_DET', 'num_AUX', 'ratio_AUX', 'num_ADV',
       'ratio_ADV', 'num_ADJ', 'ratio_ADJ', 'num_ADP', 'ratio_ADP']
cors = DF[cls].corr()
ng = np.where(abs(cors) > 0.7)
for ind, ert in enumerate(ng[0]):
  if ert < ng[1][ind]:
    print(cls[ert], cls[ng[1][ind]], DF[cls[ert]].corr(DF[cls[ng[1][ind]]]))


###
### removing highly correlated vars

cls = ['num_words',  'var_punct', 'ratio_cim_token', 'ratio_dátum_token',
       'ratio_email_token', 'ratio_idő_token', 'ratio_link_token',
       'ratio_name_token', 'ratio_szám_token', 'ratio_telefonszám_token',
       'ratio_település_m_token', 'ratio_település_token', 'MATTR',
       'avg_length_sent', 'avg_length_word', 'ratio_stopw',
       'ratio_punct', 'freq_punct', 'ratio_VERB', 'ratio_PROPN',
       'ratio_CONJ', 'ratio_PRON',  'ratio_PART',  'ratio_NOUN',  'ratio_DET',
       'ratio_AUX',  'ratio_ADV',  'ratio_ADJ',  'ratio_ADP']
#remainng max corr
cors = DF[cls].corr()
ng = np.where(abs(cors) > 0.7)
for ind, ert in enumerate(ng[0]):
  if ert < ng[1][ind]:
    print(cls[ert], cls[ng[1][ind]], DF[cls[ert]].corr(DF[cls[ng[1][ind]]]))