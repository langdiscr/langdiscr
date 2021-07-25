######################################################
###                                                ###
###                import packages                 ###
###                                                ###
######################################################
from collections import Counter

import numpy as np
import os
import pandas as pd
import random
import re

import spacy
from spacy import displacy

#-m pip install https://github.com/oroszgy/spacy-hungarian-models/releases/download/hu_core_ud_lg-0.3.1/hu_core_ud_lg-0.3.1-py3-none-any.whl
import hu_core_ud_lg
nlp = hu_core_ud_lg.load()

######################################################
###                                                ###
###                    load data                   ###
###                                                ###
######################################################
RAW = pd.read_stata('data/raw/working.dta')
print(RAW.shape)
print(RAW.columns)

######################################################
###                                                ###
###          identify automatic responses          ###
###                                                ###
######################################################
RAW['error_in_message'] = RAW.response.apply(lambda x: 'error' in x.lower())
RAW['rejected_in_message'] = RAW.response.apply(lambda x: 'rejected' in x.lower())
RAW['denied_in_message'] = RAW.response.apply(lambda x: 'denied' in x.lower())
RAW['failed_in_message'] = RAW.response.apply(lambda x: 'failed' in x.lower())
RAW['nemsik_in_message'] = RAW.response.apply(lambda x: 'üzenetét nem sikerült kézbesíteni' in x.lower())
RAW['nemtud_in_message'] = RAW.response.apply(lambda x: ('üzenetét nem tudtuk kézbesíteni' in x.lower()) or
                                              ('your message could notbe delivered' in x.lower()))
RAW['befejezetlen_in_message'] = RAW.response.apply(lambda x: 'befejezetlen kézbesítés' in x.lower())
RAW['auto_resp']= RAW.response.apply(lambda x: ('e-mail címét megváltoztattuk' in x.lower()) or
                                     ('Kérem helyette a következõ e-mail címeket használja' in x.lower()) or
                                     ('auto reply' in x.lower()))
RAW['no_resp'] = RAW.response.apply(lambda x: x == '')

RAW['no_real_resp'] = (1-
                       (1-RAW.error_in_message) *
                       (1-RAW.rejected_in_message) *
                       (1-RAW.denied_in_message) *
                       (1-RAW.failed_in_message) *
                       (1-RAW.nemsik_in_message) *
                       (1-RAW.nemtud_in_message) *
                       (1-RAW.befejezetlen_in_message) *
                       (1-RAW.auto_resp) *
                       (1-RAW.no_resp))

######################################################
###                                                ###
###             cleaning emails                    ###
###                                                ###
######################################################

RAW['response_cleaned'] = ''

#########################################
###
###   making regex pattern for date tokens

hnpk = ['január',
        'február',
        'március',
        'április',
        'május',
        'június',
        'július',
        'augusztus',
        'szepttember',
        'október',
        'november',
        'december',
        'jan.',
        'feb.',
        'febr.',
        'márc.',
        'ápr.',
        'máj.',
        'jún.',
        'júl.',
        'aug.',
        'szept.',
        'okt.',
        'nov.',
        'dec.']
p_dat = ''
for h in hnpk:
  if p_dat == '':
    p_dat += re.escape(h + ' [SZÁM]')
  else:
    p_dat += '|'
    p_dat += re.escape(h + ' [SZÁM]')

for h in hnpk:
  p_dat += '|'
  p_dat += re.escape('[SZÁM]. ' + h + ' [SZÁM]')
  p_dat += '|'
  p_dat += re.escape('[SZÁM]. ' + h)

#########################################
###
###   removing headers, orig & forw parts, insert tokens

sv = '' #string from cleaned responses (to check for remaining manually removable parts)

resp_inds = list(np.where(RAW.no_real_resp == 0)[0]) #indices of responses
for i in range(len(RAW)):
  if i in resp_inds: #answered
    cont = True     
    # print('+'*50, i)
    resp = RAW['response_raw'][i]

    # add new response header to sv
    sv += str(i)
    sv += RAW['town'][i]
    sv += '\n'
    sv += resp
    sv += '\n'
    sv += '- + '*25
    sv += '\n\n'

    telp = RAW['town'][i] #name of town

    #replace characters
    resp = re.sub(r"õ", 'ő', resp) 
    resp = re.sub(r"Õ", 'Ő', resp)

    #remove parts after '>>>' (automatic copy of original message)
    if '>>>' in resp:
        resp = re.split(r'\n*>>>', resp)[0]
        cont = False
    
    if cont: #if did not yet remove automatic copy of orig message
      #pattern for orig/forw message header
      pattern = (r'\s*-*\s*[Tt]ovábbított üzenet-*|'+
                  '\s*-*\s*[Ff]orwarded [Mm]essage-*|'+
                  '\s*-*\s*[Oo]riginal [Mm]essage-*|'+
                  '\s*-*\s*[Ee]redeti üzenet-*|'+
                  '\s*-*\s*[Ee]redeti levél-*')
      #pattern for sender and date in orig/forw message header
      p1 = r'\s*.*írta.*'
      p2 = r'\s*.*2020.*'

      ujra = True #repaet until no more orig/forw message left (until no change made in 1 cycle)
      while ujra: #remove forw & orig message parts
        ujra = False
        while re.search(r'\s',resp).start() == 0: # remove white spaces from beginning of message
          resp = resp[re.search(r'\s',resp).end():]
          ujra = True
        cont2 = True
        while ((len(re.findall(p1,resp))>0) & #if there is both p1 & p2 in resp & changes were made
              (len(re.findall(p2,resp))>0) &
              cont2):
          cont2 = False
          if ((re.findall(p1,resp)[0] == re.findall(p2,resp)[0]) & #if p1 & p2 same (forw or orig message header) & in beginning of resp --> remove
              (re.search(p1, resp).start() < 3)):
            resp = resp[re.search(p1,resp).end():]
            ujra = True
            cont2 = True
        cont3 = True
        while (len(re.findall(pattern,resp))>0) & cont3:
          cont3 = False
          if re.search(pattern,resp).start() < 3: #if forw, orig header at the beginning of message --> remove first part up to first double whitespace
            resp = resp[re.search(r'\n\n|\r\r|\r\n\r\n', resp).end():]
            ujra = True
            cont3 = True
        while (resp[0] == '>') or (resp[-1] == '>'): #strips '>' from beginning & end
          resp = resp.strip()
          resp = resp.strip(">")
          resp = resp.strip()
          ujra = True


      resp = re.split(pattern, resp)[0] #removing text after forw or orig header

      #remove everything before orig/forw header (if there is still any left)
      stl = 10000000

      f1 = re.findall(p1,resp)
      f2 = re.findall(p2,resp)

      for f1i in f1:
        st1 = re.search(re.escape(f1i), resp).start()
        for f2i in f2:
          st2 = re.search(re.escape(f2i), resp).start()

          if (st1 == st2) and (st1 < stl):
            # print('st1',st1)
            stl = st1
      if stl < 10000000:
        resp = resp[:stl]
    
    ####  insert tokens     ####
    p_email = r'[^\s]*@[^\s]*'
    resp = re.sub(p_email, '[EMAIL]', resp)
    p_web = r'[^\s]*www[^\s]*|http[^\s]*'
    resp = re.sub(p_web, '[LINK]', resp)
    p_tel = r'[+]{0,1}\s{0,1}\d{2,11}[\d\-()/\s]{6,13}\d'
    resp = re.sub(p_tel, '[TELEFONSZÁM]', resp)
    resp = re.sub(r'NAME', '[NAME]', resp)
    resp = re.sub(telp, '[TELEPÜLÉS]', resp)
    resp = re.sub(telp.upper(), '[TELEPÜLÉS]', resp)      
    resp = re.sub(r'\b' + telp.lower(), '[TELEPÜLÉS]', resp)
    if telp[-1] == 'a':
      telp = telp[:-1] + 'á'
      resp = re.sub(telp, '[TELEPÜLÉS]', resp)
      resp = re.sub(telp.upper(), '[TELEPÜLÉS]', resp)      
      resp = re.sub(r'\b' + telp.lower(), '[TELEPÜLÉS]', resp)
    if telp[-1] == 'e':
      telp = telp[:-1] + 'é'
      resp = re.sub(telp, '[TELEPÜLÉS]', resp)
      resp = re.sub(telp.upper(), '[TELEPÜLÉS]', resp)      
      resp = re.sub(r'\b' + telp.lower(), '[TELEPÜLÉS]', resp)
    p_szam = r'\d+[\./]*\d+|\d+'
    resp = re.sub(p_szam, '[SZÁM]', resp)

    while ((resp != resp.strip(">")) or 
          (resp != resp.strip())):
      resp = resp.strip(">")
      resp = resp.strip()

    doc = nlp(resp[-int(len(resp)/3):])

    #names
    nevl = [X.text.splitlines()[0] for X in doc.ents if ((X.label_ == 'PER') &
                                                         (len(X.text.split(' ')) > 1))]
    for nev in nevl:
      nev = nev.split('[LINK]')[0]
      if 'telefon' in nev.lower():          
        nev = nev.split('tel')[0]
        nev = nev.split('Tel')[0]
        nev = nev.split('TEL')[0]
      if 'LINK' not in nev:
        resp = re.sub(re.escape(nev), '[NAME]', resp)
   
    resp = re.sub(r'[dD]r\.', '', resp)
    resp = re.sub(r'\[SZÁM\]\.\[SZÁM\]\.{0,1}', '[DÁTUM]', resp)
    resp = re.sub(p_dat, '[DÁTUM]', resp)
    resp = re.sub(re.escape('[SZÁM]:[SZÁM]'), '[IDŐ]', resp)
    resp = re.sub(r'\[SZÁM\]\.\[SZÁM\]\.\[SZÁM\]\.{0,1}', '[DÁTUM]', resp)
    resp = re.sub(r' +', ' ', resp)
    resp = re.sub(r'\n{2,}|\r{2,}|(\r\n){2,}', '\n', resp)

    RAW.at[i,'response_cleaned'] = resp

    #insert cleaned response in sv
    sv += resp
    sv += '\n\n'
    sv += '+ '*100
    sv += '\n'
    sv += ' +'*100
    sv += '\n'
    sv += '+ '*100
    sv += '\n\n'

###########################
###
### save results

RAW.to_excel('data/processed/cleaned_v0_c.xlsx')

with open("resources/clean/raw_resps_v0_c.txt", "w") as text_file:
    text_file.write(sv)

###################################################
###
###  manual corrections from manual_corr_1.txt based on raw_resps_v0_c.txt
###  (deleting leftovers, changing & adding not correct tokens)

with open("resources/clean/manual_corr_1.txt", 'r') as f1:
  f1_ls = f1.readlines()

for l in f1_ls:
  ind = int(l.split(' - ')[0].strip())
  jav_l = [jav.strip() for jav in l.split(' - ')[1].split('//')]
  for jav in jav_l:
    resp = RAW.response_cleaned[ind]
    if jav.lower()[-3:] == '(t)':
      resp = re.sub(re.escape(jav[:-3].strip()), '', resp)
    elif '-->' in jav:
      resp = re.sub(re.escape(jav.split('-->')[0].strip()), jav.split('-->')[1].strip(), resp)
    else:
      if len(jav.split()) == 1:
        resp = re.sub(re.escape(jav), '[TELEPÜLÉS]', resp)
      else:
        resp = re.sub(re.escape(jav), '[NAME]', resp)
    RAW.at[ind,'response_cleaned'] = resp

###################################################
###
###  other manual corrections based on raw_resps_v0_c.txt (token corrections)

def manual_clean(resp, telp, name = True):
  p_email = r'[^\s]*@[^\s]*'
  resp = re.sub(p_email, '[EMAIL]', resp)
  p_web = r'[^\s]*www[^\s]*|http[^\s]*'
  resp = re.sub(p_web, '[LINK]', resp)
  p_tel = r'[+]{0,1}\s{0,1}\d{2,11}[\d\-()/\s]{6,13}\d'
  resp = re.sub(p_tel, '[TELEFONSZÁM]', resp)
  resp = re.sub(r'NAME', '[NAME]', resp)
  resp = re.sub(telp, '[TELEPÜLÉS]', resp)
  resp = re.sub(telp.upper(), '[TELEPÜLÉS]', resp)      
  resp = re.sub(r'\b' + telp.lower(), '[TELEPÜLÉS]', resp)
  if telp[-1] == 'a':
    telp = telp[:-1] + 'á'
    resp = re.sub(telp, '[TELEPÜLÉS]', resp)
    resp = re.sub(telp.upper(), '[TELEPÜLÉS]', resp)      
    resp = re.sub(r'\b' + telp.lower(), '[TELEPÜLÉS]', resp)
  if telp[-1] == 'e':
    telp = telp[:-1] + 'é'
    resp = re.sub(telp, '[TELEPÜLÉS]', resp)
    resp = re.sub(telp.upper(), '[TELEPÜLÉS]', resp)      
    resp = re.sub(r'\b' + telp.lower(), '[TELEPÜLÉS]', resp)
  p_szam = r'\d+[\./]*\d+|\d+'
  resp = re.sub(p_szam, '[SZÁM]', resp)

  while ((resp != resp.strip(">")) or 
        (resp != resp.strip())):
    resp = resp.strip(">")
    resp = resp.strip()

  if name == True:
    doc = nlp(resp[-int(len(resp)/3):])
    nevl = [X.text.splitlines()[0] for X in doc.ents if ((X.label_ == 'PER') & (len(X.text.split(' ')) > 1))]
    for nev in nevl:
      nev = nev.split('[LINK]')[0]
      if 'telefon' in nev.lower():          
        nev = nev.split('tel')[0]
        nev = nev.split('Tel')[0]
        nev = nev.split('TEL')[0]
      if 'LINK' not in nev:
        resp = re.sub(re.escape(nev), '[NAME]', resp)
  resp = re.sub(r'[dD]r\.', '', resp)
  resp = re.sub(r'\[SZÁM\]\.\[SZÁM\]\.{0,1}', '[DÁTUM]', resp)
  resp = re.sub(p_dat, '[DÁTUM]', resp)
  resp = re.sub(re.escape('[SZÁM]:[SZÁM]'), '[IDŐ]', resp)
  resp = re.sub(r'\[SZÁM\]\.\[SZÁM\]\.\[SZÁM\]\.{0,1}', '[DÁTUM]', resp)
  resp = re.sub(r' +', ' ', resp)
  resp = re.sub(r'\n{2,}|\r{2,}|(\r\n){2,}', '\n', resp)

  return resp

#### individual manual corrections removed from code for privacy reasons

RAW.to_excel('data/processed/cleaned_v1.xlsx')

##########################################################
###
### manually tokenizing remaining locations & names

text = ''
for r in RAW.response_cleaned[RAW.no_real_resp == 0]:
  text += r + '\n'

doc_full = nlp(text)

entities = doc.ents

#creating location list
locs = ''
for l in [X.text for X in doc.ents if (X.label_ == 'LOC')]:
  if locs != '':
    locs += '\n' + l
  else:
    locs += l

with open("resources/clean/raw_locs.txt", "w") as text_file:
    text_file.write(locs)

#creating name list
len([X.text for X in doc.ents if (X.label_ == 'PER')])
pers = ''
for l in [X.text for X in doc.ents if (X.label_ == 'PER')]:
  if pers != '':
    pers += '\n' + l
  else:
    pers += l

with open("resources/clean/raw_pers.txt", "w") as text_file:
    text_file.write(pers)

#loading manually checked lists (adresses and village names separated)
with open("resources/clean/pers.txt", 'r') as f:
  pers = f.readlines()
  pers = [p.strip('\n') for p in pers]
  pers = sorted(list(set(pers)), reverse=True)
  print('pers', len(pers))
with open("resources/clean/locs.txt", 'r') as f:
  locs = f.readlines()
  locs = [p.strip('\n') for p in locs]
  locs = sorted(list(set(locs)), reverse=True)
  locs.remove('Fő út')
  print('locs', len(locs))
with open("resources/clean/telep.txt", 'r') as f:
  telep = f.readlines()
  telep = [p.strip('\n') for p in telep]
  telep = sorted(list(set(telep)), reverse=True)
  print('telep', len(telep))

#making pattern from lists
def pattern_from_list(pat, lis):
  for l in lis:
    if pat != '':
      pat += '|' + re.escape(l)
    else:
      pat += re.escape(l)
  return pat

p_pers = ''
p_pers = pattern_from_list(p_pers, pers)
p_telep = ''
p_telep = pattern_from_list(p_telep, telep)
p_locs = ''
p_locs = pattern_from_list(p_locs, locs)

# tokenizing items of lists
for i in range(len(RAW)):
  if RAW.no_real_resp[i] == 0:
    resp = RAW.response_cleaned[i]
    resp = re.sub(p_locs, '[CIM]', resp)
    resp = re.sub(p_pers, '[NAME]', resp)
    resp = re.sub(p_telep, '[TELEPÜLÉS_M]', resp)
    #manual corr removed
    resp = re.sub(re.escape('[NAME] utca'), '[CIM]', resp)
    resp = re.sub(re.escape('[NAME] u.'), '[CIM]', resp)
    resp = re.sub(re.escape('[NAME] út'), '[CIM]', resp)
    resp = re.sub(re.escape('[CIM]ja'), '[CIM]', resp)
    resp = re.sub(re.escape('[CIM]') + r'[^\s]{0,1}n' + r'\b', '[CIM]-en', resp)
    resp = re.sub(re.escape('[CIM]')+ r'[a-z]+' + r'\b', lambda x: ']-'.join(x[0].split(']')), resp)
    # standardizing conjugation
    resp = re.sub(re.escape('[TELEPÜLÉS_M]')+ r'\w' + '[ae]l' + r'\b', '[TELEPÜLÉS_M]-val', resp)
    resp = re.sub(re.escape('[TELEPÜLÉS_M]b')+  '[ae]', '[TELEPÜLÉS_M]-ba', resp)
    resp = re.sub(re.escape('[TELEPÜLÉS_M]r')+  '[ae]', '[TELEPÜLÉS_M]-ra', resp)
    resp = re.sub(re.escape('[TELEPÜLÉS_M]r')+  '[óő]', '[TELEPÜLÉS_M]-ró', resp)
    resp = re.sub(re.escape('[TELEPÜLÉS_M]t')+  '[óő]', '[TELEPÜLÉS_M]-tó', resp)
    resp = re.sub(re.escape('[TELEPÜLÉS_M]b')+  '[óő]', '[TELEPÜLÉS_M]-bó', resp)
    resp = re.sub(re.escape('[TELEPÜLÉS_M]')+  r'\w*t' + r'\b', '[TELEPÜLÉS_M]-t', resp)
    resp = re.sub(re.escape('[TELEPÜLÉS_M]h')+  r'\w*z' + r'\b', '[TELEPÜLÉS_M]-hoz', resp)
    resp = re.sub(re.escape('[TELEPÜLÉS_M]n')+  r'\w*l' + r'\b', '[TELEPÜLÉS_M]-nél', resp)
    resp = re.sub(re.escape('[TELEPÜLÉS_M]')+  r'\w*n' + r'\b', '[TELEPÜLÉS_M]-en', resp)
    resp = re.sub(re.escape('[TELEPÜLÉS_M]ak'), '[TELEPÜLÉS_M]-iak', resp)
    resp = re.sub(re.escape('[TELEPÜLÉS_M]ére'), '[TELEPÜLÉS_M]-re', resp)
    resp = re.sub(re.escape('[TELEPÜLÉS]b')+  '[ae]', '[TELEPÜLÉS]-ba', resp)
    resp = re.sub(re.escape('[TELEPÜLÉS]r')+  '[ae]', '[TELEPÜLÉS]-ra', resp)
    resp = re.sub(re.escape('[TELEPÜLÉS]r')+  '[óő]', '[TELEPÜLÉS]-ró', resp)
    resp = re.sub(re.escape('[TELEPÜLÉS]t')+  '[óő]', '[TELEPÜLÉS]-tó', resp)
    resp = re.sub(re.escape('[TELEPÜLÉS]b')+  '[óő]', '[TELEPÜLÉS]-bó', resp)
    resp = re.sub(re.escape('[TELEPÜLÉS]')+  r'\w*t' + r'\b', '[TELEPÜLÉS]-t', resp)
    resp = re.sub(re.escape('[TELEPÜLÉS]h')+  r'\w*z' + r'\b', '[TELEPÜLÉS]-hoz', resp)
    resp = re.sub(re.escape('[TELEPÜLÉS]n')+  r'\w*l' + r'\b', '[TELEPÜLÉS]-nél', resp)
    resp = re.sub(re.escape('[TELEPÜLÉS]')+  r'\w*n' + r'\b', '[TELEPÜLÉS]-en', resp)
    resp = re.sub(re.escape('[TELEPÜLÉS]')+ r'\w' + '[ae]l' + r'\b', '[TELEPÜLÉS]-val', resp)
    resp = re.sub(re.escape('[TELEPÜLÉS_M]')+  r'\w*' + r'\b', lambda x: ']-'.join(x[0].split(']')), resp)
    resp = re.sub(re.escape('[TELEPÜLÉS]')+ r'[a-z]+' + r'\b', lambda x: ']-'.join(x[0].split(']')), resp)

    RAW.at[i,'response_cleaned'] = resp

#saving output
RAW.to_excel('data/processed/cleaned_v2.xlsx')

#########################################
###
###   removing automatic signatures (with phone numbers, urls etc.)

sv = ''
for i in range(len(RAW)):
  if (RAW.no_real_resp[i] == 0):
    sv += '\n' + '-'*20 + '\n' + '#' * 5 + '  ' + str(i) +'\n' + '-'*20

    resp = RAW.response_cleaned[i]
    # removing double newlines
    resp = re.sub('\r\n', '\n', resp)
    resp = re.sub('\r\r', '\n', resp)
    while re.sub(r'\n\s*\n', '\n', resp) != resp:
      resp = re.sub(r'\n\s*\n', '\n', resp)
    resp = resp.strip()

    ln = '' #init
    #iterate lines bacward & identify first line of automatic signature
    for l in range(len(resp.split("\n"))):
      lnp = ln #previously examined line
      ln = resp.split("\n")[-(l+1)]
      ws = [w.strip() for w in ln.split() if len(w)>1] #list of words
      if (
          (len(re.findall('\[', ln)) < len(ws)/2) or #if less than half of words are token 
          (('vírus' in ln) & (l == 0)) or #there isn't an automatic virus scan info at the end          
          (('[NAME]' in lnp) & #previosly seen line is short and there is a name in it & there is some kind of farwell formula in the line
           (len([w for w in lnp.strip('>').strip().split() if len(w)>1]) < 3) &
           (('tiszt' in ln.lower()) or
            ('kösz' in ln.lower()) or
            ('üdv' in ln.lower()) or
            ('nevéb' in ln.lower()) or
            ('megbíz' in ln.lower())))
          ):
        l -= 1
        break       
    
    resp_t = ''
    if l != 0:
      resp_t = '\n'.join(resp.split("\n")[-l:])
      resp = '\n'.join(resp.split("\n")[:-l])
    sv += '\n' + resp
    sv += '\n' + '-+'*50 + '  ' + str(l)
    sv += '\n' + resp_t
    sv += '\n' +'\\'*100
    sv += '\n' +'\\'*100
    RAW.at[i,'response_cleaned'] = resp

RAW.to_excel('data/processed/cleaned_v3.xlsx')

with open("resources/clean/raw_resps_v3.txt", "w") as text_file:
    text_file.write(sv)

##########################
##
## manual corrections based on raw_resps_v3.txt

with open("resources/clean/manual_corr_3.txt", 'r') as f1:
  f1_ls = f1.readlines()

for sz, sor in enumerate(reversed(f1_ls)):
  sor = sor.strip('\n')
  lis = sor.split(';')
  ind = int(lis[0])
  resp = RAW.response_cleaned[ind]
  # print(lis)
  if lis[1] == 'p':
    resp += '\n' + lis[2]
  elif lis[1] == 'm':
    if lis[2] in resp.split('\n')[-1]:
      resp = '\n'.join(resp.split('\n')[:-1])
    else:
      print("ROSSZ SOR", sor, 'R '*100, '\n', resp, '\n', '~'*100)
  else:
    print('ROSSZ FORMA', 'R '*100)
  
  RAW.at[ind, 'response_cleaned'] = resp
  # if sz > 20:
  #   break

RAW.to_excel('data/processed/cleaned_v3.xlsx')

######################################################
###                                                ###
###                 text editing                   ###
###                                                ###
######################################################

resp_inds = list(np.where(RAW.no_real_resp == 0)[0])
for i in range(len(RAW)):
  if i in resp_inds:
    resp = RAW['response_cleaned'][i]
    #removing all non single whitespace
    resp = resp.strip()
    resp = re.sub(r'\n\s*[/*]+\s*|\s*[/*]+\s*\n', '\n', resp)
    resp = resp.strip()
    resp = re.sub(r'\s+>*\n>*', '\n', resp)
    resp = re.sub(r'\n>*\s+>*', '\n', resp)
    resp = re.sub(r'\s{2,}', '\s', resp)
    resp = resp.strip()
    RAW.at[i,'response_cleaned'] = resp
    #removing newlines
    RAW.at[i,'response_cleaned_1line'] = " ".join(resp.split())

RAW.to_excel('data/processed/cleaned_v4.xlsx')

##################
##
##  standardizing conjugation

resp_inds = list(np.where(RAW.no_real_resp == 0)[0])
for i in range(len(RAW)):
  if i in resp_inds: #ha van válasz
    # print('+'*50, i)
    tx = RAW['response_cleaned'][i]

    tx = re.sub(r'\[[A-Z]\w+\]-on[aeáé][kl]\b', lambda x: ''.join(x[0].split('o')),tx)
    tx = re.sub(r'\[[A-Z]\w+\]-[öüóeuioőúűáéaí]s\b', lambda x: x[0].split('-')[0]+'-as',tx)
    tx = re.sub(r'\[[A-Z]\w+\]-b[ae]n\b', lambda x: x[0].split('-')[0]+'-ban',tx)
    tx = re.sub(r'\[[A-Z]\w+\]-b[óő]l\b', lambda x: x[0].split('-')[0]+'-ból',tx)
    tx = re.sub(r'\[[A-Z]\w+\]-[öüáéoei]{0,1}n\b', lambda x: x[0].split('-')[0]+'-on',tx)
    tx = re.sub(r'\[[A-Z]\w+\]-\w{0,2}ig\b', lambda x: x[0].split('-')[0]+'-ig',tx)
    tx = re.sub(r'\[[A-Z]\w+\]-n[ae]k\b', lambda x: x[0].split('-')[0]+'-nak',tx)
    tx = re.sub(r'\[[A-Z]\w+\]-n[áé]l\b', lambda x: x[0].split('-')[0]+'-nál',tx)
    tx = re.sub(r'\[[A-Z]\w+\]-r[ae]\b', lambda x: x[0].split('-')[0]+'-ra',tx)
    tx = re.sub(r'\[[A-Z]\w+\]-\w{0,2}t\b', lambda x: x[0].split('-')[0]+'-t',tx)
    tx = re.sub(r'\[[A-Z]\w+\]-\w{0,2}t\wl\b', lambda x: x[0].split('-')[0]+'-tól',tx)
    tx = re.sub(r'\[[A-Z]\w+\]-\w{0,2}v[ae]l\b', lambda x: x[0].split('-')[0]+'-val',tx)
    tx = re.sub(r'\[DÁTUM\]-j{0,1}e\b', '[DÁTUM]',tx)
    
    RAW.at[i,'response_cleaned'] = tx
    RAW.at[i,'response_cleaned_1line'] = " ".join(tx.split())

######################################################
###                                                ###
###     lowercase, lemmatize, remove stopwords     ###
###                                                ###
######################################################

resp_inds = list(np.where(RAW.no_real_resp == 0)[0])
for i in range(len(RAW)):
  if i in resp_inds:
    tx = RAW['response_cleaned_1line'][i]

    tx = re.sub(r'\[[A-Z]\w+\]-\w{1,4}\b', lambda x: x[0].split('-')[0], tx) #removing suffixes from tokens
    tx = re.sub(r'\[[A-Z]\w+\]', lambda x: ' '+x[0][1:-1]+'_TOKEN ', tx) #removing [] from tokens
    tx = tx.lower()
    tx = re.sub(r'\W', lambda x: ' '+x[0]+' ', tx)
    tx = re.sub(r'\s+', ' ', tx)
    RAW.at[i,'response_cleaned_lower'] = tx

    doc = nlp(tx)
    tx = ' '.join([token.lemma_ for token in doc])    
    RAW.at[i,'response_cleaned_lemma'] = tx
    tx = ' '.join([token.text for token in doc if not token.is_stop])
    RAW.at[i,'response_cleaned_withoutstop'] = tx
    tx = ' '.join([token.lemma_ for token in doc if not token.is_stop])
    RAW.at[i,'response_cleaned_withoutstop_lemma'] = tx

#####################
###
###    POS tokens

resp_inds = list(np.where(RAW.no_real_resp == 0)[0])
for i in range(len(RAW)):
  if i in resp_inds: 
    tx = RAW['response_cleaned_lower'][i]
    doc = nlp(tx)
    tx = [token.pos_ if '_token' not in token.text else token.text for token in doc ]

    tx = [t if '_token' not in t else 'PROPN_t' if (('name' in t) or ('település' in t))  else 'NUM_t' for t in tx]
    RAW.at[i,'response_cleaned_pos'] = ' '.join(tx)

##################
###
###  checking for left special characters & removing them
charset = set([])
notcharset = set([])
for t in RAW[RAW.no_real_resp == 0].response_cleaned_withoutstop:
  notcharset |= set(re.findall(r'[^\w\s]', t))
  charset |= set(re.findall(r'[\w\s]', t))

for idx in range(len(RAW)):
  if RAW.loc[idx, 'no_real_resp'] == 0:
    for cm in ['response_cleaned_1line', 'response_cleaned_lower', 'response_cleaned_lemma', 'response_cleaned_withoutstop', 'response_cleaned_withoutstop_lemma']:
      t = RAW.loc[idx, cm]
      t = re.sub(r'\ufeff', ' ', t)
      t = re.sub(r'\x96', '-', t)
      t = re.sub(r'\x94', '"', t)
      RAW.loc[idx, cm] = t

### removing ponctuuation
for cm in ['response_cleaned_lower', 'response_cleaned_lemma', 'response_cleaned_withoutstop', 'response_cleaned_withoutstop_lemma']:
  RAW[cm+'_nopunct'] = RAW[cm].apply(lambda x: re.sub(r'[^\w\s]\s*', '',x) if pd.notna(x) else None)

#save output
RAW.to_excel('data/processed/cleaned_v5.xlsx')

######################################################
###                                                ###
### separate test& train set & fixed folds for CV  ###
###                                                ###
######################################################

DF = RAW[RAW.no_real_resp == 0][['town', 'wave']]
DF['main_fold'] = np.random.randint(1,6,len(DF))

idxs_r = np.where(DF.roma == 1)[0]
idxs_nr = np.where(DF.roma == 0)[0]
rand_idxs_r = list(np.random.choice(idxs_r, len(idxs_r), replace = False))
rand_idxs_nr = list(np.random.choice(idxs_nr, len(idxs_nr), replace = False))
test_idx_1 = rand_idxs_r[:100]+rand_idxs_nr[:100]
test_idx_2 = rand_idxs_r[100:200]+rand_idxs_nr[100:200]
for i in range(len(DF)):
  if i in test_idx_1: #for feat selection test################################################################
    DF.at[i,'test_fold'] = 1
  elif i in test_idx_2: #for model performance test
    DF.at[i,'test_fold'] = 2
    DF.at[i, 'CV1'] = np.random.choice([0,1,2,3])
  else:
    DF.at[i,'test_fold'] = 0
    DF.at[i, 'CV2'] = np.random.choice([0,1,2,3])
    DF.at[i, 'CV1'] = np.random.choice([0,1,2,3])

DF['CV_all_4f'] = np.random.choice([0,1,2,3], len(DF))
DF['CV_all_4f_2'] = np.random.choice([0,1,2,3], len(DF))

DF.head()

RAW = pd.merge(RAW, DF, how='outer')
RAW.head()

RAW.to_excel('data/processed/cleaned_v5_withfolds.xlsx')