# -*- coding: utf-8 -*-

#to generate candidates on different dataset, please change the variables of "phrases_list_path" and "candidate_path"
phrases_list_path = "../phrase_list/twitter_g_train_list.txt"
dict_path="./data/dictionary_sampled.txt"
candidate_path = "./data/SPACY_cand_g_twitter.txt"

import numpy as np
import torch
import nltk
from nltk.corpus import stopwords
import spacy
import en_core_web_lg    
import ssl

try:
  _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
  ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
      
nlp=en_core_web_lg.load()

def remove_stopwords_fast(text):
    splited = text.split(' ')
    result = [s for s in splited if s not in stop_words]
    return " ".join(result)

phrases_list=[]
phrase_embeddings=[]
with open(phrases_list_path,'r',encoding='ascii',errors='ignore') as f:
  phrases_list = f.readlines()
f.close()
for i in range(len(phrases_list)):
  phrases_list[i] = phrases_list[i].rstrip()

#remove phrase with weird char
for i in range(len(phrases_list) - 1, -1, -1):
  tmp = ''.join([j for j in phrases_list[i] if not j.isdigit()])
  phrases_list[i] = " ".join(tmp.split())
  if '\u200b' in phrases_list[i]:
    phrases_list.remove(phrases_list[i])

  if phrases_list[i] in stop_words:
    phrases_list.remove(phrases_list[i])
    continue
  #remove all single char phrase
  phrases_list[i] = remove_stopwords_fast(phrases_list[i])
  if len(phrases_list[i]) <= 1:
    phrases_list.remove(phrases_list[i])


#txt里面有空行的
phrases_list = list(filter(None, phrases_list))

phrases_tokens = []

for i in range(len(phrases_list)):
  phrases_tokens.append(nlp(phrases_list[i].lower()))

dict_phrase = []
with open(dict_path,'r',encoding='utf-8') as f:
  dict_phrase=f.readlines()
f.close()

for i in range(len(dict_phrase)):
  dict_phrase[i]=dict_phrase[i].rstrip()
  dict_phrase[i]=dict_phrase[i].rstrip('\n')
  #might be improtant to make everything in lower case
  dict_phrase[i]=dict_phrase[i].lower()

dict_phrase = list(filter(None, dict_phrase))

dict_tokens = []

for i in range(len(dict_phrase)):
  dict_tokens.append(nlp(dict_phrase[i].lower()))


#generated candidates are stored in phrase_set
from scipy.spatial.distance import cosine
phrase_set_dict = set()
phrase_set = set()
for i in range(0, len(dict_phrase)):
  single_dict_token = dict_tokens[i]
  similarity_score = []
  num_words=dict_phrase[i].count(' ')
  for j in range(0, len(phrases_list)):
    if phrases_list[j].count(" ")!=num_words:
      similarity_score.append(0)
      continue
    single_phrase_token = phrases_tokens[j]
    similarity = single_phrase_token.similarity(single_dict_token)
    similarity_score.append(similarity)
  similarity_score = np.array(similarity_score)
  index_best = np.argsort(similarity_score)
  index_best = index_best[::-1]
  for n in range(0, 10):
    temp_res=phrases_list[index_best[n]]
    temp_score=similarity_score[index_best[n]]
    
    if temp_score > 0.6:
      if temp_res in dict_phrase:
        phrase_set_dict.add(temp_res)
      else:
        phrase_set.add(temp_res)

  if i %100 == 0:
    print(i)

my_stopword = ['first', 'second', ',','+', 'one', 'two','three','four','five','six','seven','eight','third','nine','ten']
test_string = 'first time'
candidates = list(phrase_set)

for i in range(len(candidates) - 1, -1, -1):
  for j in candidates[i].split(' '):
    if j in my_stopword:
      candidates.remove(candidates[i])
      break

candidates = candidates + dict_phrase

with open(candidate_path,'w',encoding='utf-8') as f:
  for line in candidates:
    f.write(line+'\n')
f.close()

for i in range(0, len(dict_phrase)):
  phrase_set.add(dict_phrase[i])
