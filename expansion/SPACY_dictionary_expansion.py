# -*- coding: utf-8 -*-

#the true location phrases of broad dataset, with some weird symbol needs to filter out
raw_data_path = "./data/locations_from_broad.txt"
#processes location phrases of broad dataset
process_data_path = "./data/locations_from_broad_process.txt"
#dictionary with name of countries and sampled city names
dict_path="./data/dictionary_sampled.txt"
#core dictionary combines the country-city dict and location phrases from broad dataset
dict_core_path = "./data/core_dict.txt"
#candidate generated from the twitter datasets which is similart to phrases in core dictionary combining with all the phrases in core dictionary
candidate_path = "./data/expanded_candidate.txt"

import spacy
import en_core_web_lg
import numpy as np
import torch
import ssl
import nltk
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def generate_core_dictionary():
  with open(raw_data_path,'r',encoding='utf-8') as f:
    lines = f.readlines()
  f.close()

  for i in range(len(lines)):
    lines[i] = lines[i].rstrip('\n')


  for i in range(len(lines)-1,-1,-1):
    if lines[i]=='@':
      lines.pop(i)

  for i in range(len(lines)):
    lines[i] = lines[i].replace('#','')
    lines[i] = lines[i].replace('_',' ')
    lines[i] = lines[i].replace(',','')
    lines[i] = lines[i].lstrip(' ')
    lines[i] = lines[i].replace('  ',' ')

  core_phrase = set()

  for i in range(len(lines)):
    core_phrase.add(lines[i].lower())

  phrases = list(core_phrase)

  for i in range(len(phrases)-1,-1,-1):
    if len(phrases[i]) < 2:
      phrases.pop(i)

  with open(process_data_path,'w',encoding='utf-8') as f:
    for line in phrases:
      f.write(line+'\n')
  f.close()

  dict_core = set()
  for i in phrases:
    tmp = nlp(i)
    if (tmp and tmp.vector_norm):
      dict_core.add(i)

  dict_phrase = []
  with open(dict_path,'r',encoding='utf-8') as f:
    dict_phrase=f.readlines()
  f.close()

  for i in range(len(dict_phrase)):
    dict_phrase[i]=dict_phrase[i].rstrip()
    dict_phrase[i]=dict_phrase[i].rstrip('\n')
    #might be improtant to make everything in lower case
    dict_phrase[i]=dict_phrase[i].lower()

  dict_tokens = []

  for i in range(len(dict_phrase)-1,-1,-1):
    tmp = nlp(dict_phrase[i].lower())
    if (tmp and tmp.vector_norm):
      dict_core.add(dict_phrase[i].lower())

  dict_core = list(dict_core)

  for i in range(len(dict_core)-1,-1,-1):
    if all(x.encode('UTF-8').isalpha() or x.isspace() for x in dict_core[i]):
      continue
    else:
      dict_core.pop(i)

  for i in range(len(dict_core)-1,-1,-1):
    if dict_core[i].count(' ') > 2:
      dict_core.pop(i)

  for i in range(len(dict_core)-1,-1,-1):
    if len(dict_core[i]) <= 2:
      dict_core.pop(i)

  with open(dict_core_path,'w',encoding='utf-8') as f:
    for line in dict_core:
      f.write(line+'\n')
  f.close()

  dict_tokens = []

  for i in range(len(dict_core)):
    dict_tokens.append(nlp(dict_core[i]))

  return dict_tokens, dict_core

def generate_candidate_phrase():
  try:
      _create_unverified_https_context = ssl._create_unverified_context
  except AttributeError:
  	  pass
  else:
      ssl._create_default_https_context = _create_unverified_https_context
  nltk.download('stopwords')
  stop_words = set(stopwords.words('english'))  
  stop_words.add('#')
  stop_words.add('|')
  stop_words.add('&')
  stop_words.add('@')

  def remove_stopwords_fast(text):
      splited = text.split()
      result = [s for s in splited if s not in stop_words]
      return " ".join(result)

  phrases_list=set()
  directories=['0322','0410','0515','0522','0620','0725','0805']
  for item in directories:
    with open("../phrase_list/{}.txt".format(item),'r',encoding='utf-8') as f:
        processed = f.readlines()
    f.close()
      
    for i in range(len(processed)):
      processed[i] = processed[i].rstrip()


    #remove phrase with more than 3 words
    for i in range(len(processed) - 1, -1, -1):
      if processed[i].count(' ') > 2:
        processed.pop(i)

    for i in range(len(processed) - 1, -1, -1):
      tmp = ''.join([j for j in processed[i] if not j.isdigit()])
      processed[i] = " ".join(tmp.split())

      if processed[i] in stop_words:
        processed.remove(processed[i])
        continue
      #remove all single char phrase
      processed[i] = remove_stopwords_fast(processed[i])
      if len(processed[i]) <= 1:
        processed.pop(i)
    
    for i in range(len(processed)):
      phrases_list.add(processed[i])


  phrases_list = list(phrases_list)

  for i in range(len(phrases_list)-1,-1,-1):
    if all(x.encode('UTF-8').isalpha() or x.isspace() for x in phrases_list[i]):
      continue
    else:
      phrases_list.pop(i)

  print("#######convert candidate phrases into tokens")
  phrases_tokens = []
  count = 0
  for i in range(len(phrases_list)-1,-1,-1):
    tmp = nlp(phrases_list[i].lower())
    if not (tmp and tmp.vector_norm):
      count = count + 1
      phrases_list.pop(i)
    else:
      phrases_tokens.append(tmp)
    if i%1000 ==0 :
      print(i)


  phrases_tokens = phrases_tokens[::-1]

  return phrases_tokens, phrases_list

def expansion(dict_tokens, phrases_tokens, dict_core, phrases_list):
  phrase_set_dict = set()
  phrase_set = set()

  print("#######compute similarity")
  for i in range(0, len(dict_tokens)):
    single_dict_token = dict_tokens[i]
    similarity_score = []
    num_words=dict_core[i].count(' ')
    for j in range(0, len(phrases_list)):
      if phrases_list[j].count(' ')!=num_words:
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
        if temp_res in dict_core:
          phrase_set_dict.add(temp_res)
        else:
          phrase_set.add(temp_res)

    if i %100 == 0:
      print(i)

  my_stopword = ['first', 'second', ',','+', 'one', 'two','three','four','five','six','seven','eight','third','nine','ten', 'http','https']
  test_string = 'first time'
  candidates = list(phrase_set)
  removed_candidates = list(phrase_set)

  for i in range(len(candidates) - 1, -1, -1):
    
    for j in candidates[i].split():
      if j in my_stopword:  
        candidates.pop(i)
        removed_candidates.pop(i)
        break

  phrase_in_dict = list(phrase_set_dict)


  all_phrase = candidates + phrase_in_dict
  all_phrase_vectors = []
  for i in range(len(all_phrase)):
    all_phrase_vectors.append(nlp(all_phrase[i]).vector)
  all_phrase_vectors = np.array(all_phrase_vectors)
  pca_model = PCA(n_components = 2)
  pca_model.fit(all_phrase_vectors)
  all_phrase_decompose = pca_model.fit_transform(all_phrase_vectors)


  phrase_decompose = []
  dict_decompose = []
  for i in range(all_phrase_decompose.shape[0]):
    if all_phrase[i] in dict_core:
      dict_decompose.append(all_phrase_decompose[i])
    else:
      phrase_decompose.append(all_phrase_decompose[i])
  dict_decompose = np.array(dict_decompose)
  phrase_decompose = np.array(phrase_decompose)


  phrase_vectors=[]
  for i in range(len(candidates)):
    phrase_vectors.append(nlp(candidates[i]).vector)
  dict_vectors = []
  for i in range(len(phrase_in_dict)):
    dict_vectors.append(nlp(phrase_in_dict[i]).vector)
  phrase_vectors = np.array(phrase_vectors)
  dict_vectors = np.array(dict_vectors)

  distance_min = []
  for i in range(phrase_vectors.shape[0]):
    distances = []
    for j in range(dict_vectors.shape[0]):
      distances.append(np.linalg.norm(phrase_decompose[i] - dict_decompose[j]))
    distance_min.append(min(distances))

  distance_min = np.array(distance_min)
  index_distance_min = np.argsort(distance_min)
  index_distance_min = index_distance_min[::-1]


  for i in range(900,-1,-1):
    tmp = candidates[index_distance_min[i]]
    if any(x in ['hospital','county','island','province','state'] for x in tmp.split()):
      continue
    else:
      removed_candidates.remove(tmp)

  removed_candidates = removed_candidates + dict_core

  with open(candidate_path,'w',encoding='utf-8') as f:
    for line in removed_candidates:
      f.write(line+'\n')
  f.close()

if __name__ == '__main__':
  nlp=en_core_web_lg.load()
  dict_tokens, dict_core = generate_core_dictionary()
  phrases_tokens, phrases_list = generate_candidate_phrase()
  expansion(dict_tokens,phrases_tokens, dict_core, phrases_list)