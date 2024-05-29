import numpy as np
import pandas as pd
import itertools
import json
from tqdm import tqdm
import string
import re

def compute_prec(matches_array):
# Computes precision from array of matches for a given system
  count = 0
  for m in matches_array:
    if m != 0:
      count += 1
  return count/len(matches_array)

def compute_rec(gold, matches_array):
# Computes precision from array of matches for a given system and gold annotations
  count_gold = 0
  count_matches = 0
  for sent in gold:
    count_gold += len(sent)
  for m in matches_array:
    if m != 0:
      if str(m)[-1] == '0':
        count_matches += 1
      else:
        count_matches += 2
  return count_matches/count_gold

def compute_f1(prec, rec):
# Computes F1 from precision and recall
  if prec == 0 and rec == 0:
    return 0
  return 2*prec*rec/(prec+rec)

def gen_optional_clusters(split):
  """Generates all possible combinations of a cluster (optional words)
  in : ['[optional] first argument|relation|second argument', 'first argument 2nd formulation|relation|second argument', ...]
  out : list of all possible combinations
  [[[formulation 1 combinaison 1]
   [formulation 1 combinaison 2]
   ...]
   [[formulation 2 combinaison 1]]
   [formulation 2 combinaison 2]
    ...]]
    
  """

  out = []

  for elem in split:

    elem_split = elem.split('|')

    opening = elem.count('[')
    closing = elem.count(']')

    # number of closing and opening brackets don't match
    if opening != closing:
      print('ERROR : ', elem)
      return -1

    lst = [i for i in range(opening)]
    combs = [] # all possible combinations of optional words, binary

    for i in range(1, len(lst)+1):
      els = [list(x) for x in itertools.combinations(lst, i)]
      combs.extend(els)
    combs.append([])

    # generates formulations
    for i in range(len(combs)):

      count = 0
      formulation = []

      for e in elem_split:

        el = []
        for word in e.split(' '):
          if word != '':
            if '[' not in word and ']' not in word:
              el.append(word)
            else:
              if count in combs[i]:
                count += 1
              else:
                el.append(word.replace('[', '').replace(']', ''))
                count += 1

        formulation.append(el)
      out.append(formulation)

  return out

def remove_last(string, char):
# Removes the last char from a string if it is the char passed as argument
    if string == "":
      return "XXX"
    if string[-1] == char:
      return string[:-1]
    else:
      return string

def cluster_match_benchie(df):
  # Creates a column for the df that indicates wether the current cluster cluster is
  # already exactly matched (BenchIE match) by another extraction from the same system

  prev_sys = df.iloc[0, :]['system']
  prev_idx = df.iloc[0, :]['index']
  prev_ext =df.iloc[0, :]['extraction']
  cluster_length = -1
  cluster_match_benchie = []
  count = 0
  extraction_length = 0
  prev_c_idx = float('inf')

  out = []

  for i in range(len(df)):
    sys = df.iloc[i, :]['system']
    idx = df.iloc[i, :]['index']
    c_idx = df.iloc[i, :]['cluster_index']
    match_ = df.iloc[i, :]['match_benchie']
    ext = df.iloc[i, :]['extraction']

    if prev_sys != sys or prev_idx != idx:
      cluster_length = prev_c_idx + 1
      for j in range(extraction_length):
        for k in range(cluster_length):
          out.append(1 if k in cluster_match_benchie else 0) 
          count += 1

      extraction_length = 0
      cluster_match_benchie = []
      prev_c_idx = float('inf')

    if c_idx < prev_c_idx or prev_ext != ext:
      extraction_length += 1

    if match_ != 0:
      cluster_match_benchie.append(c_idx)

    prev_sys = sys
    prev_idx = idx
    prev_c_idx = c_idx
    prev_ext = ext    

  cluster_length = prev_c_idx + 1
  for j in range(extraction_length):
    for k in range(cluster_length):
      out.append(1 if k in cluster_match_benchie else 0)

  #TODO : Bug when last sentence annotated only has one cluster???
  if len(out) != len(df):
    out.append(0)

  return out

def custom_match(df_elem):
  if df_elem['match_benchie'] == 1 or df_elem['alternate_match_benchie'] == 1:
    return 1
  elif df_elem['cluster_match_flat'] == 1 and df_elem['rel_match'] == 1 and df_elem['extraction_match_benchie'] == 0 and ((df_elem['arg1_match'] == 1 and df_elem['arg2_ref_match'] == 1) or (df_elem['arg2_match'] == 1 and df_elem['arg1_ref_match'] == 1)) and df_elem['cluster_match_benchie'] == 0:
    return 1
  
  else:
    return 0

def custom_match_nopunc(df_elem):
  if df_elem['match_benchie'] == 1 or df_elem['alternate_match_benchie'] == 1:
    return 1
  elif df_elem['cluster_match_flat'] == 1 and df_elem['rel_match'] == 1 and df_elem['extraction_match_benchie'] == 0 and ((df_elem['arg1_match'] == 1 and df_elem['arg2_ref_match'] == 1) or (df_elem['arg2_match'] == 1 and df_elem['arg1_ref_match'] == 1)) and df_elem['cluster_match_benchie'] == 0:
    return 1
  elif df_elem['match_benchie_nopunc'] == 1 or df_elem['alternate_match_benchie'] == 1:
    return 1
  elif df_elem['cluster_match_flat_nopunc'] == 1 and df_elem['rel_match_nopunc'] == 1 and df_elem['extraction_match_benchie'] == 0 and ((df_elem['arg1_match_nopunc'] == 1 and df_elem['arg2_ref_match_nopunc'] == 1) or (df_elem['arg2_match_nopunc'] == 1 and df_elem['arg1_ref_match_nopunc'] == 1)) and df_elem['cluster_match_benchie'] == 0:
    return 1
  
  else:
    return 0

def gen_pairs(gold):
# Generates all pairs of alternate possibilities (is case and xy case) for given annotations for a sentence

  is_pairs = []
  # (X - is - Y) case
  for elem in gold:
    for e in elem.split(' X '):
      if e.split('|')[1] == "is":
        for e1 in gen_optional_clusters([e.split('|')[0]]):
          for e2 in gen_optional_clusters([e.split('|')[2]]):
            is_pairs.append([' '.join(e1[0]), ' '.join(e2[0])])

  # (X and Y - rel - Z) case
  gold_flat = []
  gold_flat_index = []
  for i, elem in enumerate(gold):
    for e in gen_optional_clusters(elem.split(' X ')):
      gold_flat.append(e)
      gold_flat_index.append(i)

  xy_pairs = []
  for i in range(len(gold_flat)):
    for j in range(len(gold_flat)):
      if i == j:
        continue
      if gold_flat[i][0] == gold_flat[j][0] and gold_flat[i][1] == gold_flat[j][1] and gold_flat_index[i] != gold_flat_index[j] and [gold_flat[i][2], gold_flat[j][2]] not in xy_pairs and [gold_flat[j][2], gold_flat[i][2]] not in xy_pairs:
        xy_pairs.append([' '.join(gold_flat[i][2]), ' '.join(gold_flat[j][2])])
        if len(xy_pairs) > 1000: # Fix for annotations with too many possibilities (exponential), reduces them to 1000 max, only takes the first ones -> could miss some matches
          return is_pairs, xy_pairs

      if gold_flat[i][2] == gold_flat[j][2] and gold_flat[i][1] == gold_flat[j][1] and gold_flat_index[i] != gold_flat_index[j] and [gold_flat[i][0], gold_flat[j][0]] not in xy_pairs and [gold_flat[j][0], gold_flat[i][0]] not in xy_pairs:
        xy_pairs.append([' '.join(gold_flat[i][0]), ' '.join(gold_flat[j][0])])
        if len(xy_pairs) > 1000: # Fix for annotations with too many possibilities (exponential), reduces them to 1000 max, only takes the first ones -> could miss some matches
          return is_pairs, xy_pairs

  return is_pairs, xy_pairs

def gen_alternate_ext(ext, is_pairs, xy_pairs):
# Generates the alternate formulations from the alternate possibilites (is and xy)

  if len(ext.split(' - ')) == 2:
    ext = ext + ' - XXX'
  
  out = []
  pairs = []

  ext_arg1 = ext.split(' - ')[0]
  ext_arg2 = ext.split(' - ')[2]

  for pair in is_pairs:
    pair_matched = True
    for elem in pair:
      if elem not in ext_arg1:
        pair_matched = False
    if pair_matched:
      for elem in pair:
        adding = re.sub(' +', ' ', ext_arg1.replace(elem, '') + ' - ' + ext.split(' - ')[1] + ' - ' + ext.split(' - ')[2]).strip()
        out.append(adding)
        #if check_if_skip_word(adding):
          #out.append(' '.join([w for w in adding.split(' ') if w not in skip_words]))
  
  # case for is pairs
  for pair in is_pairs:
    pair_matched = True
    for elem in pair:
      if elem not in ext_arg2:
        pair_matched = False
    if pair_matched:
      for elem in pair:
        adding = re.sub(' +', ' ', ext.split(' - ')[0] + ' - ' + ext.split(' - ')[1] + ' - ' + ext_arg2.replace(elem, '')).strip()
        out.append(adding)
        #if check_if_skip_word(adding):
          #out.append(' '.join([w for w in adding.split(' ') if w not in skip_words]))

  # case where the xy pair is in the first argument
  for pair in xy_pairs:
    pair_matched = True
    for elem in pair:
      if elem not in ext_arg1:
        pair_matched = False
    if pair_matched:
      for elem in pair:
        adding = re.sub(' +', ' ', ext_arg1.replace(elem, '').replace('and', '').replace(',', '') + ' - ' + ext.split(' - ')[1] + ' - ' + ext.split(' - ')[2]).strip()
        if adding not in out:
          out.append(adding)
        adding = re.sub(' +', ' ', ext_arg1.replace(elem, '').replace(',', '') + ' - ' + ext.split(' - ')[1] + ' - ' + ext.split(' - ')[2]).strip()
        if adding not in out:
          out.append(adding)
        adding = re.sub(' +', ' ', ext_arg1.replace(elem, '').replace('and', '') + ' - ' + ext.split(' - ')[1] + ' - ' + ext.split(' - ')[2]).strip()
        if adding not in out:
          out.append(adding)
        adding = re.sub(' +', ' ', ext_arg1.replace(elem, '') + ' - ' + ext.split(' - ')[1] + ' - ' + ext.split(' - ')[2]).strip()
        if adding not in out:
          out.append(adding)

  # case where the xy pair is in the second argument
  for pair in xy_pairs:
    pair_matched = True
    for elem in pair:
      if elem not in ext_arg2:
        pair_matched = False
    if pair_matched:
      for elem in pair:
        adding = re.sub(' +', ' ', ext.split(' - ')[0] + ' - ' + ext.split(' - ')[1] + ' - ' + ext_arg2.replace(elem, '').replace('and', '').replace(',', '')).strip()
        if adding not in out:
          out.append(adding)
        adding = re.sub(' +', ' ', ext.split(' - ')[0] + ' - ' + ext.split(' - ')[1] + ' - ' + ext_arg2.replace(elem, '').replace(',', '')).strip()
        if adding not in out:
          out.append(adding)
        adding = re.sub(' +', ' ', ext.split(' - ')[0] + ' - ' + ext.split(' - ')[1] + ' - ' + ext_arg2.replace(elem, '').replace('and', '')).strip()
        if adding not in out:
          out.append(adding)
        adding = re.sub(' +', ' ', ext.split(' - ')[0] + ' - ' + ext.split(' - ')[1] + ' - ' + ext_arg2.replace(elem, '')).strip()
        if adding not in out:
          out.append(adding)

  return out

def check_if_skip_word(s):
# Boolean wether string contains a skip word
  for w in s.split(' '):
    if w in skip_words:
      return True
  return False

def benchie_match(extraction, gold_cluster):
# Implementation of the exact match used in BenchIE (strict : slot-separated)

  extraction_split = extraction.split(' - ')
  if len(extraction_split) == 2:
    extraction_split.append('XXX')
    extraction_split[1] = extraction_split[1][:-2]
  if len(extraction_split) <= 1:
    return 0
  
  match_ = False
  skip = False

  gold_split = gold_cluster.split(' X ')
  gold_split = gen_optional_clusters(gold_split)

  for form in gold_split:
    match_ = False
    skip = False

    for i in range(len(form)):
      if len(extraction_split) != len(form):
        match_ = False
        skip = True
      elif extraction_split[i].split(' ') == form[i]:
        match_ = True
      else:
        match_ = False
        skip = True

    if match_ and not skip:
      return 1
  return 0

def benchie_match_nopunc(extraction, gold_cluster):
# Exact match witch punctuation removed

  extraction_split = extraction.split(' - ')
  if len(extraction_split) == 2:
    extraction_split.append('XXX')
    extraction_split[1] = extraction_split[1][:-2]
  if len(extraction_split) == 1:
    return 0
  
  match_ = False
  skip = False

  gold_split = gold_cluster.split(' X ')
  gold_split = gen_optional_clusters(gold_split)

  if len(gold_split) > 100000:
    gold_split = gold_split[:5000]

  for form in gold_split:
    match_ = False
    skip = False

    for i in range(len(form)):
      if len(extraction_split) != len(form):
        match_ = False
        skip = True
      elif extraction_split[i].translate(str.maketrans('', '', string.punctuation)).strip().split(' ') == ' '.join(form[i]).translate(str.maketrans('', '', string.punctuation)).strip().split(' '):
        match_ = True
      else:
        match_ = False
        skip = True

    if match_ and not skip:
      return 1
  return 0

def read_formatted(raw_path):
# Reads tabbed output from systems : sent_id  ext_arg1  rel  arg2  arg3+ into list
  out = [[] for i in range(300)]
  for line in open(raw_path, "r"):
    line = line.split('\t')
    index = int(line[0]) - 1
    out[index].append([line[1], line[2], line[3][:-1]])
  return out

def read_benchie(filepath):
# Reads BenchIE formatted annotations : sent_id:id   sent\n id-->Cluster1:\n arg1-->rel-->arg2...
# Returns sentences (sents),
# extractions with all optional words (extractions_all),
# extractions with no optional words (extractions_simple)
# extractions with optional words still in brackets (extractions_total)
  def simple(liste):
    out = []
    for elem in liste:
      elem_out = []
      for word in elem.split(' '):
        if '[' in word or ']' in word:
          continue
        else:
          elem_out.append(word)
      out.append(' '.join(elem_out))
    return out

  def remove_char(l, x):
    out = []
    for elem in l:
      out.append(elem.replace(x, ''))
    return out

  f = open(filepath)

  sents = []
  extractions_all = []
  extractions_simple = []
  extractions_total = []

  sent_all = []
  sent_simple = []
  sent_total = []

  first = True
  for line in f:
    if line[:8] == 'sent_id:':
      sents.append(line.split('\t')[1][:-1])
      cluster_simple = []
      cluster_all = []
      cluster_total = []
    elif 'Cluster' in line and line[-2:-1] == ':':
      if not first:
        sent_all.append(cluster_all)
        sent_simple.append(cluster_simple)
        sent_total.append(cluster_total)

      first = False
      cluster_all = []
      cluster_simple = []
      cluster_total = []
    
    elif line == '\n':
      first = True
      sent_all.append(cluster_all)
      sent_simple.append(cluster_simple)
      sent_total.append(cluster_total)

      extractions_all.append(sent_all)
      extractions_simple.append(sent_simple)
      extractions_total.append(sent_total)

      sent_all = []
      sent_simple = []
      sent_total = []
    
    else:
      cluster_all.append([''.join(remove_char(arg, '\n')) for arg in line.split(' --> ')])
      cluster_simple.append(simple([''.join(remove_char(arg, '\n')) for arg in line.split(' --> ')]))
      cluster_total.append([''.join(remove_char(arg, '\n')) for arg in line.split(' --> ')])

  first = True
  sent_all.append(cluster_all)
  sent_simple.append(cluster_simple)
  sent_total.append(cluster_total)

  extractions_all.append(sent_all)
  extractions_simple.append(sent_simple)
  extractions_total.append(sent_total)

  sent_all = []
  sent_simple = []
  sent_total = []

  return sents, extractions_all, extractions_simple, extractions_total

def process_total(total):
# Processes extractions_total to format into list
  out = []
  for sent in total:
    out_ = []
    for cluster in sent:
      out__ = ""
      for elem in cluster:
        out__ += '|'.join(elem) + ' X '
      out_.append(out__[:-3])
    out.append(out_)

  return out


def correct_col(col, df):
  """
  Corrects a given matching column to:
  1 - Give priority to exact match : If a clusters matches custom and matches exact with two differents extractions, priority to exact match
  2 - Correct multi-match : If an extraction has a custom match with two differents clusters : priority to smallest index one

  *correct() actually changes the column, this function identifies the errors
  """

  prev_sys = df.iloc[0, :]['system']
  prev_idx = df.iloc[0, :]['index']
  prev_ext =df.iloc[0, :]['extraction']

  col_ = [[]]
  bm_ = [[]]
  wm_ = [[]]
  le_ = [[]]
  m_ = [[]]

  out = []

  for i in range(len(df)):
    sys = df.iloc[i, :]['system']
    idx = df.iloc[i, :]['index']
    c_idx = df.iloc[i, :]['cluster_index']
    match_ = df.iloc[i, :]['match_benchie']
    ext = df.iloc[i, :]['extraction']

    if prev_sys != sys or prev_idx != idx:
      col_ = correct(col_, bm_, wm_, le_)

      for elem in col_:
        for e in elem:
          out.append(e)

      col_ = [[]] #column to correct
      bm_ = [[]] #exact match
      wm_ = [[]] #numbers of matching words
      le_ = [[]] #length of annotation
      m_ = [[]]
    
    elif ext != prev_ext:
      col_.append([])
      bm_.append([])
      wm_.append([])
      le_.append([])
      m_.append([])

    bm_[-1].append(match_)
    col_[-1].append(df.iloc[i, :][col])
    if df.iloc[i, :][col] == 1:
      wm_[-1].append(df.iloc[i, :]['num_words_match_max'])
    else:
      wm_[-1].append(0)
    if df.iloc[i, :][col] == 1:
      le_[-1].append(df.iloc[i, :]['num_words_ref_max'])
    else:
      le_[-1].append(0)

    prev_sys = sys
    prev_idx = idx
    prev_c_idx = c_idx
    prev_ext = ext

  col_ = correct(col_, bm_, wm_, le_)

  for elem in col_:
    for e in elem:
      out.append(e)

  return out

def correct(col, bm, mw, le):
  # Makes the changes to the column from correct_col
  out = []
  m_ext = []
  m_clu = []

  for i in range(len(bm)):
    col_ = [0 for j in range(len(col[i]))]
    for j, elem in enumerate(bm[i]):
      if elem == 1 and i not in m_ext and j not in m_clu:
        col_[j] = 1
        m_clu.append(j)
    if np.sum(bm[i]) != 0:
      m_ext.append(i)
    out.append(col_)

  min_words = [[float('inf') for i in range(len(bm[0]))] for j in range(len(bm))]

  for i in range(len(col)):
    col_ = out[i]
    for j, elem in enumerate(le[i]):
      if elem != 0:
        if min_words[i][j] > elem and i not in m_ext and j not in m_clu:
          for l in range(len(out)):
            for k in range(len(out[l])):
              if k == j:
                out[l][k] = 0
          col_[j] = 1
          m_ext.append(i)
          min_words[i][j] = elem
     
    out[i] = col_
    
  return out

def get_results_match(df, sys, col, extractions):
# Computes for a given system, wich extractions have been matched
  temp_df = df[df['system'] == sys]
  out = [sub_df for _, sub_df in temp_df.groupby('extraction')]
  outt = [None for i in range(len(extractions))]
  count = 0
  for i in range(len(out)):
    ext = out[i].iloc[0,:]['extraction']
    index = extractions.index(ext)
    outt[index] = out[i]
  outtt = [float(0) for i in range(len(outt))]
  for i, elem in enumerate(outt):
    if elem is not None:
      matches = list(elem[col])
      for j, e in enumerate(matches):
        if e != 0:
          if outtt[i] != 0:
            outtt[i] = str(outtt[i])[:-1] + str(j+1)
          else:
            outtt[i] = float(j+1)

  return outtt

def create_ext_df(systems_extractions, systems):

  sys_extractions = {}
  sys_extractions_ = {}
  for i in range(len(systems_extractions)):
    sys_extractions[systems[i]] = systems_extractions[i]
    sys_extractions_[systems[i]] = []


  df = pd.DataFrame()


  # Formats systems extractions
  exts = []
  indexs = []
  syss = []
  for i in range(300):
    for sys in systems:
      ext = sys_extractions[sys][i]
      if ext != []:
        for e in ext:
          exts.append(' - '.join(e))
          indexs.append(i)
          syss.append(sys)

          sys_extractions_[sys].append(' - '.join(e))

  df['Unnamed: 0'] = indexs #sentence index
  df['extraction'] = exts #extractions
  df['system'] = syss #system

  for i in range(len(systems_extractions)):
    systems_extractions[i] = sys_extractions_[systems[i]]

  return df, systems_extractions


def create_match_df(df, systems_extractions, systems, sents, gold_all, gold_simple, gold_total):

  # Matching results
  results_match_f = {sys : {} for sys in systems}
  extractions = [[] for i in range(len(sents))]

  df_extraction = [] #extraction
  df_clusters = [] #cluster
  df_sent = [] #sentence
  df_gold = [] #gold annotations for sentence
  df_system = [] #extraction system
  df_index = [] #sentence index
  df_cluster_index = [] #cluster index

  df_match_benchie = [] #exact match (BenchIE)
  df_match_benchie_nopunc = [] #exact match without punctuation
  df_extraction_match_benchie = [] #boolean for extraction match
  df_alternate_match_benchie = [] #exact match with alternate formulation

  df_num_words = []

  df_num_words_ref_max = [] # maximum number of words for cluster
  df_num_words_match_max = [] # number of matched words

  df_cluster_match_flat = [] # extraction matches with another cluster when flattened (no slots)
  df_cluster_match_flat_nopunc = [] # cluster match without punctuation

  df_arg1_match = [] # first argument match
  df_rel_match = [] # relation match
  df_arg2_match = [] # second argument match

  df_arg1_match_nopunc = []  # first argument match without punctuation
  df_rel_match_nopunc = [] # relation match without punctuation
  df_arg2_match_nopunc = [] # second argument match without punctuation

  # indicated wether all words of given spot in the reference are included in the extraction
  df_arg1_ref_match = []
  df_rel_ref_match = []
  df_arg2_ref_match = []

  # ref match without punctuation
  df_arg1_ref_match_nopunc = []
  df_rel_ref_match_nopunc = []
  df_arg2_ref_match_nopunc = []

  prev_sys = ""
  prev_index = ""
  num_ext = 0

  df_max_word_match_cluster = []

  for i in tqdm(range(len(df))):
    index = df.iloc[i,:]['Unnamed: 0']
    extraction = df.iloc[i,:]['extraction']
    system = df.iloc[i,:]['system']
    sent_extractions = list(df[(df['system'] == system) & (df['Unnamed: 0'] == index)]['extraction'])

    if prev_sys == system and prev_index == index:
      num_ext += 1
    else:
      num_ext = 0

    if gold_total[index] == ['']:
      continue

    
    # Generates alternate pairs for given sent index
    if prev_index != index:
      #print(gold_total[index])
      is_pairs, xy_pairs = gen_pairs(gold_total[index])
   
    prev_sys = system
    prev_index = index

    cluster_match_flat = 0
    cluster_match_flat_nopunc = 0

    max_word_match_cluster = 0

    extraction_match_benchie = 0
      
    for j, form in enumerate(gold_total[index]):

      df_index.append(index)
      df_cluster_index.append(j)
      df_system.append(system)
      df_extraction.append(extraction)
      df_clusters.append(form)
      df_sent.append(sents[index])
      df_gold.append(gold_total[index])

      ext_flat = extraction.replace(' - ', ' ')
      ext_flat_nopunc = ext_flat.translate(str.maketrans('', '', string.punctuation))
      ext_split = extraction.split(' - ')

      #concatenates empty argument (XXX) to extractions shorter than 3
      if len(ext_split)<3:
        ext_split.append('XXX')

      ext_arg1 = ext_split[0]
      ext_arg1_split = ext_arg1.split(' ')
      ext_arg1_nopunc = ext_split[0].translate(str.maketrans('', '', string.punctuation))
      ext_arg1_split_nopunc = ext_arg1.translate(str.maketrans('', '', string.punctuation)).split(' ')

      ext_rel = ext_split[1]
      ext_rel_split = ext_rel.split(' ')
      ext_rel_nopunc = ext_split[1]
      ext_rel_split_nopunc = ext_rel.split(' ')

      ext_arg2 = ext_split[2]
      ext_arg2_split = ext_arg2.split(' ')
      ext_arg2_nopunc = ext_split[2].translate(str.maketrans('', '', string.punctuation))
      ext_arg2_split_nopunc = ext_arg2.translate(str.maketrans('', '', string.punctuation)).split(' ')

      ext_sep = ext_arg1 + ' SEP1 ' + ext_rel + ' SEP2 ' + ext_arg2

      num_words_ext = len(ext_flat.split(' '))

      count_form = 0
      max_num_words_ref = 0

      num_words_match_max = 0
      num_words_match_avg = 0

      arg1_match = 0
      rel_match = 0
      arg2_match = 0
      arg1_ref_match = 0
      rel_ref_match = 0
      arg2_ref_match = 0

      arg1_match_nopunc = 0
      rel_match_nopunc = 0
      arg2_match_nopunc = 0
      arg1_ref_match_nopunc = 0
      rel_ref_match_nopunc = 0
      arg2_ref_match_nopunc = 0

      for k, f in enumerate(gen_optional_clusters(form.split(' X '))):
        f = '|'.join([' '.join(elem) for elem in f])

        ref_flat = f.replace('[', '').replace(']','').replace('|', ' ')
        ref_arg1 = f.replace('[', '').replace(']','').split('|')[0]
        ref_rel = f.replace('[', '').replace(']','').split('|')[1]
        ref_arg2 = f.replace('[', '').replace(']','').split('|')[2]

        ref_flat_nopunc = f.replace('[', '').replace(']','').replace('|', ' ').translate(str.maketrans('', '', string.punctuation))
        ref_arg1_nopunc = f.replace('[', '').replace(']','').split('|')[0].translate(str.maketrans('', '', string.punctuation))
        ref_rel_nopunc = f.replace('[', '').replace(']','').split('|')[1].translate(str.maketrans('', '', string.punctuation))
        ref_arg2_nopunc = f.replace('[', '').replace(']','').split('|')[2].translate(str.maketrans('', '', string.punctuation))

        ref_sep =  ref_arg1 + ' SEP1 ' +  ref_rel + 'SEP2 ' +  ref_arg2

        if ref_arg1.replace(' ', '').lower() == ext_arg1.replace(' ', '').lower():
          arg1_match = 1
        if ref_rel.replace(' ', '').lower() == ext_rel.replace(' ', '').lower():
          rel_match = 1
        if ref_arg2.replace(' ', '').lower() == ext_arg2.replace(' ', '').lower():
          arg2_match = 1
        if ref_flat.replace(' ', '').lower() == ext_flat.replace(' ', '').lower():
          cluster_match_flat = 1

        if ref_arg1_nopunc.replace(' ', '').lower() == ext_arg1_nopunc.replace(' ', '').lower():
          arg1_match_nopunc = 1
        if ref_rel_nopunc.replace(' ', '').lower() == ext_rel_nopunc.replace(' ', '').lower():
          rel_match_nopunc = 1
        if ref_arg2_nopunc.replace(' ', '').lower() == ext_arg2_nopunc.replace(' ', '').lower():
          arg2_match_nopunc = 1
        if ref_flat_nopunc.replace(' ', '').lower() == ext_flat_nopunc.replace(' ', '').lower():
          cluster_match_flat_nopunc = 1

        in_ = 1
        for word in ref_arg1.split(' '):  
          if word not in ext_arg1_split:
            in_ = 0
        if in_ == 1:
          arg1_ref_match = 1

        in_ = 1
        for word in ref_rel.split(' '):  
          if word not in ext_rel_split:
            in_ = 0
        if in_ == 1:
          rel_ref_match = 1

        in_ = 1
        for word in ref_arg2.split(' '):  
          if word not in ext_arg2_split:
            in_ = 0
        if in_ == 1:
          arg2_ref_match = 1

        in_ = 1
        for word in ref_arg1_nopunc.split(' '):  
          if word not in ext_arg1_split_nopunc:
            in_ = 0
        if in_ == 1:
          arg1_ref_match_nopunc = 1
        in_ = 1
        for word in ref_rel_nopunc.split(' '):  
          if word not in ext_rel_split_nopunc:
            in_ = 0
        if in_ == 1:
          rel_ref_match_nopunc = 1
        in_ = 1
        for word in ref_arg2_nopunc.split(' '):  
          if word not in ext_arg2_split_nopunc:
            in_ = 0
        if in_ == 1:
          arg2_ref_match_nopunc = 1

        count_form += 1
        max_num_words_ref_ = len(ref_flat.split(' '))
        if max_num_words_ref_ > max_num_words_ref:
          max_num_words_ref = max_num_words_ref_


        num_words_match = 0

        ext_split_ = ext_flat.split(' ')
        ext_rel_split_ = ext_rel.split(' ')

        ref_split = ref_flat.split(' ')
        ref_arg1_split = ref_arg1.split(' ')
        ref_rel_split = ref_rel.split(' ')
        ref_arg2_split = ref_arg2.split(' ')

        for word_ext in ext_split_:
          if word_ext in ref_split:
            num_words_match += 1
        if num_words_match > num_words_match_max:
          num_words_match_max = num_words_match

      df_num_words_match_max.append(num_words_match_max)

      df_arg1_match.append(arg1_match)
      df_rel_match.append(rel_match)
      df_arg2_match.append(arg2_match)

      df_arg1_ref_match.append(arg1_ref_match)
      df_rel_ref_match.append(rel_ref_match)
      df_arg2_ref_match.append(arg2_ref_match)

      df_arg1_match_nopunc.append(arg1_match_nopunc)
      df_rel_match_nopunc.append(rel_match_nopunc)
      df_arg2_match_nopunc.append(arg2_match_nopunc)

      df_arg1_ref_match_nopunc.append(arg1_ref_match_nopunc)
      df_rel_ref_match_nopunc.append(rel_ref_match_nopunc)
      df_arg2_ref_match_nopunc.append(arg2_ref_match_nopunc)

      bm = benchie_match(extraction, form)
      bm_nopunc = benchie_match_nopunc(extraction, form)


      alternate_bm = 0
      for alternate_ext in gen_alternate_ext(extraction, is_pairs, xy_pairs):
        if benchie_match(alternate_ext, form) == 1:
          alternate_bm = 1

      df_match_benchie.append(bm)
      df_match_benchie_nopunc.append(bm_nopunc)
      df_alternate_match_benchie.append(alternate_bm)

      if bm == 1:
        extraction_match_benchie = 1

      if num_words_match_max > max_word_match_cluster:
        max_word_match_cluster = num_words_match_max

      if j == len(gold_total[index]) - 1:
        for l in range(len(gold_total[index])):
          df_max_word_match_cluster.append(max_word_match_cluster)
          df_cluster_match_flat.append(cluster_match_flat)
          df_cluster_match_flat_nopunc.append(cluster_match_flat_nopunc)
          df_extraction_match_benchie.append(extraction_match_benchie)

      df_num_words.append([num_words_ext])
      df_num_words_ref_max.append(max_num_words_ref)

  df_num_words = np.array(df_num_words)

  cluster_match = []
  for i in range(len(df_extraction)):
    if df_num_words[i,0] == df_max_word_match_cluster[i]:
      cluster_match.append(1)
    else:
      cluster_match.append(0)

  df_f = pd.DataFrame()

  df_f['system'] = df_system
  df_f['index'] = df_index
  df_f['cluster_index'] = df_cluster_index
  df_f['extraction'] = df_extraction
  df_f['cluster'] = df_clusters
  df_f['sent'] = df_sent
  df_f['gold'] = df_gold
  df_f['match_benchie'] = df_match_benchie
  df_f['match_benchie_nopunc'] = df_match_benchie_nopunc
  df_f['alternate_match_benchie'] = df_alternate_match_benchie

  df_f['num_words_ref_max'] = df_num_words_ref_max

  df_f['num_words_match_max'] = df_num_words_match_max

  df_f['cluster_match_flat'] = df_cluster_match_flat
  df_f['cluster_match_flat_nopunc'] = df_cluster_match_flat_nopunc

  df_f['arg1_match'] = df_arg1_match
  df_f['rel_match'] = df_rel_match
  df_f['arg2_match'] = df_arg2_match

  df_f['arg1_ref_match'] = df_arg1_ref_match
  df_f['rel_ref_match'] = df_rel_ref_match
  df_f['arg2_ref_match'] = df_arg2_ref_match

  df_f['arg1_match_nopunc'] = df_arg1_match_nopunc
  df_f['rel_match_nopunc'] = df_rel_match_nopunc
  df_f['arg2_match_nopunc'] = df_arg2_match_nopunc

  df_f['arg1_ref_match_nopunc'] = df_arg1_ref_match_nopunc
  df_f['rel_ref_match_nopunc'] = df_rel_ref_match_nopunc
  df_f['arg2_ref_match_nopunc'] = df_arg2_ref_match_nopunc

  df_f['extraction_match_benchie'] = df_extraction_match_benchie

  df_f['cluster_match_benchie'] = cluster_match_benchie(df_f)

  df_f['custom_match_nopunc'] = [custom_match_nopunc(df_f.iloc[i,:]) for i in range(len(df_f))]
  df_f['custom_match_nopunc_corr'] = correct_col('custom_match_nopunc', df_f)

  return df_f, results_match_f

