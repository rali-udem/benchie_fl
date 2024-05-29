#imports
from utils import *

# Reads extractions by systems
reverb = read_formatted('extractions/benchie/reverb_benchie_form.txt')
clausie = read_formatted('extractions/benchie/clausie_benchie_form.txt')
minie = read_formatted('extractions/benchie/minie_benchie_form.txt')
imojie = read_formatted('extractions/benchie/imojie_benchie_form.txt')
openie6 = read_formatted('extractions/benchie/openie6_benchie_form.txt')
m2oie = read_formatted('extractions/benchie/m2oie_benchie_form.txt')
compactie = read_formatted('extractions/benchie/compactie_benchie_form.txt')

# Names and extractions of the systems evaluated need to be in same order as the extractions (systems_extractions)
systems = ['reverb', 'clausie', 'minie', 'imojie', 'openie6', 'm2oie', 'compactie']
systems_extractions = [reverb, clausie, minie, imojie, openie6, m2oie, compactie]

# Reads gold annotations
sents, gold_all, gold_simple, gold_total = read_benchie('gold/benchie-annotated(300).txt')
gold_total = process_total(gold_total)


"""
CREATE EXTRACTIONS DF
"""
print('Creating Extraction DataFrame')

df, systems_extractions = create_ext_df(systems_extractions, systems)

"""
CREATE MATCHING DF
"""
print('Creating Matching DataFrame')

df_f, results_match_f = create_match_df(df, systems_extractions, systems, sents, gold_all, gold_simple, gold_total)

#Computing system scores
print('Computing System Scores')
print()
print('System, precision, recall, F1')
for i, sys in enumerate(systems):
  results_match_f[sys]['custom_match'] = get_results_match(df_f, sys, 'custom_match_nopunc_corr', systems_extractions[i])

for sys in systems:
  prec = compute_prec(results_match_f[sys]['custom_match'])
  rec = compute_rec(gold_all, results_match_f[sys]['custom_match'])
  f1 = compute_f1(prec,rec)
  print(sys, prec, rec, f1)

