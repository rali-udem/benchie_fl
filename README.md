# BenchIE^FL: A Manually Re-Annotated Fact-Based Open Information Extraction Benchmark
BenchIE^FL is an Open Information Extraction (OIE) reference, Measuring performance of OIE systems based on a set of manual (gold) annotations. It is based on the previously released BenchIE benchmark (https://github.com/gkiril/benchie) but improves on the quality of the annotations and on the matching function used in scoring systems, resulting in a fairer evaluation.

More details about the annotations, guidelines and implementation can be found in our paper [BenchIE^FL: A Manually Re-Annotated Fact-Based Open Information Extraction Benchmark](https://openreview.net/forum?id=xW_rRnxqIlg) .

## Table of contents

  - [Requirements](#requirements)
  - [Running the benchmark](#running-the-benchmark)
  - [Data formats](#data-formats)
    - [Annotations](#annotations)
    - [OIE system extractions](#oie-system-extractions)
  - [Guidelines](#guidelines)
  - [License](#license)
  - [Citing](#citing)

## Requirements

To run BenchIE^FL, you need python 3+ and both numpy and pandas. To install tested versions of the required modules, run

```console
pip install -r requirements.txt
```
## Running the benchmark

`bench.py` contains the code needed to run the benchmark on the provided formatted extractions. To run on it differents extractions, change the following lines from the file :

```python
# --> Change to use extractions from different systems
# Reads extractions by systems
reverb = read_formatted('extractions/benchie/reverb_benchie_form.txt')
clausie = read_formatted('extractions/benchie/clausie_benchie_form.txt')
minie = read_formatted('extractions/benchie/minie_benchie_form.txt')
imojie = read_formatted('extractions/benchie/imojie_benchie_form.txt')
openie6 = read_formatted('extractions/benchie/openie6_benchie_form.txt')
m2oie = read_formatted('extractions/benchie/m2oie_benchie_form.txt')
compactie = read_formatted('extractions/benchie/compactie_benchie_form.txt')

# --> Change names accordingly
# Names and extractions of the systems evaluated need to be in same order as the extractions (systems_extractions)
systems = ['reverb', 'clausie', 'minie', 'imojie', 'openie6', 'm2oie', 'compactie']
systems_extractions = [reverb, clausie, minie, imojie, openie6, m2oie, compactie]

# --> Change annotations file if needed
# Reads gold annotations
sents, gold_all, gold_simple, gold_total = read_benchie('gold/benchie-annotated(300).txt')
gold_total = process_total(gold_total)
```
to different (BenchIE formatted) extractions and annotation files. The BenchIE format is described next.

## Data formats

The data in BenchIE^FL follows the exact same data format as BenchIE. We directly use their explaination of the data formats, which can be found on their github page:

The data in BenchIE^FL (and BenchIE) is split in two majour groups:

   * Gold annotations: the gold annotations provided by manual annotations.
   * OIE systems extractions: the extractions generated by OIE systems.

In what follows, we explain these major data categories (i.e., their format) as well as how to handle them with the BenchIE^FL framework.

### Gold annotations

The gold annotations are writen in `gold/benchie-annotated(300).txt`. Each entry is written in the following format:

```
sent_id:1	He served as the first Prime Minister of Australia and became a founding justice of the High Court of Australia .
1--> Cluster 1:
He --> served as --> [the] Prime Minister
He --> served as [a] --> Prime Minister
He --> served as --> [a] Prime Minister
1--> Cluster 2:
He --> served as [the] first Prime Minister of --> Australia
He --> served as --> [the] first Prime Minister of Australia
1--> Cluster 3:
Australia --> has [had] --> [a] Prime Minister
Australia --> has [had] [a] --> Prime Minister
Australia --> had --> [a] Prime Minister
Australia --> had [a] --> Prime Minister
1--> Cluster 4:
He --> became --> [a] founding justice
He --> became [a] --> founding justice
1--> Cluster 5:
He --> became [a] founding justice of --> [the] High Court of Australia
1--> Cluster 6:
Australia --> has [had] --> [a] High Court
Australia --> has [had] [a] --> High Court
Australia --> had --> [a] High Court
Australia --> had [a] --> High Court
1--> Cluster 7:
[the] High Court of Australia --> has [had] --> [a] founding justice
[the] High Court of Australia --> has [had] [a] --> founding justice
[the] High Court of Australia --> had --> [a] founding justice
[the] High Court of Australia --> had [a] --> founding justice

sent_id:2	Graner handcuffed him to the bars of a cell window and left him there , feet dangling off the floor , for nearly five hours .
2--> Cluster 1:
Graner --> handcuffed --> him
2--> Cluster 2:
Graner --> handcuffed him to --> [the] bars
...
```
The first line is the input sentence, where there are two tab-separated entries:
   * `sent_id:` which is a placeholder where the sentence ID is written (in principle, this can be any string, though in our implementation we use integers)
   * sentence: the input sentence as a string
  
Next, each cluster (i.e., synset of facts) needs to be specified in a separate line in the following format:

```1--> Cluster 1:```

where the first string (in our case "1") has to match the sentence ID from the first line (`sent_id`). This needs to be followed by the string ```"--> Cluster "```, which is followed by  the cluster number (here, 1), followed by ```":"``` and a new line. 

Next, each triple is written between two consequtive clusters. For instance, the following two triples are in cluster 1, because they are written between clusters 1 and 2:
```
1--> Cluster 1:
He --> served as --> [the] Prime Minister
He --> served as [a] --> Prime Minister
He --> served as --> [a] Prime Minister
1--> Cluster 2:
...
```
The triples are written in the format ```subject --> relation --> object```. The tokens in square brackets (```[``` and ```]```) are optional tokens. This means that when an OIE triple is evaluated for correctness, the extraction is considered to be correct even if some of the optional tokens are missing. All possible triples (with all possible combinations of optional tokens) are automatically generated in the function `gen_optional_clusters()`.


### OIE system extractions

The OIE systems' extractions are written in the folder `/extractions`. Each file represents the extractions of one OIE system for the respective reference (benchie, wire, webassertions). Each line in the file is one OIE extraction and is written in the following tab-separated format:
```
sent_id <TAB> subject <TAB> relation <TAB> object
```
Note that `sent_id` should match `sent_id` from the golden extractions. For example, consider the following line:
``` 
19 <TAB> She <TAB> began <TAB> her film career
```
This means that this particular OIE triple will be evaluated w.r.t. the sentence with `sent_id=19` in the golden annotations file. 

## Guidelines

Guidelines presented in the paper are distributed for both annotation of sentences and matching of extraction-annotation pairs in the `/guidelines` folder.

## License

The software framework is licensed according to the license for academic or non-profit organization noncommercial research use only. Details are provided in the [license file](https://github.com/rali-udem/benchie_fl/blob/main/LICENSE.txt) and in the header of each source code file. The data is under the non-restrictive [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

## Citing
If you use BenchIE for your research, please cite the following paper:

```
Bibtex
```





