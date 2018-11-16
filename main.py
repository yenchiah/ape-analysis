import sys
import json
from nltk import word_tokenize, ngrams
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import TrigramCollocationFinder
from nltk.collocations import QuadgramCollocationFinder
import pandas as pd
from util.python3.Util import Util

util = Util()
mt_all = []
pe_all = []

def main(argv):
    read_data()
    #compute_sentence_length()
    compute_ngrams()

def read_data():
    read_file(dirname="data/dev", endswith=[".json"])
    #read_file(dirname="data/train", endswith=[".json"])
    #read_file(dirname="data/test", endswith=[".json"])

def compute_ngrams():
    mt_tokens = []
    for s in mt_all:
        mt_tokens.append(word_tokenize(s))

    # Bigram
    bi = BigramCollocationFinder.from_documents(mt_tokens)
    #bi.apply_freq_filter(2)
    #print(bi.ngram_fd.items())
    bi_mc = bi.ngram_fd.most_common(20)
    print(bi_mc)

    # Trigram
    tri = TrigramCollocationFinder.from_documents(mt_tokens)
    tri_mc = tri.ngram_fd.most_common(20)
    print(tri_mc)

    # Quadgram
    quad = QuadgramCollocationFinder.from_documents(mt_tokens)
    quad_mc = quad.ngram_fd.most_common(20)
    print(quad_mc)

def compute_sentence_length():
    # Compute the distribution of sentence length
    L_mt_all = []
    L_pe_all = []
    for s in mt_all:
        L_mt_all.append(len(word_tokenize(s)))
    for s in pe_all:
        L_pe_all.append(len(word_tokenize(s)))
    print("MT sentence length:")
    print(pd.DataFrame(data=L_mt_all).describe())
    print("PE sentence length:")
    print(pd.DataFrame(data=L_pe_all).describe())

@util.loop_files
def read_file(**kwargs):
    data = json.load(kwargs["file_obj"])
    mt = data["auto_story_text_normalized"].strip()
    mt = [i.strip() + " ." for i in mt.split(".")]
    mt = mt[:-1]
    pe = []
    for d in data["edited_stories"]:
        pe += d["normalized_edited_story_text_sent"]
    # Save to global
    for s in mt:
        mt_all.append(s)
    for s in pe:
        pe_all.append(s)

if __name__ == "__main__":
    main(sys.argv)
