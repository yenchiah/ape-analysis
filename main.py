import sys
import json
from nltk import word_tokenize, ngrams, FreqDist
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import TrigramCollocationFinder
from nltk.collocations import QuadgramCollocationFinder
from nltk.corpus import stopwords
import pandas as pd
import itertools
from util.python3.Util import Util

util = Util()
mt_all = []
pe_all = []

def main(argv):
    read_data()
    #compute_sentence_length()
    compute_ngrams(mt_all, "mt_ngram.png")
    compute_ngrams(pe_all, "pe_ngram.png")

def read_data():
    read_file(dirname="data/dev", endswith=[".json"])
    read_file(dirname="data/train", endswith=[".json"])
    read_file(dirname="data/test", endswith=[".json"])

def compute_ngrams(list_of_sentences, out_p):
    list_of_tokens = []
    n = 20
    for s in list_of_sentences:
        list_of_tokens.append(word_tokenize(s))

    # Unigram
    tokens = util.flatten_one_level(list_of_tokens)
    custom_sw = [".", "[", "]", ","]
    sw = stopwords.words("english") + custom_sw
    tokens = [w for w in tokens if w not in sw]
    word_fd = FreqDist(tokens)
    uni_mc = word_fd.most_common(n)

    # Bigram
    bi = BigramCollocationFinder.from_documents(list_of_tokens)
    #bi.apply_freq_filter(2)
    #print(bi.ngram_fd.items())
    bi_mc = bi.ngram_fd.most_common(n)

    # Trigram
    tri = TrigramCollocationFinder.from_documents(list_of_tokens)
    tri_mc = tri.ngram_fd.most_common(n)

    # Quadgram
    quad = QuadgramCollocationFinder.from_documents(list_of_tokens)
    quad_mc = quad.ngram_fd.most_common(n)

    # Plot
    data = [uni_mc, bi_mc, tri_mc, quad_mc]
    x = []
    y = []
    for i in range(4):
        x_ng = []
        y_ng = []
        for d in data[i]:
            if i==0: 
                x_ng.append(d[0])
            else:
                x_ng.append(" ".join(d[0]))
            y_ng.append(d[1])
        x.append(x_ng[::-1])
        y.append(y_ng[::-1])
    title = ["Unigram", "Bigram", "Trigram", "Quadgram"]
    util.plot_bar_chart_grid(x, y, 1, 4, title, out_p,
        tick_font_size=16, title_font_size=18, h_size=7, w_size=4, rotate=True)

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
