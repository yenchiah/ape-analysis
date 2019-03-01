import sys
import json
from nltk import word_tokenize, ngrams, FreqDist, pos_tag
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import TrigramCollocationFinder
from nltk.collocations import QuadgramCollocationFinder
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import itertools
import numpy as np
from util.python3.Util import Util
import re
from nltk.stem import WordNetLemmatizer
from collections import Counter

util = Util()

# This is for APE data
mt_all = []
pe_all = []

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]

def main(argv):
    # This is for the text analysis of APE data
    read_ape_data("GLAC/") # the Korean team
    #read_ape_data("AREL/") # the William one
    compare_pos(mt_all, pe_all)

    # BUG: the followings does not work since we changed the structure of pe_all
    # pe_all looks like [[[]]], mt_all looks like [[]]
    #compute_sentence_length(mt_all)
    #compute_sentence_length(pe_all)
    #compute_ngrams_tf_idf(mt_all, "mt_ngram_count.png", tp="count")
    #compute_ngrams_tf_idf(pe_all, "pe_ngram_count.png", tp="count")
    #compute_ngrams_tf_idf(mt_all, "mt_ngram_tf_idf.png", tp="tf-idf")
    #compute_ngrams_tf_idf(pe_all, "pe_ngram_tf_idf.png", tp="tf-idf")
    #compute_ngrams_tf_idf(mt_all, "mt_ngram_tf.png", tp="tf")
    #compute_ngrams_tf_idf(pe_all, "pe_ngram_tf.png", tp="tf")
    #compute_ngrams_tf_idf(mt_all, "mt_ngram_df.png", tp="df")
    #compute_ngrams_tf_idf(pe_all, "pe_ngram_df.png", tp="df")

    # This is for the text analysis of Smell Pittsburgh Data
    #analyze_smell_data()

def compare_pos(mt_all, pe_all):
    df_mt = compute_pos_table(mt_all)
    df_mt.index = ["Pre-Edit"]
    df_pe = compute_pos_table(pe_all)
    df_pe.index = ["Post-Edit"]
    df_diff = df_pe - df_mt
    df = pd.concat([df_mt, df_pe])
    df.loc["Diff"] = df.loc["Post-Edit"] - df.loc["Pre-Edit"]
    df = df.round(2)
    print(df)
    df.to_csv("pos.csv")

def compute_pos_table(text_corpus):
    num_story = 0
    c = Counter()
    for story in text_corpus:
        if type(story[0]) == str: # this means there is only one story
            pos_story = []
            for sentence in story:
                sentence = sentence.replace("[male]", "Tom").replace("[female]", "Amy")
                tokens = word_tokenize(sentence)
                pos_story += [p[1] for p in pos_tag(tokens, tagset="universal")]
            c += Counter(pos_story)
            num_story += 1
        elif type(story[0]) == list: # this means multiple workers submit multiple versions of the story
            num_workers = 0
            d = Counter()
            for worker_submit in story:
                pos_worker_submit = []
                for sentence in worker_submit:
                    sentence = sentence.replace("[male]", "Tom").replace("[female]", "Amy")
                    tokens = word_tokenize(sentence)
                    pos_worker_submit += [p[1] for p in pos_tag(tokens, tagset="universal")]
                d += Counter(pos_worker_submit)
                num_workers += 1
            for k in d:
                d[k] /= num_workers
            c += Counter(d)
            num_story += 1
    for k in c:
        c[k] /= num_story
    df = pd.DataFrame(c, index=[0])
    if "X" in df.columns:
        df = df.drop(["X"], axis=1)
    df = df.reindex(sorted(df.columns), axis=1) # sort by column
    df["Total"] = df.sum(axis=1)
    return df

def analyze_smell_data():
    print("Read smell data...")
    df_smell = pd.read_csv("data/smell-reports.csv")
    smell_description = [list(df_smell["smell_description"].dropna())]
    feelings_symptoms = [list(df_smell["feelings_symptoms"].dropna())]
    n = 10
    lv = 0
    tp = "count-smell"
    out_p = "smell.png"
    data = []
    data.append(compute_tf_idf(smell_description, n=n, n_gram=1, tp=tp, lv=lv))
    data.append(compute_tf_idf(smell_description, n=n, n_gram=2, tp=tp, lv=lv))
    data.append(compute_tf_idf(feelings_symptoms, n=n, n_gram=1, tp=tp, lv=lv))
    data.append(compute_tf_idf(feelings_symptoms, n=n, n_gram=2, tp=tp, lv=lv))

    # Plot
    x = []
    y = []
    title = ["Description (unigram)", "Description (bigram)", "Symptom (unigram)", "Symptom (bigram)"]
    for d in data:
        x.append(d["f"][::-1])
        y.append(d["v"][::-1])
    util.plot_bar_chart_grid(x, y, 2, 2, title, out_p,
        tick_font_size=18, title_font_size=18, h_size=5, w_size=5, wspace=0.8, hspace=0.2, rotate=True)

def read_ape_data(path):
    print("Read ape data...")
    read_file(dirname=path+"dev", endswith=[".json"])
    read_file(dirname=path+"train", endswith=[".json"])
    read_file(dirname=path+"test", endswith=[".json"])

def compute_ngrams_tf_idf(text_corpus, out_p, n=20, tp="tf-idf", g=3, lv=1, title_prefix=""):
    print("Compute ngrams: " + tp)
    data = []
    for i in range(g):
        data.append(compute_tf_idf(text_corpus, n=n, n_gram=i+1, tp=tp, lv=lv))
    
    # Plot
    x = []
    y = []
    for d in data:
        x.append(d["f"][::-1])
        y.append(d["v"][::-1])
    title = [title_prefix + str(i+1) + "-gram" for i in range(g)]
    util.plot_bar_chart_grid(x, y, 1, len(data), title, out_p,
        tick_font_size=16, title_font_size=18, h_size=6, w_size=5, wspace=0.7, rotate=True)

def replace(string, substitutions):
    substrings = sorted(substitutions, key=len, reverse=True)
    regex = re.compile('|'.join(map(re.escape, substrings)))
    return regex.sub(lambda match: substitutions[match.group(0)], string)

# defines a custom vectorizer class
class CustomTfidfVectorizer(TfidfVectorizer): 
    # overwrite the build_analyzer method, allowing one to create a custom analyzer for the vectorizer
    def build_analyzer(self):
        # load stop words using CountVectorizer's built in method
        stop_words = self.get_stop_words()
        # create the analyzer that will be returned by this method
        def analyser(doc):
            # Analyze each sentence of the input string seperately
            ngrams = []
            for s in re.split('[?.,!;-]', doc):
                tokens = word_tokenize(s)
                # use CountVectorizer's _word_ngrams built in method to remove stop words and extract n-grams
                ngrams += self._word_ngrams(tokens, stop_words)
            return ngrams
        return analyser

# defines a custom vectorizer class
class CustomCountVectorizer(CountVectorizer):
    # overwrite the build_analyzer method, allowing one to create a custom analyzer for the vectorizer
    def build_analyzer(self):
        # load stop words using CountVectorizer's built in method
        stop_words = self.get_stop_words()
        # create the analyzer that will be returned by this method
        def analyser(doc):
            # Analyze each sentence of the input string seperately
            ngrams = []
            for s in re.split('[?.,!;-]', doc):
                tokens = word_tokenize(s)
                # use CountVectorizer's _word_ngrams built in method to remove stop words and extract n-grams
                ngrams += self._word_ngrams(tokens, stop_words)
            return ngrams
        return analyser

# defines a custom vectorizer class
class SmellCountVectorizer(CountVectorizer):
    # overwrite the build_analyzer method, allowing one to create a custom analyzer for the vectorizer
    def build_analyzer(self):
        # load stop words using CountVectorizer's built in method
        stop_words = self.get_stop_words()
        # create the analyzer that will be returned by this method
        def analyser(doc):
            # Analyze each sentence of the input string seperately
            self.wnl = WordNetLemmatizer()
            ngrams = []
            doc = doc.lower()
            doc = replace(doc, {"breathe": "breath", "sulphur": "sulfur", "woodsmoke": "wood smoke", "none": "", "outside": ""})
            for s in re.split('[?.,!;-]', doc):
                tokens = word_tokenize(s)
                if self.wnl is not None:
                    tokens = [self.wnl.lemmatize(t) for t in tokens]
                # use CountVectorizer's _word_ngrams built in method to remove stop words and extract n-grams
                ngrams += self._word_ngrams(tokens, stop_words)
            return ngrams
        return analyser

def compute_tf_idf(text_corpus, n=20, n_gram=1, tp="tf-idf", lv=1):
    if n_gram == 1:
        sw = stopwords.words("english") + ["[", "]"]
    else:
        sw = ["[", "]"]

    if tp == "tf":
        # Compute term frequency
        vectorizer = CustomTfidfVectorizer(stop_words=sw, ngram_range=(n_gram,n_gram), use_idf=False)
    elif tp == "df":
        # Compute document frequency
        vectorizer = CustomTfidfVectorizer(stop_words=sw, ngram_range=(n_gram,n_gram), use_idf=True, smooth_idf=False)
    elif tp == "count":
        vectorizer = CustomCountVectorizer(stop_words=sw, ngram_range=(n_gram,n_gram))
    elif tp == "count-smell":
        vectorizer = SmellCountVectorizer(stop_words=sw, ngram_range=(n_gram,n_gram))
    else:
        # Compute tf-idf
        vectorizer = CustomTfidfVectorizer(stop_words=sw, ngram_range=(n_gram,n_gram))

    if lv == 0:
        # Sentence level
        v = vectorizer.fit_transform(util.flatten_one_level(text_corpus))
    elif lv == 1:
        # Story level
        v = vectorizer.fit_transform(reduce_dim(text_corpus))
    else:
        # All text
        v = vectorizer.fit_transform([" ".join(reduce_dim(text_corpus))])

    if tp == "df":
        v = 1 / np.exp(vectorizer.idf_)
    elif tp == "count" or tp == "count-smell":
        v = np.asarray(np.sum(v, axis=0)).squeeze()
    else:
        v = np.asarray(np.mean(v, axis=0)).squeeze()

    f = np.array(vectorizer.get_feature_names())
    idx = np.argsort(v)[::-1][:n]
    return {"f": f[idx], "v": v[idx]}

def compute_ngrams_count(text_corpus, out_p, n=20):
    print("Compute ngrams count...")
    list_of_tokens = []
    for document in text_corpus:
        for sentence in document:
            list_of_tokens.append(word_tokenize(sentence))

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
    for i in range(len(data)):
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
    util.plot_bar_chart_grid(x, y, 1, len(data), title, out_p,
        tick_font_size=14, title_font_size=14, h_size=8, w_size=5, rotate=True)

def compute_sentence_length(text_corpus):
    # Compute the distribution of sentence length
    L_all = []
    for document in text_corpus:
        for sentence in document:
            L_all.append(len(word_tokenize(sentence)))
    print(pd.DataFrame(data=L_all).describe())

# Reduce one dimension of a nested list with strings
def reduce_dim(nested_list):
    return [" ".join(k) for k in nested_list]

@util.loop_files
def read_file(**kwargs):
    data = json.load(kwargs["file_obj"])
    mt = data["auto_story_text_normalized"].strip()
    mt = [i.strip() + " ." for i in mt.split(".")]
    mt = mt[:-1]
    mt_all.append(mt)
    pe = []
    for d in data["edited_stories"]:
        pe.append(d["normalized_edited_story_text_sent"])
        pe_all.append(pe)

if __name__ == "__main__":
    main(sys.argv)
