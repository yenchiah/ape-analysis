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
from Util import Util
import re
from nltk.stem import WordNetLemmatizer
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

util = Util()

# This is for APE data
mt_all = []
pe_all = []
p_id = [] # photo id
a_id = [] # album id

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]

def main(argv):
    # This is for the text analysis of APE data
    #read_ape_data("GLAC/") # the Korean team
    #read_ape_data("AREL/") # the William one
    #compare_pos(mt_all, pe_all)
    #compare_ttr_by_story(mt_all, pe_all)
    #compare_ttr_all(mt_all, pe_all)

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
    analyze_smell_data()

def compare_ttr_all(mt_all, pe_all):
    d = {"ttr": []}
    d["ttr"].append(compute_ttr_all(mt_all))
    d["ttr"].append(compute_ttr_all(pe_all))
    df = pd.DataFrame(d)
    df.index = ["Pre-Edit", "Post-Edit"]
    df = df.round(2)
    print(df)
    df.to_csv("ttr_all.csv")

def compute_ttr_all(text_corpus):
    word_type = Counter()
    num_words = 0
    for story in text_corpus:
        if type(story[0]) == str: # this means there is only one story
            for sentence in story:
                sentence = sentence.replace("[male]", "Tom").replace("[female]", "Amy")
                tokens = word_tokenize(sentence)
                for word, pos in pos_tag(tokens, tagset="universal"):
                    if pos is not ".":
                        word_type[word.lower() + "#" + pos] += 1
                        num_words += 1
        elif type(story[0]) == list: # this means multiple workers submit multiple versions of the story
            ttr_story = []
            for worker_submit in story:
                for sentence in worker_submit:
                    sentence = sentence.replace("[male]", "Tom").replace("[female]", "Amy")
                    tokens = word_tokenize(sentence)
                    for word, pos in pos_tag(tokens, tagset="universal"):
                        if pos is not ".":
                            word_type[word.lower() + "#" + pos] += 1
                            num_words += 1
    return len(word_type) / num_words

def compare_ttr_by_story(mt_all, pe_all):
    d = {"mean_ttr": [], "std_ttr": [], "mean_num_words": [], "std_num_words": []}
    ttr_mt, num_words_mt = compute_ttr_by_story(mt_all)
    d["mean_ttr"].append(np.mean(ttr_mt))
    d["std_ttr"].append(np.std(ttr_mt))
    d["mean_num_words"].append(np.mean(num_words_mt))
    d["std_num_words"].append(np.std(num_words_mt))
    ttr_pe, num_words_pe = compute_ttr_by_story(pe_all)
    d["mean_ttr"].append(np.mean(ttr_pe))
    d["std_ttr"].append(np.std(ttr_pe))
    d["mean_num_words"].append(np.mean(num_words_pe))
    d["std_num_words"].append(np.std(num_words_pe))
    df = pd.DataFrame(d)
    df.index = ["Pre-Edit", "Post-Edit"]
    df = df.round(2)
    print(df)
    df.to_csv("ttr_story.csv")

    plot_kde(ttr_mt, ttr_pe, "ttr_story.png")

    print("n=", len(ttr_mt), len(ttr_pe))
    s_ttr, p_ttr = stats.ttest_rel(ttr_mt, ttr_pe)
    s_num_words, p_num_words = stats.ttest_rel(num_words_mt, num_words_pe)
    df_ttest = pd.DataFrame({"test_statistic": [s_ttr, s_num_words], "p_value": [p_ttr, p_num_words], "N": [len(ttr_mt), len(num_words_mt)]})
    df_ttest = df_ttest.round(6)
    df_ttest.index = ["ttr", "num_words"]
    print(df_ttest)
    df_ttest.to_csv("ttr_paired_ttest_two_tail.csv")

def plot_kde(a, b, file_name):
    sns.set(rc={"figure.figsize":(12,1.5)})
    plot = sns.kdeplot(a, shade=True, linewidth=2, color="#4575b4", legend=True, label="Automatic")
    #plot = sns.distplot(a,kde=False,rug=False,hist=True,kde_kws={"linewidth":1},norm_hist=True,bins=np.linspace(0,1,81))
    plot = sns.kdeplot(b, shade=True, linewidth=4, color="#d73027", legend=True, label="Human-Edited")
    #plot = sns.distplot(b,kde=False,rug=False,hist=True,kde_kws={"linewidth":1},norm_hist=True,bins=np.linspace(0,1,81))
    axes = plot.axes
    axes.legend(loc="upper left", frameon=False, ncol=1, fontsize=18, borderaxespad=0, borderpad=0)
    #axes.grid(False)
    axes.set_facecolor((1, 1, 1))
    axes.axhline(linewidth=3, color="#000000")
    #axes.axvline(linewidth=1, color="#000000")
    axes.tick_params(axis="both", pad=0, labelsize=18)
    axes.set_xlim([0,1])
    #axes.set_ylim([0,1])
    #plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(file_name, dpi=150)

# Token-type ratio
def compute_ttr_by_story(text_corpus):
    ttr = []
    num_words = []
    for story in text_corpus:
        word_type_story = Counter()
        num_words_story = 0
        if type(story[0]) == str: # this means there is only one story
            for sentence in story:
                sentence = sentence.replace("[male]", "Tom").replace("[female]", "Amy")
                tokens = word_tokenize(sentence)
                for word, pos in pos_tag(tokens, tagset="universal"):
                    if pos is not ".":
                        word_type = word.lower() + "#" + pos # this is a type
                        word_type_story[word_type] += 1
                        num_words_story += 1
            ttr.append(len(word_type_story) / num_words_story)
            num_words.append(num_words_story)
        elif type(story[0]) == list: # this means multiple workers submit multiple versions of the story
            ttr_story = []
            num_words_story = []
            for worker_submit in story:
                word_type_worker = Counter()
                num_words_worker = 0
                for sentence in worker_submit:
                    sentence = sentence.replace("[male]", "Tom").replace("[female]", "Amy")
                    tokens = word_tokenize(sentence)
                    for word, pos in pos_tag(tokens, tagset="universal"):
                        if pos is not ".":
                            word_type = word.lower() + "#" + pos # this is a type
                            word_type_worker[word_type] += 1
                            num_words_worker += 1
                ttr_story.append(len(word_type_worker) / num_words_worker)
                num_words_story.append(num_words_worker)
            ttr.append(np.mean(ttr_story))
            num_words.append(np.mean(num_words_story))
    return (ttr, num_words)

def compare_pos(mt_all, pe_all):
    df_mt, c_story_list_mt = compute_pos_table(mt_all)
    df_mt.index = ["Pre-Edit"]
    df_pe, c_story_list_pe = compute_pos_table(pe_all)
    df_pe.index = ["Post-Edit"]
    df_diff = df_pe - df_mt
    df = pd.concat([df_mt, df_pe])
    df.loc["Diff"] = df.loc["Post-Edit"] - df.loc["Pre-Edit"]
    df = df.round(1)
    print(df)
    df.to_csv("pos.csv")

    df_noun = {"p_id": [], "a_id": [], "c_mt_noun": [], "c_pe_noun": [], "mt": [], "pe": []}
    index = 0
    for c_mt, c_pe in zip(c_story_list_mt, c_story_list_pe):
        df_noun["p_id"].append(p_id[index])
        df_noun["a_id"].append(a_id[index])
        df_noun["c_mt_noun"].append(c_mt["NOUN"])
        df_noun["c_pe_noun"].append(c_pe["NOUN"])
        df_noun["mt"].append(mt_all[index])
        df_noun["pe"].append(pe_all[index])
        index += 1
    df_noun = pd.DataFrame(df_noun)
    df_noun.to_csv("df_noun.csv")

def compute_pos_table(text_corpus):
    num_story = 0
    c = Counter()
    c_story_list = []
    for story in text_corpus:
        if type(story[0]) == str: # this means there is only one story
            pos_story = []
            for sentence in story:
                sentence = sentence.replace("[male]", "Tom").replace("[female]", "Amy")
                tokens = word_tokenize(sentence)
                pos_story += [p[1] for p in pos_tag(tokens, tagset="universal")]
            cc = Counter(pos_story)
            c += cc
            c_story_list.append(cc)
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
            cc = Counter(d)
            c += cc
            c_story_list.append(cc)
            num_story += 1
    for k in c:
        c[k] /= num_story
    df = pd.DataFrame(c, index=[0])
    if "X" in df.columns:
        df = df.drop(["X"], axis=1)
    df = df.reindex(sorted(df.columns), axis=1) # sort by column
    df["Total"] = df.sum(axis=1)
    return (df, c_story_list)

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
        tick_font_size=18, title_font_size=18, h_size=5, w_size=5, wspace=1, hspace=0.2, rotate=True)

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
    p_id.append(data["photo_sequence_id"])
    a_id.append(data["album_id"])
    pe = []
    for d in data["edited_stories"]:
        pe.append(d["normalized_edited_story_text_sent"])
    pe_all.append(pe)

if __name__ == "__main__":
    main(sys.argv)
