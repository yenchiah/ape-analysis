import pandas as pd
from nltk import word_tokenize, pos_tag
from ast import literal_eval

def only_noun(story_str):
    story = literal_eval(story_str)
    filtered = []
    if type(story[0]) == str: # this means there is only one story
        for sentence in story:
            tokens = word_tokenize(sentence)
            for word, pos in pos_tag(tokens, tagset="universal"):
                if pos == "NOUN":
                    filtered.append(word)
    elif type(story[0]) == list: # this means multiple workers submit multiple versions of the story
        for worker_submit in story:
            f = []
            for sentence in worker_submit:
                tokens = word_tokenize(sentence)
                for word, pos in pos_tag(tokens, tagset="universal"):
                    if pos == "NOUN":
                        f.append(word)
            filtered.append(f)
    return filtered

df_noun_glac = pd.read_csv("df_noun_glac.csv", index_col=0)
df_noun_arel = pd.read_csv("df_noun_arel.csv", index_col=0)

df_noun_glac.set_index("p_id", inplace=True)
df_noun_arel.set_index("p_id", inplace=True)

desired_id = []
for index, row in df_noun_glac.iterrows():
    if row["c_pe_noun"] - row["c_mt_noun"] > 3:
        desired_id.append(index)

df_noun_glac_selected = df_noun_glac.loc[desired_id]
df_noun_arel_selected = df_noun_arel.loc[desired_id]
df_noun_arel_selected = df_noun_arel_selected.dropna()

print("=========================================================================================")
df = {"p_id": [], "a_id": [], "c_mt_noun_glac": [], "c_pe_noun_glac": [], "mt_glac": [], "mt_glac_noun": [], "pe_glac": [], "pe_glac_noun": [], "c_mt_noun_arel": [], "c_pe_noun_arel": [], "mt_arel": [], "mt_arel_noun": [], "pe_arel": [], "pe_arel_noun": []}
for index, row in df_noun_arel_selected.iterrows():
    if row["c_mt_noun"] - row["c_pe_noun"] > 2:
        s_glac = df_noun_glac_selected.loc[index]
        s_arel = df_noun_arel_selected.loc[index]
        print("photo id: ", index)
        print("album id: ", s_glac["a_id"])
        df["p_id"].append(index)
        df["a_id"].append(s_glac["a_id"])
        print("----------------------------------------------------------------------------------------")
        print("glac")
        print(s_glac["c_mt_noun"], s_glac["c_pe_noun"])
        df["c_mt_noun_glac"].append(s_glac["c_mt_noun"])
        df["c_pe_noun_glac"].append(s_glac["c_pe_noun"])
        print("\n")
        print(s_glac["mt"])
        df["mt_glac"].append(literal_eval(s_glac["mt"]))
        mt_glac_noun = only_noun(s_glac["mt"])
        df["mt_glac_noun"].append(mt_glac_noun)
        print("\n")
        print(mt_glac_noun)
        print("\n")
        print(s_glac["pe"])
        df["pe_glac"].append(literal_eval(s_glac["pe"]))
        pe_glac_noun = only_noun(s_glac["pe"])
        df["pe_glac_noun"].append(pe_glac_noun)
        print("\n")
        print(pe_glac_noun)
        print("----------------------------------------------------------------------------------------")
        print("arel")
        print(s_arel["c_mt_noun"], s_arel["c_pe_noun"])
        df["c_mt_noun_arel"].append(s_arel["c_mt_noun"])
        df["c_pe_noun_arel"].append(s_arel["c_pe_noun"])
        print("\n")
        print(s_arel["mt"])
        df["mt_arel"].append(literal_eval(s_arel["mt"]))
        mt_arel_noun = only_noun(s_arel["mt"])
        df["mt_arel_noun"].append(mt_arel_noun)
        print("\n")
        print(mt_arel_noun)
        print("\n")
        print(s_arel["pe"])
        df["pe_arel"].append(literal_eval(s_arel["pe"]))
        pe_arel_noun = only_noun(s_arel["pe"])
        df["pe_arel_noun"].append(pe_arel_noun)
        print("\n")
        print(pe_arel_noun)
        print("=========================================================================================")

df = pd.DataFrame(df)
df = df.rename(index=str, columns={"p_id": "photo_sequence_id", "a_id": "album_id"})
df.to_json("noun_analysis.json", orient="records")
