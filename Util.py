"""
Usage:
    from Util import Util
    util = Util()
"""

import os
import itertools
import matplotlib
matplotlib.use("Agg") # generate images without having a window appear
import matplotlib.pyplot as plt

class Util(object):

    # Use this function to initialize parameters
    def __init__(self):
        pass

    """
    Loop files in a folder
    (for file name that ends with strings in the "endswith" list)
    (this is a decorator function)
    Usage:
        read_file(dirname="data/dev", endswith=[".json"])
        @util.loop_files
        def read_file(**kwargs):
            data = json.load(kwargs["file_obj"])
    """
    def loop_files(self, fn):
        def wrap(dirname=None, endswith=[]):
            for file in os.listdir(dirname):
                fname = os.fsdecode(file)
                skip = False
                if len(endswith) > 0:
                    skip = True
                    for s in endswith:
                        if fname.endswith(s):
                            skip = False
                            break
                if skip: continue
                with open(os.path.join(dirname, fname)) as f:
                    fn(file_obj=f)
        return wrap

    """
    Flatten one level of nesting list
    Usage:
        util.flatten_one_level([[1,2,3],[4,5,6]])
    """
    def flatten_one_level(self, list_of_lists):
        return itertools.chain.from_iterable(list_of_lists)

    """
    Plot a grid of bar charts
    (x and y are one-level nesting lists)
    (title is a list)
    """
    def plot_bar_chart_grid(self, x, y, h, w, title, sup_title, out_p, sup_title_font_size=16, sup_top=0.8,
            h_size=1.5, w_size=12, tick_font_size=10, title_font_size=14, hspace=0, wspace=0, rotate=False):
        fig = plt.figure(figsize=(w*w_size, h*h_size))
        c = 1
        for i in range(0, h*w):
            ax = plt.subplot(h, w, i+1)
            ax.margins(0, 0)
            plt.title(title[i], fontsize=title_font_size)
            if rotate:
                plt.barh(range(0,len(x[i])), y[i], 0.6, color=(0.4,0.4,0.4), align="center")
                plt.yticks(range(0,len(x[i])), x[i], fontsize=tick_font_size)
            else:
                plt.bar(range(0,len(x[i])), y[i], 0.6, color=(0.4,0.4,0.4), align="center")
                plt.xticks(range(0,len(x[i])), x[i], fontsize=tick_font_size)
            #for j in range(0, len(y[i])):
            #   ax.text(j, y[i][j], int(y[i][j]), color=(0.2,0.2,0.2), ha="center", fontsize=10)

        plt.suptitle(sup_title, fontsize=sup_title_font_size)
        plt.tight_layout()
        plt.subplots_adjust(hspace=hspace, wspace=wspace, top=sup_top)
        fig.savefig(out_p, dpi=150, transparent=True)
        fig.clf()
        plt.close()
