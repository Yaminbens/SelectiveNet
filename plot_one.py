import pickle
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import column, row
from bokeh.models.widgets import TextInput
from bokeh.palettes import Category10 as palette
from bokeh.models import Legend, LegendItem
import itertools

import os

import numpy as np


def open_history(direc ,filename):
    with open(direc +"/{}".format(filename), 'rb') as handle:
        return pickle.load(handle)

def show_hists(hists, direc, params, mean):
    keys = []
    keys.append('selective_head_coverage')
    keys.append('classification_head_acc')
    keys.append('classification_head_selective_acc')
    keys.append('val_selective_head_coverage')
    keys.append('val_classification_head_acc')
    keys.append('val_classification_head_selective_acc')

    for key in keys:
        ys = []
        xs = []
        items = []
        for filename in hists:
            hist = open_history(direc, filename)
            xs.append(np.arange(np.size(hist[key])))
            ys.append(np.nan_to_num(hist[key]))

        if mean!=1:
            for y in ys:
                for i in range(len(y)-1,mean-1,-1):
                    y[i] = np.mean(y[i-mean+1:i])

        colors = palette[len(hists)]

        output_file(key+'_'+params+"_line.html")
        p = figure(title=key+'_'+params, plot_width=600, plot_height=600)
        r = p.multi_line(xs=xs, ys=ys, line_color=colors, line_width=2)

        for i, filename in enumerate(hists):
            items.append(LegendItem(label=filename[31:36], renderers=[r], index=i))
        legend = Legend(items=items)
        p.add_layout(legend)

        show(column(p))

params = 'l=2_r=0.04_3103'
direc = 'checkpoints/'+params
files = os.listdir(direc)
show_hists(files, direc, params, mean=1)
