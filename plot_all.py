import pickle
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import column, row
from bokeh.models.widgets import TextInput
import os

import numpy as np


def open_history(direc ,filename):
    with open(direc +"/{}".format(filename), 'rb') as handle:
        return pickle.load(handle)

def create_plot_line(title,y):
    p = figure(title=title , plot_width=350, plot_height=350)
    p.line(np.arange(np.size(y)),y, line_width=2)
    return p

def show_hists(hists, direc):
    for filename in hists:
        hist = open_history(direc, filename)
        output_file(filename+".html")
        train_plots = []
        val_plots = []
        for key in hist.keys():
            if key[0:3] == 'val':
                val_plots.append(create_plot_line(key,np.nan_to_num(hist[key])))
            else:
                train_plots.append(create_plot_line(key,np.nan_to_num(hist[key])))

        show(column(TextInput(value=filename), row(train_plots), row(val_plots)))
    # handle.close()


params = 'l=2_r=0.048'
direc = 'checkpoints/'+params
files = os.listdir(direc)
show_hists(files, direc)