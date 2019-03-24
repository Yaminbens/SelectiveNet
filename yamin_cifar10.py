import pickle
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import column, row

import numpy as np


def open_history(filename):
    with open("checkpoints\{}.pkl".format(filename), 'rb') as handle:
        return pickle.load(handle)

def create_plot_line(title,y):
    p = figure(title=title , plot_width=350, plot_height=350)
    p.line(np.arange(np.size(y)),y, line_width=2)
    return p

hist = open_history('full_cov_2_1.0_history')
output_file("line.html")
train_plots = []
val_plots = []
for key in hist.keys():
    if key[0:3] == 'val':
        val_plots.append(create_plot_line(key,np.nan_to_num(hist[key])))
    else:
        train_plots.append(create_plot_line(key,np.nan_to_num(hist[key])))

show(column(row(train_plots), row(val_plots)))


