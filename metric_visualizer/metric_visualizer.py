# -*- coding: utf-8 -*-
# file: metric_visualizer.py
# time: 03/02/2022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
import os.path
import pickle
import random
import shlex
import subprocess
from collections import OrderedDict
from functools import wraps
import time

import matplotlib.colors
import numpy as np
import tikzplotlib
from findfile import find_cwd_files
from matplotlib import pyplot as plt
from scipy.stats import iqr
from scipy.stats import ranksums
from tabulate import tabulate

from metric_visualizer import __version__

retry_count = 100


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))


def exception_handle(f, disable=False):
    @wraps(f)
    def decorated(*args, **kwargs):
        if disable:
            return f(*args, **kwargs)
        else:
            try:
                return f(*args, **kwargs)
            except Exception as e:
                print('Exception <{}> found in function <{}>'.format(e, f))

    return decorated


class MetricVisualizer:
    COLORS_DICT = matplotlib.colors.XKCD_COLORS
    COLORS_DICT.update(matplotlib.colors.CSS4_COLORS)
    COLORS = list(COLORS_DICT.values())
    MARKERS = [".", "o", "+", "P",
               "x", "X", "D", "d",
               ]

    HATCHES = ['/', '\\', '|', '-', '+', 'x',
               'o', 'O', '.', '*']

    box_plot_tex_template = r"""
    \documentclass{article}
    \usepackage{pgfplots}
    \usepackage{tikz}
    \usepackage{caption}
    \usetikzlibrary{intersections}
    \usepackage{helvet}
    \usepackage[eulergreek]{sansmath}
    \usepackage{amsfonts,amssymb,amsmath,amsthm,amsopn}	% math related

    \begin{document}
        \pagestyle{empty}
        \pgfplotsset{ compat=1.3,every axis/.append style={
            grid = major,
            thick,
            font=\normalsize,
            xtick={$xtick$},
            xticklabels={$xticklabel$},
            ylabel = {$ylabel$},
            xlabel = {$xlabel$},
            x tick label style={rotate=0,anchor=north},
            y tick label style={rotate=0,anchor=east},
            xticklabel shift=$xtickshift$pt,
            yticklabel shift=$ytickshift$pt,
            xlabel shift=$xlabelshift$pt,
            ylabel shift=$ylabelshift$pt,
            line width = 1pt,
            tick style = {line width = 0.8pt}}}
        \pgfplotsset{every plot/.append style={thin}}


        \begin{figure}
        \centering

        $tikz_code$

        \end{figure}

    \end{document}
    """

    bar_plot_tex_template = r"""
    \documentclass{article}
    \usepackage{pgfplots}
    \usepackage{tikz}
    \usepackage{caption}
    \usetikzlibrary{intersections}
    \usepackage{helvet}
    \usepackage[eulergreek]{sansmath}
    \usepackage{amsfonts,amssymb,amsmath,amsthm,amsopn}	% math related

    \begin{document}
        \pagestyle{empty}
        \pgfplotsset{ compat=1.3,every axis/.append style={
            grid = major,
            thick,
            font=\normalsize,
            xtick={$xtick$},
            xticklabels={$xticklabel$},
            ylabel = {$ylabel$},
            xlabel = {$xlabel$},
            x tick label style={rotate=0,anchor=north},
            y tick label style={rotate=0,anchor=east},
            xticklabel shift=$xtickshift$pt,
            yticklabel shift=$ytickshift$pt,
            xlabel shift=$xlabelshift$pt,
            ylabel shift=$ylabelshift$pt,
            line width = 1pt,
            tick style = {line width = 0.8pt}}}
        \pgfplotsset{every plot/.append style={thin}}


        \begin{figure}
        \centering
		\usetikzlibrary{patterns}

        $tikz_code$

        \end{figure}

    \end{document}
    """

    violin_plot_tex_template = r"""
    \documentclass{article}
    \usepackage{pgfplots}
    \usepackage{tikz}
    \usepackage{caption}
    \usetikzlibrary{intersections}
    \usepackage{helvet}
    \usepackage[eulergreek]{sansmath}
    \usepackage{amsfonts,amssymb,amsmath,amsthm,amsopn}	% math related

    \begin{document}
        \pagestyle{empty}
        \pgfplotsset{ compat=1.3,every axis/.append style={
            grid = major,
            thick,
            font=\normalsize,
            xticklabels={$xticklabel$},
            xtick={$xtick$},
            ylabel = {$ylabel$},
            xlabel = {$xlabel$},
            x tick label style={rotate=0,anchor=north},
            y tick label style={rotate=0,anchor=east},
            xticklabel shift=$xtickshift$pt,
            yticklabel shift=$ytickshift$pt,
            xlabel shift=$xlabelshift$pt,
            ylabel shift=$ylabelshift$pt,
            line width = 1pt,
            tick style = {line width = 0.8pt}}}
        \pgfplotsset{every plot/.append style={thin}}


        \begin{figure}
        \centering

        $tikz_code$

        \end{figure}

    \end{document}
    """

    traj_plot_tex_template = r"""
    \documentclass{article}
    \usepackage{pgfplots}
    \usepackage{tikz}
    \usetikzlibrary{intersections}
    \usepackage{helvet}
    \usepackage[eulergreek]{sansmath}

    \begin{document}
        \pagestyle{empty}
        \pgfplotsset{every axis/.append style={
            grid = major,
            thick,
            font=\normalsize,
            xtick={$xtick$},
            xticklabels={$xticklabel$},
            xlabel = {$xlabel$},
            ylabel = {$ylabel$},
            x tick label style={rotate=0,anchor=north},
            y tick label style={rotate=0,anchor=east},
            xticklabel shift=$xtickshift$pt,
            yticklabel shift=$ytickshift$pt,
            xlabel shift=$xlabelshift$pt,
            ylabel shift=$ylabelshift$pt,
            line width = 1pt,
            tick style = {line width = 0.8pt}}
        }
        \pgfplotsset{every plot/.append style={very thin}}


        \begin{figure}
        \centering

        $tikz_code$

        \end{figure}

    \end{document}
"""

    def set_box_plot_tex_template(self, box_plot_tex_template):
        self.box_plot_tex_template = box_plot_tex_template

    def set_bar_plot_tex_template(self, violin_plot_tex_template):
        self.violin_plot_tex_template = violin_plot_tex_template

    def set_violin_plot_tex_template(self, violin_plot_tex_template):
        self.violin_plot_tex_template = violin_plot_tex_template

    def set_traj_plot_tex_template(self, traj_plot_tex_template):
        self.traj_plot_tex_template = traj_plot_tex_template

    def __init__(self, name=None, trial_tag='', trial_tag_list=None, metric_dict=None):
        """
        Used for plotting, e.g.,
            'Metric1': {
                'trial-0': [77.06, 2.52, 35.14, 3.04, 77.29, 3.8, 57.4, 38.52, 60.36, 22.45],
                'train-1': [64.53, 58.33, 97.89, 68.12, 88.6, 60.33, 70.99, 75.91, 42.49, 15.03],
                'train-2': [97.74, 86.05, 41.34, 81.66, 75.08, 1.76, 94.63, 27.26, 47.11, 42.06],
            },
            'Metric2': {
                'trial-0': [111.5, 105.61, 179.08, 167.25, 181.85, 152.75, 194.82, 130.86, 108.51, 151.44] ,
                'train-1': [187.58, 106.35, 134.22, 167.68, 188.24, 196.54, 154.21, 193.71, 183.34, 150.18],
                'train-2': [159.24, 148.44, 119.49, 160.24, 169.6, 133.27, 129.36, 180.36, 165.24, 152.38],
            }

        :param metric_dict: If you want to plot figure, it is recommended to add multiple trial experiments. In these trial, the experimental results
        are comparative, e.g., just minor different in these experiments.
        """

        if not trial_tag:
            self.trial_tag = 'Trial'
        else:
            self.trial_tag = trial_tag

        if not trial_tag_list:
            self.trial_tag_list = []
        else:
            self.trial_tag_list = [str(t) for t in trial_tag_list]

        self.trial_rank_test_result = {}
        self.metric_rank_test_result = {}

        self.version = __version__
        self.name = name
        if metric_dict is None:

            self.metrics = OrderedDict(
                {
                    # 'Metric1': {
                    #     'trial-0': [80.41, 79.78, 81.03, 80.09, 79.62, 80.56, 80.88, 79.94, 79.47, 79.78, 80.72, 79.78, 81.35, 80.88, 81.03],
                    #     'train-1': [80.41, 79.78, 81.03, 80.09, 79.62, 80.56, 80.88, 79.94, 79.47, 79.78, 80.72, 79.78, 81.35, 80.88, 81.03],
                    #     'train-2': [80.41, 79.78, 81.03, 80.09, 79.62, 80.56, 80.88, 79.94, 79.47, 79.78, 80.72, 79.78, 81.35, 80.88, 81.03],
                    # },
                    # 'Metric2': {
                    #     'trial-0': [76.79, 75.49, 77.92, 77.21, 75.63, 76.96, 77.44, 76.26, 76.35, 76.12, 76.12, 76.78, 75.64, 77.31, 73.79],
                    #     'train-1': [76.79, 75.49, 77.92, 77.21, 75.63, 76.96, 77.44, 76.26, 76.35, 76.12, 76.12, 76.78, 75.64, 77.31, 73.79],
                    #     'train-2': [76.79, 75.49, 77.92, 77.21, 75.63, 76.96, 77.44, 76.26, 76.35, 76.12, 76.12, 76.78, 75.64, 77.31, 73.79],
                    # }
                })
        else:
            self.metrics = metric_dict

        self.trial_id = 0
        self.dump_pointer = None

    def next_trial(self):
        self.trial_id += 1
        self.dump()

    def add_metric(self, metric_name='Accuracy', value=0):
        if metric_name in self.metrics:
            if 'trial{}'.format(self.trial_id) not in self.metrics[metric_name]:
                self.metrics[metric_name]['trial{}'.format(self.trial_id)] = [value]
            else:
                self.metrics[metric_name]['trial{}'.format(self.trial_id)].append(value)
        else:
            self.metrics[metric_name] = {'trial{}'.format(self.trial_id): [value]}

    def traj_plot_by_metric(self, save_path=None, **kwargs):
        plot_metrics = self.transpose()
        self.traj_plot(plot_metrics, save_path, **kwargs)

    def box_plot_by_metric(self, save_path=None, **kwargs):
        plot_metrics = self.transpose()
        self.box_plot(plot_metrics, save_path, **kwargs)

    def avg_bar_plot_by_metric(self, save_path=None, **kwargs):
        plot_metrics = self.transpose()
        self.avg_bar_plot(plot_metrics, save_path, **kwargs)

    def sum_bar_plot_by_metric(self, save_path=None, **kwargs):
        plot_metrics = self.transpose()
        self.sum_bar_plot(plot_metrics, save_path, **kwargs)

    def violin_plot_by_metric(self, save_path=None, **kwargs):
        plot_metrics = self.transpose()
        self.violin_plot(plot_metrics, save_path, **kwargs)

    def traj_plot_by_trial(self, save_path=None, **kwargs):
        plot_metrics = self.metrics
        self.traj_plot(plot_metrics, save_path, **kwargs)

    def box_plot_by_trial(self, save_path=None, **kwargs):
        plot_metrics = self.metrics
        self.box_plot(plot_metrics, save_path, **kwargs)

    def avg_bar_plot_by_trial(self, save_path=None, **kwargs):
        plot_metrics = self.metrics
        self.avg_bar_plot(plot_metrics, save_path, **kwargs)

    def sum_bar_plot_by_trial(self, save_path=None, **kwargs):
        plot_metrics = self.metrics
        self.sum_bar_plot(plot_metrics, save_path, **kwargs)

    def violin_plot_by_trial(self, save_path=None, **kwargs):
        plot_metrics = self.metrics
        self.violin_plot(plot_metrics, save_path, **kwargs)

    @exception_handle
    def traj_plot(self, plot_metrics=None, save_path=None, **kwargs):

        markers = self.MARKERS[:]
        colors = self.COLORS[:]
        hatches = self.HATCHES[:]

        if isinstance(plot_metrics, str):  # warning for early version (<0.4.0)
            print('Please do not use this function directly for version (<0.4.0)')
        if not plot_metrics:
            plot_metrics = self.metrics

        alpha = kwargs.pop('alpha', 0.1)

        legend_loc = kwargs.pop('legend_loc', 2)

        markersize = kwargs.pop('markersize', 3)

        xlabel = kwargs.pop('xlabel', None)

        xticks = kwargs.pop('xticks', None)

        ylabel = kwargs.pop('ylabel', None)

        yticks = kwargs.pop('yticks', None)

        linewidth = kwargs.pop('linewidth', 0.5)

        hatches = kwargs.pop('hatches', hatches)

        xrotation = kwargs.pop('xrotation', 0)

        yrotation = kwargs.pop('yrotation', 0)

        xlabelshift = kwargs.pop('xlabelshift', 1)

        ylabelshift = kwargs.pop('ylabelshift', 1)

        xtickshift = kwargs.pop('xtickshift', 1)

        ytickshift = kwargs.pop('ytickshift', 1)

        minorticks_on = kwargs.pop('minorticks_on', False)

        traj_parts = []
        legend_labels = []
        trial_tag_list = kwargs.get('trial_tag_list ', self.trial_tag_list)
        for metric_name in plot_metrics.keys():
            metrics = plot_metrics[metric_name]
            if not trial_tag_list or len(trial_tag_list) != len(metrics.keys()):
                if self.trial_tag_list:
                    print('Unequal length of trial_tag_list and trial_num:', trial_tag_list, '<->', list(metrics.keys()))
                trial_tag_list = list(metrics.keys())
            else:
                trial_tag_list = trial_tag_list
            ax = plt.subplot()

            y = np.array([metrics[metric_name] for metric_name in metrics])
            x = np.array([[i for i, label in enumerate(metrics)] for _ in range(y.shape[1])])

            # y_avg = np.median(y, axis=1)
            y_avg = np.average(y, axis=1)
            y_std = np.std(y, axis=1)
            if not markers:
                markers = self.MARKERS[:]
            marker = random.choice(markers)
            markers.remove(marker)
            if not colors:
                colors = self.COLORS[:]
            color = random.choice(colors)
            colors.remove(color)

            if kwargs.pop('avg_point', True):
                avg_point = ax.plot(x[0],
                                    y_avg,
                                    marker=marker,
                                    color=color,
                                    markersize=markersize,
                                    linewidth=linewidth
                                    )

            if kwargs.pop('traj_fill', True):
                traj_fill = plt.subplot().fill_between(x[0],
                                                       y_avg - y_std,
                                                       y_avg + y_std,
                                                       color=color,
                                                       alpha=alpha
                                                       )

            if kwargs.pop('traj_point', True):
                # color = random.choice(colors)
                # colors.remove(color)
                traj_point = plt.subplot().scatter(x,
                                                   y,
                                                   marker=marker,
                                                   color=color
                                                   )
            tex_xtick = list(trial_tag_list) if xticks is None else xticks

            traj_parts.append(avg_point[0])
            legend_labels.append(metric_name)
            if kwargs.get('legend', True):
                plt.legend(traj_parts, legend_labels, loc=legend_loc)

        plt.grid()
        if minorticks_on:
            plt.minorticks_on()

        plt.xticks(rotation=xrotation)
        plt.yticks(rotation=yrotation)
        plt.xlabel('' if xlabel is None else xlabel)
        plt.ylabel(', '.join(list(plot_metrics.keys())) if ylabel is None else ylabel)

        if save_path:
            global retry_count
            try:
                tikz_code = tikzplotlib.get_tikz_code()
            except ValueError as e:
                if retry_count > 0:
                    retry_count -= 1

                    self.traj_plot(save_path, **kwargs)
                else:
                    raise RuntimeError(e)

            tex_src = self.traj_plot_tex_template.replace('$tikz_code$', tikz_code)

            tex_src = tex_src.replace('$xticklabel$', ','.join(str(x) for x in tex_xtick))
            tex_src = tex_src.replace('$xtick$', ','.join([str(x) for x in range(len(tex_xtick))]))
            tex_src = tex_src.replace('$xlabel$', ', '.join(list(trial_tag_list)) if xlabel is None else xlabel)
            tex_src = tex_src.replace('$ylabel$', ', '.join(list(plot_metrics.keys())) if ylabel is None else ylabel)
            tex_src = tex_src.replace('$xtickshift$', str(xtickshift))
            tex_src = tex_src.replace('$ytickshift$', str(ytickshift))
            tex_src = tex_src.replace('$xlabelshift$', str(xlabelshift))
            tex_src = tex_src.replace('$ylabelshift$', str(ylabelshift))

            # tex_src = fix_tex_traj_plot_legend(tex_src, self.metrics)

            plt.savefig(save_path + '/' + self.name + '.pdf', dpi=1000)
            plt.show()
            fout = open((save_path + '/' + self.name + '_metric_traj_plot.tikz.tex').lstrip('_'), mode='w', encoding='utf8')
            fout.write(tex_src)
            fout.close()

            texs = find_cwd_files(['.tex', self.name, '_metric_traj_plot'])
            for pdf in texs:
                cmd = 'pdflatex "{}" '.format(pdf).replace(os.path.sep, '/')
                subprocess.check_call(shlex.split(cmd), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                # os.system(cmd)

            pdfs = find_cwd_files(['.pdf', self.name,  '_metric_traj_plot'], exclude_key='crop')
            for pdf in pdfs:
                cmd = 'pdfcrop "{}" "{}" '.format(pdf, pdf).replace(os.path.sep, '/')
                subprocess.check_call(shlex.split(cmd), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                # os.system(cmd)

            for f in find_cwd_files(['.aux', self.name]) + find_cwd_files(['.log',  self.name]) + find_cwd_files(['crop',  self.name]):
                os.remove(f)
            print('Tikz plot saved at ', find_cwd_files(['_metric_traj_plot', self.name], exclude_key='crop'))
        else:
            plt.show()
        print('Traj plot finished')
        plt.close()

    @exception_handle
    def box_plot(self, plot_metrics=None, save_path=None, **kwargs):

        markers = self.MARKERS[:]
        colors = self.COLORS[:]
        hatches = self.HATCHES[:]

        if isinstance(plot_metrics, str):  # warning for early version (<0.4.0)
            print('Please do not use this function directly for version (<0.4.0)')
        if not plot_metrics:
            plot_metrics = self.metrics
        ax = plt.subplot()

        alpha = kwargs.pop('alpha', 1)

        markersize = kwargs.pop('markersize', 3)

        xlabel = kwargs.pop('xlabel', None)

        xticks = kwargs.pop('xticks', None)

        ylabel = kwargs.pop('ylabel', None)

        yticks = kwargs.pop('yticks', None)

        legend_loc = kwargs.pop('legend_loc', 2)

        hatches = kwargs.pop('hatches', hatches)

        xrotation = kwargs.pop('xrotation', 0)

        yrotation = kwargs.pop('yrotation', 0)

        xlabelshift = kwargs.pop('xlabelshift', 1)

        ylabelshift = kwargs.pop('ylabelshift', 1)

        xtickshift = kwargs.pop('xtickshift', 1)

        ytickshift = kwargs.pop('ytickshift', 1)

        linewidth = kwargs.pop('linewidth', 3)

        widths = kwargs.pop('widths', 0.9)

        minorticks_on = kwargs.pop('minorticks_on', False)

        box_parts = []
        legend_labels = []
        trial_tag_list = kwargs.get('trial_tag_list', self.trial_tag_list)
        for metric_name in plot_metrics.keys():
            metrics = plot_metrics[metric_name]
            if not trial_tag_list or len(trial_tag_list) != len(metrics.keys()):
                if trial_tag_list:
                    print('Unequal length of trial_tag_list and trial_num:', trial_tag_list, '<->', list(metrics.keys()))
                trial_tag_list = list(metrics.keys())
            else:
                trial_tag_list = trial_tag_list
            color = random.choice(colors)
            colors.remove(color)
            tex_xtick = list(trial_tag_list) if xticks is None else xticks

            data = [metrics[trial] for trial in metrics.keys()]

            boxs_parts = ax.boxplot(data, positions=list(range(len(trial_tag_list))), widths=widths, meanline=True)

            box_parts.append(boxs_parts['boxes'][0])
            legend_labels.append(metric_name)

            for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
                plt.setp(boxs_parts[item], color=color)

            plt.setp(boxs_parts["fliers"], markeredgecolor=color)

        plt.grid()
        if minorticks_on:
            plt.minorticks_on()
        plt.xticks(rotation=xrotation)
        plt.yticks(rotation=yrotation)
        plt.xlabel('' if xlabel is None else xlabel)
        plt.ylabel(', '.join(list(plot_metrics.keys())) if ylabel is None else ylabel)

        if kwargs.get('legend', True):
            plt.legend(box_parts, legend_labels, loc=legend_loc)

        if save_path:
            global retry_count
            try:
                tikz_code = tikzplotlib.get_tikz_code()
            except ValueError as e:
                if retry_count > 0:
                    retry_count -= 1

                    self.traj_plot(save_path, **kwargs)
                else:
                    raise RuntimeError(e)

            tex_src = self.box_plot_tex_template.replace('$tikz_code$', tikz_code)

            tex_src = tex_src.replace('$xticklabel$', ','.join([str(x) for x in tex_xtick]))
            tex_src = tex_src.replace('$xtick$', ','.join([str(x) for x in range(len(tex_xtick))]))
            tex_src = tex_src.replace('$xlabel$', ', '.join(list(trial_tag_list)) if xlabel is None else xlabel)
            tex_src = tex_src.replace('$ylabel$', ', '.join(list(plot_metrics.keys())) if ylabel is None else ylabel)
            tex_src = tex_src.replace('$xtickshift$', str(xtickshift))
            tex_src = tex_src.replace('$ytickshift$', str(ytickshift))
            tex_src = tex_src.replace('$xlabelshift$', str(xlabelshift))
            tex_src = tex_src.replace('$ylabelshift$', str(ylabelshift))

            plt.savefig(save_path + '/' + self.name + '.pdf', dpi=1000)
            plt.show()
            fout = open((save_path + '/' + self.name + '_metric_box_plot.tikz.tex').lstrip('_'), mode='w', encoding='utf8')
            fout.write(tex_src)
            fout.close()

            texs = find_cwd_files(['.tex', self.name, '_metric_box_plot'])
            for pdf in texs:
                cmd = 'pdflatex "{}" '.format(pdf).replace(os.path.sep, '/')
                subprocess.check_call(shlex.split(cmd), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                # os.system(cmd)

            pdfs = find_cwd_files(['.pdf', self.name,  '_metric_box_plot'], exclude_key='crop')
            for pdf in pdfs:
                cmd = 'pdfcrop "{}" "{}" '.format(pdf, pdf).replace(os.path.sep, '/')
                subprocess.check_call(shlex.split(cmd), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                # os.system(cmd)

            for f in find_cwd_files(['.aux', self.name]) + find_cwd_files(['.log',  self.name]) + find_cwd_files(['crop',  self.name]):
                os.remove(f)
            print('Tikz plot saved at ', find_cwd_files(['_metric_box_plot', self.name], exclude_key='crop'))
        else:
            plt.show()
        print('Box plot finished')
        plt.close()

    @exception_handle
    def avg_bar_plot(self, plot_metrics=None, save_path=None, **kwargs):

        markers = self.MARKERS[:]
        colors = self.COLORS[:]
        hatches = self.HATCHES[:]

        if isinstance(plot_metrics, str):  # warning for early version (<0.4.0)
            print('Please do not use this function directly for version (<0.4.0)')
        if not plot_metrics:
            plot_metrics = self.metrics

        ax = plt.subplot()

        alpha = kwargs.pop('alpha', 1)

        markersize = kwargs.pop('markersize', 3)

        xlabel = kwargs.pop('xlabel', None)

        xticks = kwargs.pop('xticks', None)

        ylabel = kwargs.pop('ylabel', None)

        yticks = kwargs.pop('yticks', None)

        legend_loc = kwargs.pop('legend_loc', 2)

        hatches = kwargs.pop('hatches', hatches)

        xrotation = kwargs.pop('xrotation', 0)

        yrotation = kwargs.pop('yrotation', 0)

        xlabelshift = kwargs.pop('xlabelshift', 1)

        ylabelshift = kwargs.pop('ylabelshift', 1)

        xtickshift = kwargs.pop('xtickshift', 1)

        ytickshift = kwargs.pop('ytickshift', 1)

        linewidth = kwargs.pop('linewidth', 3)

        widths = kwargs.pop('widths', 0.9)

        minorticks_on = kwargs.pop('minorticks_on', False)

        sum_bar_parts = []
        total_width = 0.9
        trial_tag_list = kwargs.get('trial_tag_list ', self.trial_tag_list)
        for i, metric_name in enumerate(plot_metrics.keys()):
            metrics = plot_metrics[metric_name]
            if not trial_tag_list or len(trial_tag_list) != len(metrics.keys()):
                if trial_tag_list:
                    print('Unequal length of trial_tag_list and trial_num:', trial_tag_list, '<->', list(metrics.keys()))
                trial_tag_list = list(metrics.keys())
            else:
                trial_tag_list = trial_tag_list
            metric_num = len(plot_metrics.keys())
            trial_num = len(plot_metrics[metric_name])
            width = total_width / metric_num
            x = np.arange(trial_num)
            x = x - (total_width - width) / 2
            x = x + i * width
            Y = np.array([np.average(plot_metrics[m_name][trial]) for m_name in plot_metrics.keys() for trial in plot_metrics[m_name] if metric_name == m_name])
            hatch = random.choice(hatches)
            hatches.remove(hatch)
            color = random.choice(colors)
            colors.remove(color)
            if save_path:
                bar = plt.bar(x, Y, width=width, label=metric_name, hatch=hatch, color=color)
                plt.legend()
            else:
                bar = plt.bar(x, Y, width=width, hatch=hatch, color=color)
                sum_bar_parts.append(bar[0])
                legend_labels = list(plot_metrics.keys())
                if kwargs.get('legend', True):
                    plt.legend(sum_bar_parts, legend_labels, loc=legend_loc)

            for i, j in zip(x, Y):
                plt.text(i, j + max(Y) // 100, '%.1f' % j, ha='center', va='bottom')

            tex_xtick = list(trial_tag_list) if xticks is None else xticks

        plt.grid()
        if minorticks_on:
            plt.minorticks_on()
        plt.xticks(rotation=xrotation)
        plt.yticks(rotation=yrotation)
        plt.xlabel('' if xlabel is None else xlabel)
        plt.ylabel(', '.join(list(plot_metrics.keys())) if ylabel is None else ylabel)

        if save_path:
            global retry_count
            try:
                tikz_code = tikzplotlib.get_tikz_code()
            except ValueError as e:
                if retry_count > 0:
                    retry_count -= 1

                    self.traj_plot(save_path, **kwargs)
                else:
                    raise RuntimeError(e)

            tex_src = self.bar_plot_tex_template.replace('$tikz_code$', tikz_code)

            tex_src = tex_src.replace('$xticklabel$', ','.join([str(x) for x in tex_xtick]))
            tex_src = tex_src.replace('$xtick$', ','.join([str(x) for x in range(len(tex_xtick))]))
            tex_src = tex_src.replace('$xlabel$', ', '.join(list(trial_tag_list)) if xlabel is None else xlabel)
            tex_src = tex_src.replace('$ylabel$', ', '.join(list(plot_metrics.keys())) if ylabel is None else ylabel)
            tex_src = tex_src.replace('$xtickshift$', str(xtickshift))
            tex_src = tex_src.replace('$ytickshift$', str(ytickshift))
            tex_src = tex_src.replace('$xlabelshift$', str(xlabelshift))
            tex_src = tex_src.replace('$ylabelshift$', str(ylabelshift))

            plt.savefig(save_path + '/' + self.name + '.pdf', dpi=1000)
            plt.show()
            fout = open((save_path + '/' + self.name + '_metric_avg_bar_plot.tikz.tex').lstrip('_'), mode='w', encoding='utf8')
            fout.write(tex_src)
            fout.close()

            texs = find_cwd_files(['.tex', self.name, '_metric_avg_bar_plot'])
            for pdf in texs:
                cmd = 'pdflatex "{}" '.format(pdf).replace(os.path.sep, '/')
                subprocess.check_call(shlex.split(cmd), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                # os.system(cmd)

            pdfs = find_cwd_files(['.pdf', self.name,  '_metric_avg_bar_plot'], exclude_key='crop')
            for pdf in pdfs:
                cmd = 'pdfcrop "{}" "{}" '.format(pdf, pdf).replace(os.path.sep, '/')
                subprocess.check_call(shlex.split(cmd), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                # os.system(cmd)

            for f in find_cwd_files(['.aux', self.name]) + find_cwd_files(['.log',  self.name]) + find_cwd_files(['crop',  self.name]):
                os.remove(f)
            print('Tikz plot saved at ', find_cwd_files(['_metric_avg_bar_plot', self.name], exclude_key='crop'))
        else:
            plt.show()
        print('Avg Bar plot finished')
        plt.close()

    @exception_handle
    def sum_bar_plot(self, plot_metrics=None, save_path=None, **kwargs):

        markers = self.MARKERS[:]
        colors = self.COLORS[:]
        hatches = self.HATCHES[:]

        if isinstance(plot_metrics, str):  # warning for early version (<0.4.0)
            print('Please do not use this function directly for version (<0.4.0)')
        if not plot_metrics:
            plot_metrics = self.metrics

        ax = plt.subplot()

        alpha = kwargs.pop('alpha', 1)

        markersize = kwargs.pop('markersize', 3)

        xlabel = kwargs.pop('xlabel', None)

        xticks = kwargs.pop('xticks', None)

        ylabel = kwargs.pop('ylabel', None)

        yticks = kwargs.pop('yticks', None)

        legend_loc = kwargs.pop('legend_loc', 2)

        hatches = kwargs.pop('hatches', hatches)

        xrotation = kwargs.pop('xrotation', 0)

        yrotation = kwargs.pop('yrotation', 0)

        xlabelshift = kwargs.pop('xlabelshift', 1)

        ylabelshift = kwargs.pop('ylabelshift', 1)

        xtickshift = kwargs.pop('xtickshift', 1)

        ytickshift = kwargs.pop('ytickshift', 1)

        linewidth = kwargs.pop('linewidth', 3)

        widths = kwargs.pop('widths', 0.9)

        minorticks_on = kwargs.pop('minorticks_on', False)

        sum_bar_parts = []
        total_width = 0.9
        trial_tag_list = kwargs.get('trial_tag_list ', self.trial_tag_list)
        for i, metric_name in enumerate(plot_metrics.keys()):
            metrics = plot_metrics[metric_name]
            if not trial_tag_list or len(trial_tag_list) != len(metrics.keys()):
                if trial_tag_list:
                    print('Unequal length of trial_tag_list and trial_num:', trial_tag_list, '<->', list(metrics.keys()))
                trial_tag_list = list(metrics.keys())
            else:
                trial_tag_list = trial_tag_list
            metric_num = len(plot_metrics.keys())
            trial_num = len(plot_metrics[metric_name])
            width = total_width / metric_num
            x = np.arange(trial_num)
            x = x - (total_width - width) / 2
            x = x + i * width
            Y = np.array([np.sum(plot_metrics[m_name][trial]) for m_name in plot_metrics.keys() for trial in plot_metrics[m_name] if metric_name == m_name])
            hatch = random.choice(hatches)
            hatches.remove(hatch)
            color = random.choice(colors)
            colors.remove(color)
            if save_path:
                bar = plt.bar(x, Y, width=width, label=metric_name, hatch=hatch, color=color)
                plt.legend()
            else:
                bar = plt.bar(x, Y, width=width, hatch=hatch, color=color)
                sum_bar_parts.append(bar[0])
                legend_labels = list(plot_metrics.keys())
                if kwargs.get('legend', True):
                    plt.legend(sum_bar_parts, legend_labels, loc=legend_loc)

            for i, j in zip(x, Y):
                plt.text(i, j + max(Y) // 100, '%.1f' % j, ha='center', va='bottom')

            tex_xtick = list(trial_tag_list) if xticks is None else xticks

        plt.grid()
        if minorticks_on:
            plt.minorticks_on()
        plt.xticks(rotation=xrotation)
        plt.yticks(rotation=yrotation)
        plt.xlabel('' if xlabel is None else xlabel)
        plt.ylabel(', '.join(list(plot_metrics.keys())) if ylabel is None else ylabel)

        if save_path:
            global retry_count
            try:
                tikz_code = tikzplotlib.get_tikz_code()
            except ValueError as e:
                if retry_count > 0:
                    retry_count -= 1

                    self.traj_plot(save_path, **kwargs)
                else:
                    raise RuntimeError(e)

            tex_src = self.bar_plot_tex_template.replace('$tikz_code$', tikz_code)

            tex_src = tex_src.replace('$xticklabel$', ','.join([str(x) for x in tex_xtick]))
            tex_src = tex_src.replace('$xtick$', ','.join([str(x) for x in range(len(tex_xtick))]))
            tex_src = tex_src.replace('$xlabel$', ', '.join(list(trial_tag_list)) if xlabel is None else xlabel)
            tex_src = tex_src.replace('$ylabel$', ', '.join(list(plot_metrics.keys())) if ylabel is None else ylabel)
            tex_src = tex_src.replace('$xtickshift$', str(xtickshift))
            tex_src = tex_src.replace('$ytickshift$', str(ytickshift))
            tex_src = tex_src.replace('$xlabelshift$', str(xlabelshift))
            tex_src = tex_src.replace('$ylabelshift$', str(ylabelshift))

            plt.savefig(save_path + '/' + self.name + '.pdf', dpi=1000)
            plt.show()
            fout = open((save_path + '/' + self.name + '_metric_sum_bar_plot.tikz.tex').lstrip('_'), mode='w', encoding='utf8')
            fout.write(tex_src)
            fout.close()

            texs = find_cwd_files(['.tex', self.name, '_metric_sum_bar_plot'])
            for pdf in texs:
                cmd = 'pdflatex "{}" '.format(pdf).replace(os.path.sep, '/')
                subprocess.check_call(shlex.split(cmd), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                # os.system(cmd)

            pdfs = find_cwd_files(['.pdf', self.name,  '_metric_sum_bar_plot'], exclude_key='crop')
            for pdf in pdfs:
                cmd = 'pdfcrop "{}" "{}" '.format(pdf, pdf).replace(os.path.sep, '/')
                subprocess.check_call(shlex.split(cmd), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                # os.system(cmd)

            for f in find_cwd_files(['.aux', self.name]) + find_cwd_files(['.log',  self.name]) + find_cwd_files(['crop',  self.name]):
                os.remove(f)
            print('Tikz plot saved at ', find_cwd_files(['_metric_sum_bar_plot', self.name], exclude_key='crop'))
        else:
            plt.show()
        print('Sum Bar plot finished')
        plt.close()

    @exception_handle
    def violin_plot(self, plot_metrics=None, save_path=None, **kwargs):

        markers = self.MARKERS[:]
        colors = self.COLORS[:]
        hatches = self.HATCHES[:]

        if isinstance(plot_metrics, str):  # warning for early version (<0.4.0)
            print('Please do not use this function directly for version (<0.4.0)')
        if not plot_metrics:
            plot_metrics = self.metrics

        legend_labels = []

        # def add_label(violin, label):
        #     color = violin["bodies"][0].get_facecolor().flatten()
        #     legend_labels.append((mpatches.Patch(color=color), label))

        ax = plt.subplot()

        alpha = kwargs.pop('alpha', 1)

        markersize = kwargs.pop('markersize', 3)

        xlabel = kwargs.pop('xlabel', None)

        xticks = kwargs.pop('xticks', None)

        ylabel = kwargs.pop('ylabel', None)

        yticks = kwargs.pop('yticks', None)

        legend_loc = kwargs.pop('legend_loc', 2)

        hatches = kwargs.pop('hatches', hatches)

        xrotation = kwargs.pop('xrotation', 0)

        yrotation = kwargs.pop('yrotation', 0)

        xlabelshift = kwargs.pop('xlabelshift', 1)

        ylabelshift = kwargs.pop('ylabelshift', 1)

        xtickshift = kwargs.pop('xtickshift', 1)

        ytickshift = kwargs.pop('ytickshift', 1)

        linewidth = kwargs.pop('linewidth', 3)

        widths = kwargs.pop('widths', 0.9)

        minorticks_on = kwargs.pop('minorticks_on', False)

        violin_parts = []
        legend_labels = []
        trial_tag_list = kwargs.get('trial_tag_list ', self.trial_tag_list)
        for metric_name in plot_metrics.keys():
            metrics = plot_metrics[metric_name]

            if not kwargs.get('trial_tag_list ', trial_tag_list) or len(trial_tag_list) != len(metrics.keys()):
                if trial_tag_list:
                    print('Unequal length of trial_tag_list and trial_num:', trial_tag_list, '<->', list(metrics.keys()))
                trial_tag_list = list(metrics.keys())
            else:
                trial_tag_list = trial_tag_list
            tex_xtick = list(trial_tag_list) if xticks is None else xticks
            data = [metrics[trial] for trial in metrics.keys()]

            if save_path:
                violin = ax.violinplot(data, widths=widths, positions=list(range(len(trial_tag_list))), showmeans=True, showmedians=True, showextrema=True)
                violin_parts.append(violin['bodies'][0])
                legend_labels = list(plot_metrics.keys())
                if kwargs.get('legend', True):
                    plt.legend(violin_parts, legend_labels, loc=0)
            else:
                violin = ax.violinplot(data, widths=widths, positions=list(range(len(trial_tag_list))), showmeans=True, showmedians=True, showextrema=True)
                violin_parts.append(violin['bodies'][0])
                legend_labels = list(plot_metrics.keys())
                if kwargs.get('legend', True):
                    plt.legend(violin_parts, legend_labels, loc=legend_loc)

            for pc in violin['bodies']:
                pc.set_linewidth(linewidth)

        plt.grid()
        if minorticks_on:
            plt.minorticks_on()
        plt.xticks(rotation=xrotation)
        plt.yticks(rotation=yrotation)
        plt.xlabel('' if xlabel is None else xlabel)
        plt.ylabel(', '.join(list(plot_metrics.keys())) if ylabel is None else ylabel)

        if save_path:
            global retry_count
            try:
                tikz_code = tikzplotlib.get_tikz_code()
            except ValueError as e:
                if retry_count > 0:
                    retry_count -= 1

                    self.traj_plot(save_path, **kwargs)
                else:
                    raise RuntimeError(e)

            tex_src = self.box_plot_tex_template.replace('$tikz_code$', tikz_code)

            tex_src = tex_src.replace('$xticklabel$', ','.join([str(x) for x in tex_xtick]))
            tex_src = tex_src.replace('$xtick$', ','.join([str(x) for x in range(len(tex_xtick))]))
            tex_src = tex_src.replace('$xlabel$', ', '.join(list(trial_tag_list)) if xlabel is None else xlabel)
            tex_src = tex_src.replace('$ylabel$', ', '.join(list(plot_metrics)) if ylabel is None else ylabel)
            tex_src = tex_src.replace('$xtickshift$', str(xtickshift))
            tex_src = tex_src.replace('$ytickshift$', str(ytickshift))
            tex_src = tex_src.replace('$xlabelshift$', str(xlabelshift))
            tex_src = tex_src.replace('$ylabelshift$', str(ylabelshift))

            plt.savefig(save_path + '/' + self.name + '.pdf', dpi=1000)
            plt.show()
            fout = open((save_path + '/' + self.name + '_metric_violin_plot.tikz.tex').lstrip('_'), mode='w', encoding='utf8')
            fout.write(tex_src)
            fout.close()

            texs = find_cwd_files(['.tex', self.name, '_metric_violin_plot'])
            for pdf in texs:
                cmd = 'pdflatex "{}"'.format(pdf).replace(os.path.sep, '/')
                subprocess.check_call(shlex.split(cmd), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                # os.system(cmd)

            pdfs = find_cwd_files(['.pdf', self.name, '_metric_violin_plot'], exclude_key='crop')
            for pdf in pdfs:
                cmd = 'pdfcrop "{}" "{}"'.format(pdf, pdf).replace(os.path.sep, '/')
                subprocess.check_call(shlex.split(cmd), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                # os.system(cmd)

            for f in find_cwd_files(['.aux', self.name]) + find_cwd_files(['.log',  self.name]) + find_cwd_files(['crop',  self.name]):
                os.remove(f)
            print('Tikz plot saved at ', find_cwd_files(['_metric_violin_plot', self.name], exclude_key='crop'))
        else:
            plt.show()
        print('Violin plot finished')
        plt.close()

    @exception_handle
    def transpose(self):
        transposed_metrics = OrderedDict()
        for metric_name in self.metrics.keys():
            for trial_tag_list in self.metrics[metric_name].keys():
                if trial_tag_list not in transposed_metrics:
                    transposed_metrics[trial_tag_list] = {}
                transposed_metrics[trial_tag_list][metric_name] = self.metrics[metric_name][trial_tag_list]
        return transposed_metrics

    @exception_handle
    def A12_plot(self):
        raise NotImplementedError()

    # @exception_handle
    # def trial_sk_rank_test_plot(self, save_path=None, **kwargs):
    #     raise NotImplementedError()

    # @exception_handle
    # def metric_sk_rank_test_plot(self, save_path=None, **kwargs):
    #     raise NotImplementedError()

    @exception_handle
    def _rank_test_by_trial(self):
        transposed_metrics = self.transpose()
        for trial in transposed_metrics.keys():
            self.trial_rank_test_result[trial] = {}
            for metric1 in transposed_metrics[trial].keys():
                for metric2 in transposed_metrics[trial].keys():
                    if metric1 != metric2:
                        result = ranksums(transposed_metrics[trial][metric1], transposed_metrics[trial][metric2])
                        self.trial_rank_test_result[trial]['{}<->{}'.format(metric1, metric2)] = result

        return self.trial_rank_test_result

    @exception_handle
    def rank_test_by_trail(self, trial):
        self._rank_test_by_trial()
        try:
            return self.trial_rank_test_result[trial]
        except KeyError:
            raise KeyError('Trial {} not found, please select trial in {}'.format(trial, list(self.transpose().keys())))

    @exception_handle
    def _rank_test_by_metric(self):
        trial_tag_list = list(self.transpose().keys())
        for metric in self.metrics.keys():
            self.metric_rank_test_result[metric] = {}
            for trial1 in trial_tag_list:
                for trial2 in trial_tag_list:
                    if trial1 != trial2:
                        result = ranksums(self.metrics[metric][trial1], self.metrics[metric][trial2])
                        self.metric_rank_test_result[metric]['{}<->{}'.format(trial1, trial2)] = result
        return self.metric_rank_test_result

    @exception_handle
    def rank_test_by_metric(self, metric):
        self._rank_test_by_metric()
        try:
            return self.metric_rank_test_result[metric]
        except KeyError:
            raise KeyError('Metric {} not found, please select metric in {}'.format(metric, list(self.metrics.keys())))

    @exception_handle
    def scott_knott_plot(self, save_path=None, **kwargs):
        from metric_visualizer.external import Rx
        data_dict = {'Scott-Knott Rank Test': {}}

        for metric in self.metrics:
            Rx.data(**self.metrics[metric])
            for i, d in enumerate(Rx.list_algorithm_rank):
                data_dict['Scott-Knott Rank Test'][d[0]] = data_dict['Scott-Knott Rank Test'].get(d[0], [])
                data_dict['Scott-Knott Rank Test'][d[0]].append(d[1])

        trial_tag_list = sorted(list(data_dict['Scott-Knott Rank Test'].keys()))
        mv = MetricVisualizer(trial_tag='Scott-Knott Rank Test', trial_tag_list=trial_tag_list, metric_dict=data_dict)
        mv.box_plot_by_trial(save_path='./sk_rank', ylabel='Scott-Knott Rank Test', xlabel='Model', **kwargs)

    @exception_handle
    def summary(self, save_path=None, no_print=False, **kwargs):
        summary_str = ' ------------------------------------- Metric Visualizer ------------------------------------- \n'
        header = ['Metric', self.trial_tag, 'Values (First 10 values)', 'Summary']

        table_data = []
        trial_tag_list = kwargs.get('trial_tag_list ', self.trial_tag_list)
        for mn in self.metrics.keys():
            metrics = self.metrics[mn]
            if not trial_tag_list or len(trial_tag_list) != len(metrics.keys()):
                if len(trial_tag_list) > len(metrics.keys()):
                    trial_tag_list = trial_tag_list[:len(metrics.keys())]
                else:
                    trial_tag_list = list(metrics.keys())
            else:
                trial_tag_list = trial_tag_list
            for i, trial in enumerate(metrics.keys()):
                _data = []
                _data += [[mn, trial_tag_list[i], [round(x, 2) for x in metrics[trial][:10]]]]
                _data[-1].append(
                    ['Avg:{}, Median: {}, IQR: {}, STD:{}, Max: {}, Min: {}'.format(
                        round(np.average(metrics[trial]), 2),
                        round(np.median(metrics[trial]), 2),
                        round(np.std(metrics[trial]), 2),
                        round(iqr(metrics[trial], rng=(25, 75), interpolation='midpoint'), 2),
                        round(np.max(metrics[trial]), 2),
                        round(np.min(metrics[trial]), 2)
                    )]
                )
                table_data += _data

        summary_str += tabulate(table_data,
                                headers=header,
                                numalign='center',
                                tablefmt='fancy_grid')
        summary_str += '\n -------------------- https://github.com/yangheng95/metric_visualizer --------------------\n'
        if not no_print:
            print(summary_str)

        if save_path:
            fout = open(save_path + '/' + self.name + '_summary.txt', mode='w', encoding='utf8')
            summary_str += '\n{}\n'.format(str(self.metrics))
            fout.write(summary_str)
            fout.close()

            self.dump()

        return summary_str

    def dump(self, filename=None):
        if not filename:
            if self.dump_pointer:
                os.remove(self.dump_pointer)
            time_tag = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            trial_id = self.trial_id
            postfix = 'mv'
            self.dump_pointer = '{}_{}_id{}.{}'.format(self.name, time_tag, trial_id, postfix)
        else:
            self.dump_pointer = filename

        pickle.dump(self, open(self.dump_pointer, mode='wb'))

    @staticmethod
    def load(filename='metric_visualizer.dat'):
        if not os.path.exists(filename):
            dats = find_cwd_files(filename)
            if not dats:
                raise ValueError('Can not find {}'.format(filename))
            else:
                filename = max(dats)
        print('Load', filename)
        mv = pickle.load(open(filename, mode='rb'))
        return mv
