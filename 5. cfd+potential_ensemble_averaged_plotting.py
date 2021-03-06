import numpy as np
import pandas as pd
import re
import os.path

import matplotlib.pyplot as plt

from matplotlib import colors
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle

from mpl_plotter_mpl_plotting_methods import MatPlotLibPublicationPlotter as mplPlotter

plane = 'y=0'

data_analysis = os.path.dirname(__file__)
data_path = os.path.join(data_analysis, 'data')
img_path = os.path.join(data_analysis, 'images')
bin_path = os.path.join(data_path, 'bins')
uncertainty_path = os.path.join(data_path, 'uncertainty_mean_estimate')

comp_field_path = os.path.join(data_path, 'comparison_fields')

ensemble_method = 'rbf'
comp_fields = ['CFD', 'potential']
comp_field = comp_fields[0]

comparison_field_path = os.path.join(os.path.join(comp_field_path, comp_field), plane)

if plane is 'x=10' or plane is 'x=-10':
    figsize = (20, 5)
    fill = 0
    shrink = 0.9
    cbtit_y = -5
    save = True

    y_ticks = 4
    x_ticks = 5 if plane == 'y=0' or plane == 'z=0' else 4
    degree = 2
    tsize = 22
    axsize = 25
    pad = 9
    tit_y = 1.05
    cbtit_size = 18
    fillsphere = True
    aspect = 1
    cb_tcksz = 15
    tick_size = 15
else:
    figsize = (20, 5)
    fill = 0
    unified_color = True
    shrink = 0.69
    cbtit_y = -5
    surface = False
    save = True

    y_ticks = 4
    x_ticks = 5 if plane == 'y=0' or plane == 'z=0' else 4
    degree = 2
    tsize = 22
    axsize = 25
    pad = 15
    tit_y = 1.05
    cbtit_size = 18
    cb_tcksz = 15
    tick_size = 15
    fillsphere = True
    aspect = 1

# Figure setup
fig = mplPlotter(light=True).setup2d(figsize=figsize)

if plane == 'z=0' or plane == 'y=0':
    x_bounds = [0, 40]
    y_bounds = [0, 30]

else:
    x_bounds = [0, 30]
    y_bounds = [0, 30]

# CFD field
values = True

if values is True:
    if plane == 'x=-10':
        actualmax1 = 12
        actualmin1 = 0

        actualmax2 = 2.5
        actualmin2 = -2.5

        actualmax3 = 2.5
        actualmin3 = -2.5
    if plane == 'x=10':
        actualmax1 = 12
        actualmin1 = 0

        actualmax2 = 2.5
        actualmin2 = -2.5

        actualmax3 = 2.5
        actualmin3 = -2.5
    if plane == 'y=0':
        actualmax1 = 12.5
        actualmin1 = 0

        actualmax2 = 1
        actualmin2 = -1

        actualmax3 = 5
        actualmin3 = -5
    if plane == 'z=0':
        actualmax1 = 12.5
        actualmin1 = 0

        actualmax2 = 5
        actualmin2 = -5

        actualmax3 = 1
        actualmin3 = -1
else:
    actualmax1 = None
    actualmin1 = None

    actualmax2 = None
    actualmin2 = None

    actualmax3 = None
    actualmin3 = None


def find_real_extremes(mosaic):
    df = pd.DataFrame(mosaic)
    df = df.loc[:, (df != 0).any(axis=0)]
    min = df.min().min()
    max = df.max().max()
    return max, min


"""
Get fields
"""

u_mosaic = np.loadtxt(os.path.join(comparison_field_path, '{}_{}.txt'.format('u', comp_field) if comp_field == 'CFD' else '{}_{}.txt'.format(comp_field, 'u')))

try:
    v_mosaic = np.loadtxt(os.path.join(comparison_field_path, '{}_{}.txt'.format('v', comp_field) if comp_field == 'CFD' else '{}_{}.txt'.format(comp_field, 'v')))
except:
    if plane is 'y=0' or 'z=0':
        v_mosaic = np.zeros((30, 40))
    else:
        v_mosaic = np.zeros((30, 30))

try:
    w_mosaic = np.loadtxt(os.path.join(comparison_field_path, '{}_{}.txt'.format('w',
                                                                                 comp_field) if comp_field == 'CFD' else '{}_{}.txt'.format(
        comp_field, 'w')))
except:
    if plane is 'y=0' or 'z=0':
        w_mosaic = np.zeros((30, 40))
    else:
        w_mosaic = np.zeros((30, 30))


masks = [np.where(u_mosaic == 0), np.where(v_mosaic == 0), np.where(w_mosaic == 0)]

def empty_value_fields():
    u_empty = np.zeros((30, 40))
    v_empty = np.zeros((30, 40))
    w_empty = np.zeros((30, 40))

    u_empty[masks[0]] = 1
    v_empty[masks[1]] = 1
    w_empty[masks[2]] = 1

    return u_empty, v_empty, w_empty


def transp_colormap():
    cmap = colors.ListedColormap([(0, 0, 0, 0), (0, 0, 0, 1)])
    bounds = [1]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    return cmap


u_empty, v_empty, w_empty = empty_value_fields()

cmap_empty = transp_colormap()

"""
u
"""
comp = 'u'

ax1 = mplPlotter(fig=fig, shape_and_position=131).heatmap(array=u_mosaic, resize_axes=True, aspect=aspect,
                                                          tick_ndecimals=1,
                                                          xresize_pad=0, yresize_pad=0,
                                                          x_bounds=x_bounds,
                                                          y_bounds=y_bounds,
                                                          cb_vmax=actualmax1 if not isinstance(actualmax1, type(None)) else find_real_extremes(u_mosaic)[0],
                                                          cb_vmin=actualmin1 if not isinstance(actualmin1, type(None)) else find_real_extremes(u_mosaic)[1],
                                                          plot_title=None,  # r'$\mathit{' + comp + '}$' + ' in {} plane'.format(plane),
                                                          color_bar=True,
                                                          cb_axis_labelpad=10,
                                                          title_size=tsize,
                                                          title_y=tit_y,
                                                          custom_x_ticklabels=(-20, 20) if plane == 'y=0' or plane == 'z=0' else (-15, 15),
                                                          custom_y_ticklabels= (-15, 15),
                                                          xaxis_labelpad=pad - 10,
                                                          yaxis_labelpad=pad + 10,
                                                          xaxis_label_size=axsize,
                                                          yaxis_label_size=axsize,
                                                          xaxis_bold=True,
                                                          yaxis_bold=True,
                                                          x_label='x $[cm]$', y_label='y $[cm]$',
                                                          cb_top_title=True,
                                                          cb_top_title_pad=cbtit_y,
                                                          cb_top_title_x=-1,
                                                          cb_title='{} $[m/s]$'.format(comp),
                                                          cb_title_weight='bold',
                                                          cb_title_size=cbtit_size,
                                                          cb_top_title_y=1.1,
                                                          x_tick_number=x_ticks,
                                                          y_tick_number=y_ticks,
                                                          more_subplots_left=True,
                                                          shrink=shrink,
                                                          cb_nticks=5,
                                                          cb_ticklabelsize=cb_tcksz,
                                                          tick_label_size=tick_size,
                                                          )

mplPlotter(fig=fig, ax=ax1).heatmap(array=u_empty, resize_axes=False, cmap=cmap_empty,
                                    x_tick_number=x_ticks,
                                    y_tick_number=y_ticks,
                                    custom_x_ticklabels=(-20, 20) if plane == 'y=0' or plane == 'z=0' else (-15, 15),
                                    custom_y_ticklabels=(-15, 15),
                                    plot_title=None,
                                    more_subplots_left=True,
                                    )

"""
v
"""
comp = 'v'

ax2 = mplPlotter(fig=fig, shape_and_position=132).heatmap(array=v_mosaic, resize_axes=True, aspect=aspect,
                                                          tick_ndecimals=1,
                                                          xresize_pad=0, yresize_pad=0,
                                                          x_bounds=x_bounds,
                                                          y_bounds=y_bounds,
                                                          plot_title=None,  # r'$\mathit{' + comp + '}$' + ' in {} plane'.format(plane),
                                                          color_bar=True,
                                                          cb_vmax=actualmax2 if not isinstance(actualmax2, type(None)) else find_real_extremes(v_mosaic)[0],
                                                          cb_vmin=actualmin2 if not isinstance(actualmin2, type(None)) else find_real_extremes(v_mosaic)[1],
                                                          cb_axis_labelpad=10,
                                                          title_size=tsize,
                                                          custom_x_ticklabels=(-20, 20) if plane == 'y=0' or plane == 'z=0' else (-15, 15),
                                                          custom_y_ticklabels= (-15, 15),
                                                          title_y=tit_y,
                                                          cb_top_title_x=-1,
                                                          x_tick_number=x_ticks,
                                                          y_tick_number=0,
                                                          more_subplots_left=True,
                                                          shrink=shrink,
                                                          cb_top_title=True,
                                                          cb_top_title_pad=cbtit_y,
                                                          cb_title='{} $[m/s]$'.format(comp),
                                                          cb_title_weight='bold',
                                                          cb_top_title_y=1.1,
                                                          cb_title_size=cbtit_size,
                                                          cb_nticks=5,
                                                          cb_ticklabelsize=cb_tcksz,
                                                          tick_label_size=tick_size,
                                                          )

mplPlotter(fig=fig, ax=ax2).heatmap(array=v_empty, resize_axes=False, cmap=cmap_empty,
                                    x_tick_number=x_ticks,
                                    y_tick_number=y_ticks,
                                    custom_x_ticklabels=(-20, 20) if plane == 'y=0' or plane == 'z=0' else (-15, 15),
                                    custom_y_ticklabels=(-15, 15),
                                    plot_title=None,
                                    more_subplots_left=True,
                                    )

"""
w
"""
comp = 'w'

ax3 = mplPlotter(fig=fig, shape_and_position=133).heatmap(array=w_mosaic, resize_axes=True, aspect=aspect,
                                                          tick_ndecimals=1,
                                                          xresize_pad=0, yresize_pad=0,
                                                          x_bounds=x_bounds,
                                                          y_bounds=y_bounds,
                                                          plot_title=None,  # r'$\mathit{' + comp + '}$' + ' in {} plane'.format(plane),
                                                          color_bar=True,
                                                          cb_vmax=actualmax3 if not isinstance(actualmax3, type(None)) else find_real_extremes(w_mosaic)[0],
                                                          cb_vmin=actualmin3 if not isinstance(actualmin3, type(None)) else find_real_extremes(w_mosaic)[1],
                                                          cb_axis_labelpad=10,
                                                          title_size=tsize,
                                                          title_y=tit_y,
                                                          cb_top_title_x=-1,
                                                          x_tick_number=x_ticks,
                                                          y_tick_number=0,
                                                          custom_x_ticklabels=(-20, 20) if plane == 'y=0' or plane == 'z=0' else (-15, 15),
                                                          custom_y_ticklabels=(-15, 15),
                                                          more_subplots_left=True,
                                                          shrink=shrink,
                                                          cb_top_title=True,
                                                          cb_top_title_pad=cbtit_y,
                                                          cb_title='{} $[m/s]$'.format(comp),
                                                          cb_title_weight='bold',
                                                          cb_top_title_y=1.1,
                                                          cb_title_size=cbtit_size,
                                                          cb_nticks=5,
                                                          cb_ticklabelsize=cb_tcksz,
                                                          tick_label_size=tick_size,
                                                          )

mplPlotter(fig=fig, ax=ax3).heatmap(array=w_empty, resize_axes=False, cmap=cmap_empty,
                                    x_tick_number=x_ticks,
                                    y_tick_number=y_ticks,
                                    custom_x_ticklabels=(-20, 20) if plane == 'y=0' or plane == 'z=0' else (-15, 15),
                                    custom_y_ticklabels=(-15, 15),
                                    plot_title=None,
                                    more_subplots_left=True,
                                    )

"""
u shapes
"""
if plane == 'z=0':
    rect_loc = (0, 20)
    rect_width = 40
    rect_height = 10
    rect_show = True

    text_x = 5
    text_y = 24
    text_show = True

    sphere_loc = (20, 15)
elif plane == 'x=10' or plane == 'x=-10':
    rect_loc = (20, 0)
    rect_width = 10
    rect_height = 30
    rect_show = True

    text_x = 20.5
    text_y = 24
    text_show = True

    fillsphere = False

    sphere_loc = (15, 15)
else:
    rect_show = False
    text_show = False
    sphere_loc = (20, 15)

if rect_show is True:
    unknown = Rectangle(rect_loc, width=rect_width, height=rect_height, facecolor=(4/255, 15/255, 115/255))
    ax1.add_patch(unknown)

sphere = Circle(sphere_loc, 7.5, facecolor='white', edgecolor='w', lw=2, fill=fillsphere)
ax1.add_patch(sphere)

if text_show is True:
    mplPlotter(fig=fig, ax=ax1).floating_text2d(text='NO DATA', color='white', size=15,
                                                              x=text_x, y=text_y, weight='bold')

"""
v shapes
"""

if rect_show is True:
    unknown = Rectangle(rect_loc, width=rect_width, height=rect_height, facecolor=(4/255, 15/255, 115/255))
    ax2.add_patch(unknown)

sphere = Circle(sphere_loc, 7.5, facecolor='white', edgecolor='w', lw=2, fill=fillsphere)
ax2.add_patch(sphere)

"""
w shapes
"""

if rect_show is True:
    unknown = Rectangle(rect_loc, width=rect_width, height=rect_height, facecolor=(4/255, 15/255, 115/255))
    ax3.add_patch(unknown)

sphere = Circle(sphere_loc, 7.5, facecolor='white', edgecolor='w', lw=2, fill=fillsphere)
ax3.add_patch(sphere)

if save is True:
    plt.savefig(os.path.join(img_path, '{}_Ensemble_Averaging_{}.png'.format(comp_field, plane)),
                dpi=150)


plt.show()
