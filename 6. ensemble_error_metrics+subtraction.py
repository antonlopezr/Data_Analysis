import pandas as pd
import numpy as np
import re
import sys

from os import listdir
from scipy import interpolate
from scipy.interpolate import Rbf
import os.path

import matplotlib.pyplot as plt

from matplotlib.patches import Circle
from matplotlib.patches import Rectangle

from mpl_plotter_mpl_plotting_methods import MatPlotLibPublicationPlotter as mplPlotter

"""
Uncertainty of mean estimates
"""
data_analysis = os.path.dirname(__file__)
data_path = os.path.join(data_analysis, 'data')
img_path = os.path.join(data_analysis, 'images')
bin_path = os.path.join(data_path, 'bins')
uncertainty_path = os.path.join(data_path, 'uncertainty_mean_estimate')
ens_avg_path = os.path.join(data_path, 'velocity_ensemble_averaged')

def remove_nan():
    un = pd.read_excel(
        os.path.join(uncertainty_path, 'ErrorEst.xlsx'),
        index_col=0)
    un.columns = ['u', 'v', 'w']
    un = un[un['u'] != ' nan']
    un.to_csv(
        os.path.join(uncertainty_path, 'ErrorEst.csv'))

# remove_nan()

un = pd.read_csv(
    os.path.join(uncertainty_path, 'ErrorEst.csv'),
    index_col=0)

"""
RBF interpolation
    epsilon = 1
    smooth = 0.03
"""
def rbf2d(x, y, z, x_new, y_new):
    try:
        """
        Optimal values
            x=-10:      500, 0.02
            x=10:       500, 0.02
            y=0:        500, 0.02
            z=0:        500, 0.02
        """
        rbf = Rbf(x, y, z, epsilon=500, smooth=0.02)
    except:
        return False
    z_new = rbf(x_new, y_new)
    return z_new

def poly2(x, y, z, x_new, y_new):
    A = np.array([x*0+1, x, y, x*y, x**2, y**2, x**2*y, y**2*x, y**2*x**2]).T
    B = z
    coeff, r, rank, s = np.linalg.lstsq(A, B, rcond=None)
    a, b, c, d, e, f, g, h, i = coeff
    z_new = a + b*x_new + c*y_new + d*x_new*y_new + e*x_new**2 + f*y_new**2 + g*x_new**2*y_new + h*y_new**2*x_new + i*y_new**2*x_new**2
    return z_new

"""
 """
def interpolate_all(fill, plane, version, quirk, filenamestart, var, f):
    if quirk == 'xten' or quirk == 'xminusten':
        array_shape = (30, 30)
        k_top = 15
    else:
        array_shape = (30, 40)
        k_top = 20
    u_mosaic = np.empty(array_shape)
    v_mosaic = np.empty(array_shape)
    w_mosaic = np.empty(array_shape)
    i = 0
    for k in range(-k_top, k_top+1):
        for l in range(-15, 16):
            if quirk == 'xten' or quirk == 'xminusten':
                file = '{}{}_{}_{}.xlsx'.format(quirk, var, k, l)
            if quirk == 'yzero':
                file = '{}{}_{}_{}.xlsx'.format(quirk, k, var, l)
            if quirk == 'zzero':
                file = '{}{}_{}_{}.xlsx'.format(quirk, k, l, var)

            if os.path.isfile(os.path.join(bin_path, '{}_wo_outliers\{}'.format(
                              plane, file))):

                df = pd.read_excel(os.path.join(bin_path, '{}_wo_outliers\{}'.format(
                                   plane, file)))
                df.columns = ['x', 'y', 'z', 'u', 'v', 'w']
                x = df['x']
                y = df['y']
                z = df['z']
                u = df['u']
                v = df['v']
                w = df['w']

                # Interpolation setup
                loc_x, loc_y, loc_z = re.findall(r"-?\d+", file)
                loc_x = 10 * int(loc_x)
                loc_y = 10 * int(loc_y)
                loc_z = 10 * int(loc_z)  # In mm

                xx = loc_x+5
                yy = loc_y+5
                zz = loc_z+5
                if quirk == 'xten' or quirk == 'xminusten':
                    x1 = y
                    x2 = z
                    x_new = yy
                    y_new = zz
                if quirk == 'yzero':
                    x1 = x
                    x2 = z
                    x_new = xx
                    y_new = zz
                if quirk == 'zzero':
                    x1 = x
                    x2 = y
                    x_new = xx
                    y_new = yy

                # Uncertainty of mean estimates
                index = file[filenamestart:-5] + "'"

                """
                Average u
                """
                if un.loc[index][0] > 0.01:
                    # Interpolation
                    uu = f(x=x1, y=x2, z=u, x_new=x_new, y_new=y_new)

                    if uu is False:
                        print('{}: u interpolation not converged'.format(file))
                        uu = fill
                    else:
                        uu = uu.item()

                else:
                    uu = fill

                """
                Average v
                """
                if un.loc[index][1] > 0.01:
                    # Interpolation
                    vv = f(x=x1, y=x2, z=v, x_new=x_new, y_new=y_new)

                    if vv is False:
                        print('{}: u interpolation not converged'.format(file))
                        vv = fill
                    else:
                        vv = vv.item()

                else:
                    vv = fill

                """
                Average w
                """
                if un.loc[index][2] > 0.01:
                    # Interpolation
                    ww = f(x=x1, y=x2, z=w, x_new=x_new, y_new=y_new)

                    if ww is False:
                        print('{}: u interpolation not converged'.format(file))
                        ww = fill
                    else:
                        ww = ww.item()
                else:
                    ww = fill

            else:
                uu = fill
                vv = fill
                ww = fill

            if k < k_top and l < 15:
                u_mosaic[l+15][k+k_top] = uu
                v_mosaic[l+15][k+k_top] = vv
                w_mosaic[l+15][k+k_top] = ww

        print(i)
        i = i + 1

    names = ['u', 'v', 'w']
    msics = [u_mosaic, v_mosaic, w_mosaic]
    for comp in range(3):
        np.savetxt(os.path.join(ens_avg_path, r'{}\{}_{}.txt'.format(plane, names[comp], version)), msics[comp])

    return u_mosaic, v_mosaic, w_mosaic

def find_real_extremes(mosaic):
    df = pd.DataFrame(mosaic)
    df = df.loc[:, (df != 0).any(axis=0)]
    min = df.min().min()
    max = df.max().max()
    return max, min

"""
Ensemble averaging plot setup
"""
plane = 'y=0'

versions = ['rbf', 'polynomial']
version = versions[0]

comp = 'w'
ensemble_method = version
comp_field = 'potential'

fill = 0
unified_color = True
shrink = 0.69
cbtit_y = -5
surface = False
save = True

if version == 'rbf':
    f = rbf2d
if version == 'polynomial':
    f = poly2

if plane == 'z=0':
    quirk = 'zzero'
if plane == 'y=0':
    quirk = 'yzero'
if plane == 'x=10':
    quirk = 'xten'
if plane == 'x=-10':
    quirk = 'xminusten'

filenamestart = len(quirk)

try:
    u_mosaic = np.loadtxt(os.path.join(ens_avg_path, '{}\{}_{}.txt'.format(plane, 'u', version)))
    v_mosaic = np.loadtxt(os.path.join(ens_avg_path, '{}\{}_{}.txt'.format(plane, 'v', version)))
    w_mosaic = np.loadtxt(os.path.join(ens_avg_path, '{}\{}_{}.txt'.format(plane, 'w', version)))
except:
    u_mosaic, v_mosaic, w_mosaic = interpolate_all(fill=fill, plane=plane, version=version, quirk=quirk, filenamestart=filenamestart, var=re.findall(r'-?\d+', plane)[0], f=f)

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------END OF ENSEMBLE AVERAGING------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

"""
------------------------------------------------------------------------------------------------------------------------
1. Numerical error metrics
      All planes: x=10, x=-10, z=0, y=0
          RBF
          2D Polynomial interpolation             --->         num_metrics.txt
          Potential flow
          CFD
------------------------------------------------------------------------------------------------------------------------
"""
error_metrics_path = os.path.join(data_path, 'num_error_metrics')


def num_error_metrics(method, field_u, field_v, field_w):

    """
    mean
    """
    # u
    u_mean = field_u.mean()
    # v
    v_mean = field_v.mean()
    # w
    w_mean = field_w.mean()

    """
    max/min
    """
    # u
    u_max = field_u.max()
    u_min = field_u.min()
    # v
    v_max = field_v.max()
    v_min = field_v.min()
    # w
    w_max = field_w.max()
    w_min = field_w.min()
    """
    std
    """
    # u
    u_std = field_u.std()
    # v
    v_std = field_v.std()
    # w
    w_std = field_w.std()

    #names = ['u', 'v', 'w']
    means = [u_mean, v_mean, w_mean]
    max = [u_max, v_max, w_max]
    min = [u_min, v_min, w_min]
    std = [u_std, v_std, w_std]

    all_met = [means, max, min, std]
    all_met_array = np.vstack(all_met)

    np.savetxt(os.path.join(error_metrics_path, '_' + method + '.txt'), all_met_array.T)


num_error_metrics(method='rbf', field_u=u_mosaic, field_v=v_mosaic, field_w=w_mosaic)

"""
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
2. Field subtraction
        - potential flow file
        - CFD flow file
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
"""

# Paths
plane_path = os.path.join(ens_avg_path, plane)
comp_field_path = os.path.join(data_path, 'comparison_fields')
sub_field_path = os.path.join(data_path, 'subtracted_fields')


def subtraction(plane, comp, ensemble_method, comparison_field):

    ens_field = np.loadtxt(os.path.join(plane_path, '{}_{}.txt'.format(comp, ensemble_method)))
    try:
        comp_field = np.loadtxt(os.path.join(os.path.join(os.path.join(comp_field_path, comparison_field), plane),
                                             '{}_{}.txt'.format(comparison_field, comp)))
    except:
        if plane is 'y=0' or 'z=0':
            comp_field = np.zeros((30, 40))
        else:
            comp_field = np.zeros((30, 30))

    subtracted_field = ens_field - comp_field

    dif_field = comp + '_' + ensemble_method + '_vs_' + comparison_field
    np.savetxt(os.path.join(sub_field_path, '{}.txt'.format(dif_field)), subtracted_field)


subtraction(plane=plane, comp=comp, ensemble_method=ensemble_method, comparison_field=comp_field)

"""
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
3. Plotting subtracted field
        - subtracted fields
        
diff_field = np.loadtxt(os.path.join(os.path.join(os.path.join(comp_field_path, comp_field), plane), '{}_{}.txt'.format(comp_field, comp)))
diff_field = w_mosaic
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
"""
diff_field = np.loadtxt(os.path.join(sub_field_path, '{}_{}_vs_{}.txt'.format(comp, ensemble_method, comp_field)))

# Fields in plot
ens_field_title = 'RBF'
comp_field_title = 'Potential Flow'

# Figure setup
fig = mplPlotter(light=True).setup2d(figsize=(20, 6))

y_ticks = 4
x_ticks = 5 if plane == 'y=0' or plane == 'z=0' else 4
degree = 2
tsize=25
axsize = 25
pad = 15
tit_y = 1.05
cbtit_size = 15
fillsphere = True
aspect = 1

if plane == 'z=0' or plane == 'y=0':
    x_bounds = [0, 40]
    y_bounds = [0, 30]
    if version == 'polynomial':
        values = True
else:
    x_bounds = [0, 30]
    y_bounds = [0, 30]
    if version == 'polynomial' and plane == 'x=-10':
        values = False

if unified_color is True:
    values = True

# Field subtraction
values = False

if values is True:
    if plane == 'x=-10':
        actualmax1 = 12
        actualmin1 = 5

        actualmax2 = 2
        actualmin2 = -2

        actualmax3 = 2.5
        actualmin3 = -2.5
    if plane == 'x=10':
        actualmax1 = 15
        actualmin1 = -6

        actualmax2 = 6
        actualmin2 = -6

        actualmax3 = 5
        actualmin3 = -5
    if plane == 'y=0' or plane == 'z=0':
        actualmax1 = 14.5
        actualmin1 = -3

        actualmax2 = 6
        actualmin2 = -6

        actualmax3 = 6
        actualmin3 = -6
else:
    actualmax1 = None
    actualmin1 = None

    actualmax2 = None
    actualmin2 = None

    actualmax3 = None
    actualmin3 = None



"""
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
4. CFD field plot
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
"""
cfd_field_path = os.path.join(os.path.join(comp_field_path, 'CFD'), plane)

# Figure setup
fig = mplPlotter(light=True).setup2d(figsize=(8, 8))

y_ticks = 4
x_ticks = 5 if plane == 'y=0' or plane == 'z=0' else 4
degree = 2
tsize=25
axsize = 25
pad = 15
tit_y = 1.05
cbtit_size = 15
fillsphere = True
aspect = 1

if plane == 'z=0' or plane == 'y=0':
    x_bounds = [0, 40]
    y_bounds = [0, 30]
    if version == 'polynomial':
        values = True
else:
    x_bounds = [0, 30]
    y_bounds = [0, 30]
    if version == 'polynomial' and plane == 'x=-10':
        values = False

if unified_color is True:
    values = True

# CFD field
values = False

if values is True:
    if plane == 'x=-10':
        actualmax1 = 12
        actualmin1 = 5

        actualmax2 = 2
        actualmin2 = -2

        actualmax3 = 2.5
        actualmin3 = -2.5
    if plane == 'x=10':
        actualmax1 = 15
        actualmin1 = -6

        actualmax2 = 6
        actualmin2 = -6

        actualmax3 = 5
        actualmin3 = -5
    if plane == 'y=0' or plane == 'z=0':
        actualmax1 = 14.5
        actualmin1 = -3

        actualmax2 = 6
        actualmin2 = -6

        actualmax3 = 6
        actualmin3 = -6
else:
    actualmax1 = None
    actualmin1 = None

    actualmax2 = None
    actualmin2 = None

    actualmax3 = None
    actualmin3 = None

"""
u
"""
comp = 'u'

u_mosaic = np.loadtxt(os.path.join(cfd_field_path, '{}_{}.txt'.format(comp_field, comp)))

ax1 = mplPlotter(fig=fig, shape_and_position=131).heatmap(array=mosaic, resize_axes=True, aspect=aspect,
                                                          tick_ndecimals=1,
                                                          xresize_pad=0, yresize_pad=0,
                                                          x_bounds=x_bounds,
                                                          y_bounds=y_bounds,
                                                          cb_vmax=actualmax1 if not isinstance(actualmax1, type(None)) else find_real_extremes(mosaic)[0],
                                                          cb_vmin=actualmin1 if not isinstance(actualmin1, type(None)) else find_real_extremes(mosaic)[1],
                                                          plot_title=r'$\mathit{' + comp + '}$' + ' in {} plane'.format(plane),
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
                                                          x_label='x $[cm]$', y_label='z $[cm]$',
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
                                                          cb_nticks=5
                                                          )

"""
v
"""
comp = 'v'

try:
    v_mosaic = np.loadtxt(os.path.join(cfd_field_path, '{}_{}.txt'.format(comp_field, comp)))
except:
    if plane is 'y=0' or 'z=0':
        v_mosaic = np.zeros((40, 30))
    else:
        v_mosaic = np.zeros((30, 30))

ax2 = mplPlotter(fig=fig, shape_and_position=132).heatmap(array=mosaic, resize_axes=True, aspect=aspect,
                                                          tick_ndecimals=1,
                                                          xresize_pad=0, yresize_pad=0,
                                                          x_bounds=x_bounds,
                                                          y_bounds=y_bounds,
                                                          plot_title=r'$\mathit{' + comp + '}$' + ' in {} plane'.format(plane),
                                                          color_bar=True,
                                                          cb_vmax=actualmax2 if not isinstance(actualmax2, type(None)) else find_real_extremes(mosaic)[0],
                                                          cb_vmin=actualmin2 if not isinstance(actualmin2, type(None)) else find_real_extremes(mosaic)[1],
                                                          cb_axis_labelpad=10,
                                                          title_size=tsize,
                                                          custom_x_ticklabels=(-20, 20) if plane == 'y=0' or plane == 'z=0' else (-15, 15),
                                                          custom_y_ticklabels= (-15, 15),
                                                          title_y=tit_y,
                                                          cb_top_title_x=-1,
                                                          x_tick_number=x_ticks,
                                                          y_tick_number=y_ticks,
                                                          more_subplots_left=True,
                                                          shrink=shrink,
                                                          cb_top_title=True,
                                                          cb_top_title_pad=cbtit_y,
                                                          cb_title='{} $[m/s]$'.format(comp),
                                                          cb_title_weight='bold',
                                                          cb_top_title_y=1.1,
                                                          cb_title_size=cbtit_size,
                                                          cb_nticks=5
                                                          )

"""
w
"""
comp = 'w'

try:
    w_mosaic = np.loadtxt(os.path.join(cfd_field_path, '{}_{}.txt'.format(comp_field, comp)))
except:
    if plane is 'y=0' or 'z=0':
        w_mosaic = np.zeros((40, 30))
    else:
        w_mosaic = np.zeros((30, 30))

ax3 = mplPlotter(fig=fig, shape_and_position=133).heatmap(array=mosaic, resize_axes=True, aspect=aspect,
                                                          tick_ndecimals=1,
                                                          xresize_pad=0, yresize_pad=0,
                                                          x_bounds=x_bounds,
                                                          y_bounds=y_bounds,
                                                          plot_title=r'$\mathit{' + comp + '}$' + ' in {} plane'.format(plane),
                                                          color_bar=True,
                                                          cb_vmax=actualmax3 if not isinstance(actualmax3, type(None)) else find_real_extremes(mosaic)[0],
                                                          cb_vmin=actualmin3 if not isinstance(actualmin3, type(None)) else find_real_extremes(mosaic)[1],
                                                          cb_axis_labelpad=10,
                                                          title_size=tsize,
                                                          title_y=tit_y,
                                                          cb_top_title_x=-1,
                                                          x_tick_number=x_ticks,
                                                          y_tick_number=y_ticks,
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
                                                          cb_nticks=5
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
    plt.savefig(os.path.join(img_path, '3D{}_Ensemble_Averaging_{}.png'.format(version, plane)),
                dpi=150)

"""
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
5. Comparison plots
    - PIV-comparison-subtraction
        - potential flow
            - u
            - w
        - CFD
            - u
            - w
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
"""


def get_comp_field():
    return np.loadtxt(os.path.join(os.path.join(os.path.join(comp_field_path, comp_field), plane), '{}_{}.txt'.format(comp_field, comp)))


def get_diff_field():
    return np.loadtxt(os.path.join(sub_field_path, '{}_{}_vs_{}.txt'.format(comp, ensemble_method, comp_field)))

"""
u
"""
comp = 'u'

try:
    u_mosaic = np.loadtxt(os.path.join(sub_field_path, '{}_{}_vs_{}.txt'.format(comp, ensemble_method, comp_field)))
except:
    subtraction(plane=plane, comp=comp, ensemble_method=ensemble_method, comparison_field=comp_field)
    u_mosaic = np.loadtxt(os.path.join(sub_field_path, '{}_{}_vs_{}.txt'.format(comp, ensemble_method, comp_field)))

ax1 = mplPlotter(fig=fig, shape_and_position=131).heatmap(array=u_mosaic, resize_axes=True, aspect=aspect,
                                                          tick_ndecimals=1,
                                                          xresize_pad=0, yresize_pad=0,
                                                          x_bounds=x_bounds,
                                                          y_bounds=y_bounds,
                                                          cb_vmax=actualmax1 if not isinstance(actualmax1, type(None)) else find_real_extremes(u_mosaic)[0],
                                                          cb_vmin=actualmin1 if not isinstance(actualmin1, type(None)) else find_real_extremes(u_mosaic)[1],
                                                          plot_title=r'$\mathit{' + comp + '}$' + ' in {} plane'.format(plane),
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
                                                          x_label='x $[cm]$', y_label='z $[cm]$',
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
                                                          cb_nticks=5
                                                          )

"""
v
"""
comp = 'v'

try:
    v_mosaic = np.loadtxt(os.path.join(sub_field_path, '{}_{}_vs_{}.txt'.format(comp, ensemble_method, comp_field)))
except:
    subtraction(plane=plane, comp=comp, ensemble_method=ensemble_method, comparison_field=comp_field)
    v_mosaic = np.loadtxt(os.path.join(sub_field_path, '{}_{}_vs_{}.txt'.format(comp, ensemble_method, comp_field)))

ax2 = mplPlotter(fig=fig, shape_and_position=132).heatmap(array=v_mosaic, resize_axes=True, aspect=aspect,
                                                          tick_ndecimals=1,
                                                          xresize_pad=0, yresize_pad=0,
                                                          x_bounds=x_bounds,
                                                          y_bounds=y_bounds,
                                                          plot_title=r'$\mathit{' + comp + '}$' + ' in {} plane'.format(plane),
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
                                                          y_tick_number=y_ticks,
                                                          more_subplots_left=True,
                                                          shrink=shrink,
                                                          cb_top_title=True,
                                                          cb_top_title_pad=cbtit_y,
                                                          cb_title='{} $[m/s]$'.format(comp),
                                                          cb_title_weight='bold',
                                                          cb_top_title_y=1.1,
                                                          cb_title_size=cbtit_size,
                                                          cb_nticks=5
                                                          )

"""
w
"""
comp = 'w'

try:
    w_mosaic = np.loadtxt(os.path.join(sub_field_path, '{}_{}_vs_{}.txt'.format(comp, ensemble_method, comp_field)))
except:
    subtraction(plane=plane, comp=comp, ensemble_method=ensemble_method, comparison_field=comp_field)
    w_mosaic = np.loadtxt(os.path.join(sub_field_path, '{}_{}_vs_{}.txt'.format(comp, ensemble_method, comp_field)))

ax3 = mplPlotter(fig=fig, shape_and_position=133).heatmap(array=w_mosaic, resize_axes=True, aspect=aspect,
                                                          tick_ndecimals=1,
                                                          xresize_pad=0, yresize_pad=0,
                                                          x_bounds=x_bounds,
                                                          y_bounds=y_bounds,
                                                          plot_title=r'$\mathit{' + comp + '}$' + ' in {} plane'.format(plane),
                                                          color_bar=True,
                                                          cb_vmax=actualmax3 if not isinstance(actualmax3, type(None)) else find_real_extremes(w_mosaic)[0],
                                                          cb_vmin=actualmin3 if not isinstance(actualmin3, type(None)) else find_real_extremes(w_mosaic)[1],
                                                          cb_axis_labelpad=10,
                                                          title_size=tsize,
                                                          title_y=tit_y,
                                                          cb_top_title_x=-1,
                                                          x_tick_number=x_ticks,
                                                          y_tick_number=y_ticks,
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
                                                          cb_nticks=5
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
    plt.savefig(os.path.join(img_path, 'Field_Subtraction_{}_vs_{}.png'.format(version, comp_field)),
                dpi=150)

"""
Numerical error metrics
"""

def max_min_avg_std():
    data = np.array([['Maximum'],
                     ['u: ', u_mosaic.max()],
                     ['v: ', v_mosaic.max()],
                     ['w: ', w_mosaic.max()],
                     ['Minimum'],
                     ['u: ', u_mosaic.min()],
                     ['v: ', v_mosaic.min()],
                     ['w: ', w_mosaic.min()],
                     ['Mean'],
                     ['u: ', u_mosaic.mean()],
                     ['v: ', v_mosaic.mean()],
                     ['w: ', w_mosaic.mean()],
                     ['Standard deviation'],
                     ['u: ', u_mosaic.std()],
                     ['v: ', v_mosaic.std()],
                     ['w: ', w_mosaic.std()]]
                    )
    np.savetxt(os.path.join(os.path.join(data_analysis, r'data\num_error_metrics\field_subtraction'), ensemble_method + '_vs_' + comp_field + '.txt'), data, fmt='%s')


max_min_avg_std()

plt.show()
