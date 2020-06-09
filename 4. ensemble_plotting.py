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
data_path = os.path.join(data_analysis, r'data')
img_path = os.path.join(data_analysis, 'images')
bin_path = os.path.join(data_path, 'bins')
uncertainty_path = os.path.join(data_path, 'uncertainty_mean_estimate')
ens_avg_path = os.path.join(data_path, 'velocity_ensemble_averaged')
num_metrics_path = os.path.join(data_path, 'num_error_metrics')
int_comp_path = os.path.join(data_path, 'interpolation_comparison')

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
def rbf(x, y, z, x_new, y_new):
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

def rbf2(x, y, z, x_new, y_new):
    try:
        """
        Optimal values
            x=-10:      500, 0.02
            x=10:       500, 0.02
            y=0:        500, 0.02
            z=0:        500, 0.02
        """
        rbf = Rbf(x, y, z, epsilon=1, smooth=0)
    except:
        return False
    z_new = rbf(x_new, y_new)
    return z_new

def rbf3(x, y, z, x_new, y_new):
    try:
        """
        Optimal values
            x=-10:      500, 0.02
            x=10:       500, 0.02
            y=0:        500, 0.02
            z=0:        500, 0.02
        """
        rbf = Rbf(x, y, z, epsilon=1, smooth=0, function='gaussian')
    except:
        return False
    z_new = rbf(x_new, y_new)
    return z_new

def rbf4(x, y, z, x_new, y_new):
    try:
        """
        Optimal values
            x=-10:      500, 0.02
            x=10:       500, 0.02
            y=0:        500, 0.02
            z=0:        500, 0.02
        """
        rbf = Rbf(x, y, z, epsilon=500, smooth=0.02, function='gaussian')
    except:
        return False
    z_new = rbf(x_new, y_new)
    return z_new

def rbf5(x, y, z, x_new, y_new):
    try:
        """
        Optimal values
            x=-10:      500, 0.02
            x=10:       500, 0.02
            y=0:        500, 0.02
            z=0:        500, 0.02
        """
        rbf = Rbf(x, y, z, epsilon=500, smooth=0.02, function='inverse')
    except:
        return False
    z_new = rbf(x_new, y_new)
    return z_new

def rbf6(x, y, z, x_new, y_new):
    try:
        """
        Optimal values
            x=-10:      500, 0.02
            x=10:       500, 0.02
            y=0:        500, 0.02
            z=0:        500, 0.02
        """
        rbf = Rbf(x, y, z, smooth=0.02, function='cubic')
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

def clean_field():
    # u
    u_correction_pc = (len(u_mosaic[np.where(u_mosaic > 14.5)]) + len(u_mosaic[np.where(u_mosaic < -3)])) / (30*40)
    u_max_threshold = u_mosaic > 14.5
    u_mosaic[u_max_threshold] = 0
    u_min_threshold = u_mosaic < -3
    u_mosaic[u_min_threshold] = 0
    # v
    v_correction_pc = (len(v_mosaic[np.where(v_mosaic > 6)]) + len(v_mosaic[np.where(v_mosaic < -6)])) / (30*40)
    v_max_threshold = v_mosaic > 6
    v_mosaic[v_max_threshold] = 0
    v_min_threshold = v_mosaic < -6
    v_mosaic[v_min_threshold] = 0
    #w
    w_correction_pc = (len(w_mosaic[np.where(w_mosaic > 6)]) + len(w_mosaic[np.where(w_mosaic < -6)])) / (30*40)
    w_max_threshold = w_mosaic > 6
    w_mosaic[w_max_threshold] = 0
    w_min_threshold = w_mosaic < -6
    w_mosaic[w_min_threshold] = 0

    print('u correction percent: {}'.format(u_correction_pc))
    print('v correction percent: {}'.format(v_correction_pc))
    print('w correction percent: {}'.format(w_correction_pc))

"""
Ensemble averaging plot setup
"""

fill = 0
plane = 'y=0'
versions = ['rbf', 'rbf2', 'rbf3', 'rbf4', 'rbf5', 'rbf6', 'polynomial']
version = versions[6]
unified_color = True
shrink = 0.69
cbtit_y = -5
surface = True
save = False

if version is 'rbf':
    f = rbf
    clean = False
if version is 'rbf2':
    f = rbf2
    clean = True
if version is 'rbf3':
    f = rbf3
    clean = True
if version is 'rbf4':
    f = rbf4
    clean = False
if version is 'rbf5':
    f = rbf5
    clean = False
if version is 'rbf6':
    f = rbf5
    clean = False
if version is 'polynomial':
    f = poly2
    clean = False

if plane is 'z=0':
    quirk = 'zzero'
if plane is 'y=0':
    quirk = 'yzero'
if plane is 'x=10':
    quirk = 'xten'
if plane is 'x=-10':
    quirk = 'xminusten'

filenamestart = len(quirk)

try:
    u_mosaic = np.loadtxt(os.path.join(ens_avg_path, '{}\{}_{}.txt'.format(plane, 'u', version)))
    v_mosaic = np.loadtxt(os.path.join(ens_avg_path, '{}\{}_{}.txt'.format(plane, 'v', version)))
    w_mosaic = np.loadtxt(os.path.join(ens_avg_path, '{}\{}_{}.txt'.format(plane, 'w', version)))
except:
    u_mosaic, v_mosaic, w_mosaic = interpolate_all(fill=fill, plane=plane, version=version, quirk=quirk, filenamestart=filenamestart, var=re.findall(r'-?\d+', plane)[0], f=f)

if clean is True:
    clean_field()

"""
Plot
"""

# Subplot setup
fig = mplPlotter(light=True).setup2d(figsize=(20, 5))

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

print(x_bounds)

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

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# -------------Surface or 2D plot-------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if surface is True:
    y_ticks = 4
    z_ticks = 6
    x_ticks = 5 if plane == 'y=0' or plane == 'z=0' else 4
    degree = 2
    tsize = 25
    axsize = 15
    tick_size = 10
    pad = 15
    tit_y = 1.05
    cbtit_size = 15
    fillsphere = True
    aspect = 1
    """
    u
    """
    comp = 'u'
    mosaic = u_mosaic
    lighting = False

    x = np.linspace(-20, 20, 40)
    y = np.linspace(-15, 15, 30)
    x, y = np.meshgrid(x, y)
    mplPlotter(fig=fig, shape_and_position=131).surface3d(x=x, y=y, z=mosaic, rstride=1, cstride=1,
                                                          plot_title=r'$\mathit{' + comp + '}$' + ' in {} plane'.format(
                                                              plane),
                                                          color_bar=True,
                                                          cb_vmax=actualmax1 if not isinstance(actualmax1,
                                                                                               type(None)) else
                                                          find_real_extremes(mosaic)[0],
                                                          cb_vmin=actualmin1 if not isinstance(actualmin1,
                                                                                               type(None)) else
                                                          find_real_extremes(mosaic)[1],
                                                          cb_axis_labelpad=10,
                                                          title_size=tsize,
                                                          title_y=tit_y,
                                                          xaxis_label_size=axsize,
                                                          yaxis_label_size=axsize,
                                                          zaxis_label_size=axsize+5,
                                                          xaxis_bold=True,
                                                          yaxis_bold=True,
                                                          zaxis_bold=True,
                                                          tick_label_size=tick_size,
                                                          x_tick_number=x_ticks,
                                                          y_tick_number=y_ticks,
                                                          z_tick_number=z_ticks,
                                                          zaxis_labelpad=20,
                                                          z_label='${}$'.format(comp) + ' ' + '$[m/s]$',
                                                          cb_top_title_x=-1,
                                                          more_subplots_left=True,
                                                          shrink=shrink,
                                                          cb_top_title=True,
                                                          cb_top_title_pad=cbtit_y,
                                                          cb_title='{} $[m/s]$'.format(comp),
                                                          cb_title_weight='bold',
                                                          cb_top_title_y=1.1,
                                                          cb_title_size=cbtit_size,
                                                          cb_nticks=5,
                                                          lighting=lighting
                                                          )

    """
    v
    """
    comp = 'v'
    mosaic = v_mosaic

    x = np.linspace(-20, 20, 40)
    y = np.linspace(-15, 15, 30)
    x, y = np.meshgrid(x, y)
    mplPlotter(fig=fig, shape_and_position=132).surface3d(x=x, y=y, z=mosaic, rstride=1, cstride=1,
                                                          plot_title=r'$\mathit{' + comp + '}$' + ' in {} plane'.format(
                                                              plane),
                                                          color_bar=True,
                                                          cb_vmax=actualmax2 if not isinstance(actualmax2,
                                                                                               type(None)) else
                                                          find_real_extremes(mosaic)[0],
                                                          cb_vmin=actualmin2 if not isinstance(actualmin2,
                                                                                               type(None)) else
                                                          find_real_extremes(mosaic)[1],
                                                          cb_axis_labelpad=10,
                                                          title_size=tsize,
                                                          title_y=tit_y,
                                                          xaxis_label_size=axsize,
                                                          yaxis_label_size=axsize,
                                                          zaxis_label_size=axsize+5,
                                                          xaxis_bold=True,
                                                          yaxis_bold=True,
                                                          zaxis_bold=True,
                                                          tick_label_size=tick_size,
                                                          x_tick_number=x_ticks,
                                                          y_tick_number=y_ticks,
                                                          z_tick_number=z_ticks,
                                                          zaxis_labelpad=20,
                                                          z_label='${}$'.format(comp) + ' ' + '$[m/s]$',
                                                          cb_top_title_x=-1,
                                                          more_subplots_left=True,
                                                          shrink=shrink,
                                                          cb_top_title=True,
                                                          cb_top_title_pad=cbtit_y,
                                                          cb_title='{} $[m/s]$'.format(comp),
                                                          cb_title_weight='bold',
                                                          cb_top_title_y=1.1,
                                                          cb_title_size=cbtit_size,
                                                          cb_nticks=5,
                                                          lighting=lighting
                                                          )

    """
    w
    """
    comp = 'w'
    mosaic = w_mosaic

    x = np.linspace(-20, 20, 40)
    y = np.linspace(-15, 15, 30)
    x, y = np.meshgrid(x, y)
    mplPlotter(fig=fig, shape_and_position=133).surface3d(x=x, y=y, z=mosaic, rstride=1, cstride=1,
                                                          plot_title=r'$\mathit{' + comp + '}$' + ' in {} plane'.format(
                                                              plane),
                                                          color_bar=True,
                                                          cb_vmax=actualmax3 if not isinstance(actualmax3,
                                                                                               type(None)) else
                                                          find_real_extremes(mosaic)[0],
                                                          cb_vmin=actualmin3 if not isinstance(actualmin3,
                                                                                               type(None)) else
                                                          find_real_extremes(mosaic)[1],
                                                          cb_axis_labelpad=10,
                                                          title_size=tsize,
                                                          title_y=tit_y,
                                                          xaxis_label_size=axsize,
                                                          yaxis_label_size=axsize,
                                                          zaxis_label_size=axsize+5,
                                                          xaxis_bold=True,
                                                          yaxis_bold=True,
                                                          zaxis_bold=True,
                                                          tick_label_size=tick_size,
                                                          x_tick_number=x_ticks,
                                                          y_tick_number=y_ticks,
                                                          z_tick_number=z_ticks,
                                                          zaxis_labelpad=20,
                                                          z_label='${}$'.format(comp) + ' ' + '$[m/s]$',
                                                          cb_top_title_x=-1,
                                                          more_subplots_left=True,
                                                          shrink=shrink,
                                                          cb_top_title=True,
                                                          cb_top_title_pad=cbtit_y,
                                                          cb_title='{} $[m/s]$'.format(comp),
                                                          cb_title_weight='bold',
                                                          cb_top_title_y=1.1,
                                                          cb_title_size=cbtit_size,
                                                          cb_nticks=5,
                                                          lighting=lighting
                                                          )

    if save is True:
        if version == '1.0':
            plt.savefig(r'C:\Users\sownd\Documents\GitHub\Data_Analysis\3DRBF_Ensemble_Averaging_{}.png'.format(plane),
                        dpi=150)
        if version == 'polynomial':
            plt.savefig(r'C:\Users\sownd\Documents\GitHub\Data_Analysis\3DPolynomial_Ensemble_Averaging_{}.png'.format(plane),
                            dpi=150)

else:
    """
    u
    """
    comp = 'u'
    mosaic = u_mosaic

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
    mosaic = v_mosaic

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
    mosaic = w_mosaic

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
    np.savetxt(os.path.join(os.path.join(data_analysis, r'data\num_error_metrics\interpolation_comparison'), '{}.txt'.format(version)), data, fmt='%s')

max_min_avg_std()

plt.show()
