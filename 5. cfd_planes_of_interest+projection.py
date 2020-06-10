import pandas as pd
import os.path

from shutil import copyfile
from itertools import chain
from os import listdir


def z_0():
    df_o = None
    df_i = None

    for x in chain(range(-20, 20)):
        for y in range(-15, 15):
            files = []

            z = 0
            if os.path.isfile(r'C:\Users\dsanc\Python\Python36\100bins\{}_{}_{}.csv'.format(x, y, z)):
                df_o = pd.read_csv(r'C:\Users\dsanc\Python\Python36\100bins\{}_{}_{}.csv'.format(x, y, z), header=0)
            z = -1
            if os.path.isfile(r'C:\Users\dsanc\Python\Python36\100bins\{}_{}_{}.csv'.format(x, y, z)):
                df_i = pd.read_csv(r'C:\Users\dsanc\Python\Python36\100bins\{}_{}_{}.csv'.format(x, y, z), header=0)

            if not isinstance(df_o, type(None)):
                files.append(df_o)
            if not isinstance(df_i, type(None)):
                files.append(df_i)

            if not isinstance(df_o, type(None)) or not isinstance(df_i, type(None)):
                df = pd.concat(files)
            else:
                df = None

            if not isinstance(df, type(None)):
                df.to_csv(
                    r'C:\Users\dsanc\Documents\GitHub\Data_Analysis\data\bins_cfd\z=0\[0, 1)\{}_{}_{}.csv'.format(x, y,
                                                                                                                  z))
            df_o = None
            df_i = None


    for file in listdir(r'C:\Users\dsanc\Documents\GitHub\Data_Analysis\data\bins_cfd\z=0\[0, 1)'):

        df = pd.read_csv(r'C:\Users\dsanc\Documents\GitHub\Data_Analysis\data\bins_cfd\z=0\[0, 1)\{}'.format(file), header=0)
        mask = df['z'] < 0.005
        df = df[mask].sort_values(['z'])

        mask = df['z'] < -0.005
        df = df[mask].sort_values(['z'])

        df.to_csv(r'C:\Users\dsanc\Documents\GitHub\Data_Analysis\data\bins_cfd\z=0\[0, 0.25)\{}'.format(file))

    for file in listdir(r'C:\Users\dsanc\Documents\GitHub\Data_Analysis\data\bins_cfd\z=0\[0, 0.25)'):

        df = pd.read_csv(r'C:\Users\dsanc\Documents\GitHub\Data_Analysis\data\bins_cfd\z=0\[0, 0.25)\{}'.format(file))

        df['z'] = 0

        df.to_csv(r'C:\Users\dsanc\Documents\GitHub\Data_Analysis\data\bins_cfd\z=0\z=0_projected\{}'.format(file))


def y_0():
    df_o = None
    df_i = None
    y = 0
    for x in chain(range(-20, 20)):
        for z in range(-15, 15):
            files = []
            if os.path.isfile(r'C:\Users\dsanc\Python\Python36\100bins\{}_{}_{}.csv'.format(x, y, z)):
                copyfile(r'C:\Users\dsanc\Python\Python36\100bins\{}_{}_{}.csv'.format(x, y, z),
                         r'C:\Users\dsanc\Documents\GitHub\Data_Analysis\data\bins_cfd\y=0\[0, 1)\{}_{}_{}.csv'.format(
                             x, y, z))

            # y = 0
            # if os.path.isfile(r'C:\Users\dsanc\Python\Python36\100bins\{}_{}_{}.csv'.format(x, y, z)):
            #     df_o = pd.read_csv(r'C:\Users\dsanc\Python\Python36\100bins\{}_{}_{}.csv'.format(x, y, z), header=0)
            # y = -1
            # if os.path.isfile(r'C:\Users\dsanc\Python\Python36\100bins\{}_{}_{}.csv'.format(x, y, z)):
            #     df_i = pd.read_csv(r'C:\Users\dsanc\Python\Python36\100bins\{}_{}_{}.csv'.format(x, y, z), header=0)
            #
            # if not isinstance(df_o, type(None)):
            #     files.append(df_o)
            # if not isinstance(df_i, type(None)):
            #     files.append(df_i)
            #
            # if not isinstance(df_o, type(None)) or not isinstance(df_i, type(None)):
            #     df = pd.concat(files)
            # else:
            #     df = None
            #
            # if not isinstance(df, type(None)):
            #     df.to_csv(
            #         r'C:\Users\dsanc\Documents\GitHub\Data_Analysis\data\bins_cfd\z=0\[0, 1)\{}_{}_{}.csv'.format(x, y,
            #                                                                                                       z))
            # df_o = None
            # df_i = None


    for file in listdir(r'C:\Users\dsanc\Documents\GitHub\Data_Analysis\data\bins_cfd\y=0\[0, 1)'):

        df = pd.read_csv(r'C:\Users\dsanc\Documents\GitHub\Data_Analysis\data\bins_cfd\y=0\[0, 1)\{}'.format(file), header=0)
        mask = df['y'] < 0.010
        df = df[mask].sort_values(['y'])

        df.to_csv(r'C:\Users\dsanc\Documents\GitHub\Data_Analysis\data\bins_cfd\y=0\[0, 0.25)\{}'.format(file))

    for file in listdir(r'C:\Users\dsanc\Documents\GitHub\Data_Analysis\data\bins_cfd\y=0\[0, 0.25)'):

        df = pd.read_csv(r'C:\Users\dsanc\Documents\GitHub\Data_Analysis\data\bins_cfd\y=0\[0, 0.25)\{}'.format(file))

        df['y'] = 0

        df.to_csv(r'C:\Users\dsanc\Documents\GitHub\Data_Analysis\data\bins_cfd\y=0\y=0_projected\{}'.format(file))


def x_m10():
    x = -10
    for y in chain(range(-15, 15)):
        for z in range(-15, 15):
            if os.path.isfile(r'C:\Users\dsanc\Python\Python36\100bins\{}_{}_{}.csv'.format(x, y, z)):
                copyfile(r'C:\Users\dsanc\Python\Python36\100bins\{}_{}_{}.csv'.format(x, y, z),
                         r'C:\Users\dsanc\Documents\GitHub\Data_Analysis\data\bins_cfd\x=-10\[0, 1)\{}_{}_{}.csv'.format(
                             x, y, z))

    for file in listdir(r'C:\Users\dsanc\Documents\GitHub\Data_Analysis\data\bins_cfd\x=-10\[0, 1)'):
        df = pd.read_csv(r'C:\Users\dsanc\Documents\GitHub\Data_Analysis\data\bins_cfd\x=-10\[0, 1)\{}'.format(file),
                         header=0)
        mask = df['x'] < -0.090
        df = df[mask].sort_values(['x'])

        df.to_csv(r'C:\Users\dsanc\Documents\GitHub\Data_Analysis\data\bins_cfd\x=-10\[0, 0.25)\{}'.format(file))

    for file in listdir(r'C:\Users\dsanc\Documents\GitHub\Data_Analysis\data\bins_cfd\x=-10\[0, 0.25)'):
        df = pd.read_csv(r'C:\Users\dsanc\Documents\GitHub\Data_Analysis\data\bins_cfd\x=-10\[0, 0.25)\{}'.format(file))

        df['x'] = 0

        df.to_csv(r'C:\Users\dsanc\Documents\GitHub\Data_Analysis\data\bins_cfd\x=-10\x=-10_projected\{}'.format(file))


def x_10():
    x = 10

    for z in chain(range(-15, 15)):
        for y in range(-15, 15):
            if os.path.isfile(r'C:\Users\dsanc\Python\Python36\100bins\{}_{}_{}.csv'.format(x, y, z)):
                copyfile(r'C:\Users\dsanc\Python\Python36\100bins\{}_{}_{}.csv'.format(x, y, z),
                         r'C:\Users\dsanc\Documents\GitHub\Data_Analysis\data\bins_cfd\x=10\[0, 1)\{}_{}_{}.csv'.format(
                             x, y, z))

    for file in listdir(r'C:\Users\dsanc\Documents\GitHub\Data_Analysis\data\bins_cfd\x=10\[0, 1)'):
        df = pd.read_csv(r'C:\Users\dsanc\Documents\GitHub\Data_Analysis\data\bins_cfd\x=10\[0, 1)\{}'.format(file),
                         header=0)
        mask = df['x'] < 0.110
        df = df[mask].sort_values(['x'])

        df.to_csv(r'C:\Users\dsanc\Documents\GitHub\Data_Analysis\data\bins_cfd\x=10\[0, 0.25)\{}'.format(file))

    for file in listdir(r'C:\Users\dsanc\Documents\GitHub\Data_Analysis\data\bins_cfd\x=10\[0, 0.25)'):
        df = pd.read_csv(r'C:\Users\dsanc\Documents\GitHub\Data_Analysis\data\bins_cfd\x=10\[0, 0.25)\{}'.format(file))

        df['x'] = 0

        df.to_csv(r'C:\Users\dsanc\Documents\GitHub\Data_Analysis\data\bins_cfd\x=10\x=10_projected\{}'.format(file))


#z_0()
y_0()
#x_10()
#x_m10()