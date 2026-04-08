from os.path import isfile
import torch
from types import GeneratorType as generator

import numpy as np
import csv
from matplotlib import pyplot as plt
from setup import *

def load_array_if_path(var, load_as_numpy=True):
    """If var is a string and load_as_numpy is True, this function loads the array writen at the path indicated by var.
    Otherwise it simply returns var as it is."""
    if (isinstance(var, str)) & load_as_numpy:
        assert isfile(var), 'No such path: %s' % var
        var = np.load(var)
    return var

def worker_init_fn(wid):
    np.random.seed(np.mod(torch.utils.data.get_worker_info().seed, 2**32-1))

def create_dir(results_dir, subdirs=None):
    if subdirs is None:
        subdirs = ['checkpoints', 'results']

    if not exists(results_dir):
        for sd in subdirs:
            makedirs(join(results_dir, sd))
    else:
        for sd in subdirs:
            if not exists(join(results_dir, sd)):
                makedirs(join(results_dir, sd))
def make_sequence(input, n=None, crop=True, *args, **kwargs):
    """Ensure that the input is a sequence and pad/crop if necessary.
    Parameters
    ----------
    input : scalar or sequence or generator
        Input argument(s).
    n : int, optional
        Target length.
    crop : bool, default=True
        Crop input sequence if longer than `n`.
    default : optional
        Default value to pad with.
        If not provided, replicate the last value.
    Returns
    -------
    output : list or tuple or generator
        Output arguments.
    """
    default = None
    has_default = False
    if len(args) > 0:
        default = args[0]
        has_default = True
    elif 'default' in kwargs.keys():
        default = kwargs['default']
        has_default = True

    if isinstance(input, generator):
        # special case for generators
        def make_gen():
            last = None
            i = None
            for i, elem in input:
                if crop and (i == n):
                    return
                last = elem
                yield elem
            if i is None:
                if n is None:
                    return
                if not has_default:
                    raise ValueError('Empty sequence')
                last = default
            for j in range(i + 1, n):
                yield last

        return make_gen()
    else:
        # generic case -> induces a copy
        if not isinstance(input, (list, tuple, range)):
            input = [input]
        return_type = type(input) if isinstance(input, (list, tuple)) else list
        input = list(input)
        if len(input) == 0 and n is not None and not has_default:
            raise ValueError('Empty sequence')
        if n is not None:
            if crop:
                input = input[:min(n, len(input))]
            if not has_default:
                default = input[-1]
            input += [default] * max(0, n - len(input))
        return return_type(input)


def make_list(*args, **kwargs):
    """Ensure that the input is a list and pad/crop if necessary.
    Parameters
    ----------
    input : scalar or sequence generator
        Input argument(s).
    n : int, optional
        Target length.
    crop : bool, default=True
        Crop input sequence if longer than `n`.
    default : optional
        Default value to pad with.
        If not provided, replicate the last value.
    Returns
    -------
    output : list
        Output arguments.
    """
    return [elem for elem in make_sequence(*args, **kwargs)]

def convert_to_numpy(data, dtype=None):
    """
    Utility to convert the input data to a numpy array. If passing a dictionary, list or tuple,
    recursively check every item and convert it to numpy array.

    Args:
        data: input data can be PyTorch Tensor, numpy array, list, dictionary, int, float, bool, str, etc.
            will convert Tensor, Numpy array, float, int, bool to numpy arrays, strings and objects keep the original.
            for dictionary, list or tuple, convert every item to a numpy array if applicable.
        dtype: target data type when converting to numpy array.
        wrap_sequence: if `False`, then lists will recursively call this function.
            E.g., `[1, 2]` -> `[array(1), array(2)]`. If `True`, then `[1, 2]` -> `array([1, 2])`.
        safe: if `True`, then do safe dtype convert when intensity overflow. default to `False`.
            E.g., `[256, -12]` -> `[array(0), array(244)]`. If `True`, then `[256, -12]` -> `[array(255), array(0)]`.
    """
    if isinstance(data, torch.Tensor):
        data = np.asarray(data.detach().to(device="cpu").numpy())

    elif isinstance(data, (np.ndarray, float, int, bool)):
        # Convert into a contiguous array first if the current dtype's size is smaller than the target dtype's size.
        # This help improve the performance because (convert to contiguous array) -> (convert dtype) is faster
        # than (convert dtype) -> (convert to contiguous array) when src dtype (e.g., uint8) is smaller than
        # target dtype(e.g., float32) and we are going to convert it to contiguous array anyway later in this
        # method.
        if isinstance(data, np.ndarray) and data.ndim > 0 and data.dtype.itemsize < np.dtype(dtype).itemsize:
            data = np.ascontiguousarray(data)
        data = np.asarray(data, dtype=dtype)
    elif isinstance(data, list):
        return [convert_to_numpy(i, dtype=dtype) for i in data]
    elif isinstance(data, tuple):
        return tuple(convert_to_numpy(i, dtype=dtype) for i in data)
    elif isinstance(data, dict):
        return {k: convert_to_numpy(v, dtype=dtype) for k, v in data.items()}

    if isinstance(data, np.ndarray) and data.ndim > 0:
        data = np.ascontiguousarray(data)

    return data


color_code = ['b', 'r', 'k', 'g', 'm', 'y', 'c', 'pink']
def plot_results(filepath, keys=None, show=False, restrict_ylim=False):
    past_backend = plt.get_backend()
    plt.switch_backend('Agg')

    if keys is None:
        keys = ['loss']
    x_axis = []
    x_axis_val = []
    starting_epoch = []
    n_epochs = -1
    results_dict = {'Train': {k: [] for k in keys}, 'Validation': {k: [] for k in keys}}
    with open(filepath, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for it_row, row in enumerate(csvreader):
            if row['Phase'] == 'Train':
                x_axis.append(it_row + 1)
            else:
                x_axis_val.append(it_row + 1)

            if int(row['epoch']) > n_epochs:
                n_epochs = int(row['epoch'])
                starting_epoch.append(it_row)

            for k in keys:
                try:
                    results_dict[row['Phase']][k].append(float(row[k]))
                except:
                    n_iterations = it_row - starting_epoch[-1]
                    results_dict[row['Phase']][k].append(np.mean(results_dict['Train'][k][-n_iterations:]))

    # n_iter = len(results_dict['Train'][keys[0]])
    # n_iter_per_epoch = 1.0 * n_iter / n_epochs

    # x_axis = np.arange(0, n_iter )
    # x_axis_val = np.arange(0, n_iter, n_iter_per_epoch)
    delta_epoch = int(np.ceil(n_epochs/100)*10)
    plt.figure()
    for k in keys:
        it_color_code = 0
        if results_dict['Train'][k]:
            plt.plot(x_axis, results_dict['Train'][k], color=color_code[it_color_code], marker='*')
            it_color_code +=1

        if results_dict['Validation'][k]:
            plt.plot(x_axis_val, results_dict['Validation'][k], color=color_code[it_color_code], marker='*')
            it_color_code +=1

        if restrict_ylim:
            # ymin, ymax = np.min(results_dict['Validation'][k]), np.max(results_dict['Validation'][k])
            ymin, ymax = np.percentile(results_dict['Train'][k],1), np.percentile(results_dict['Train'][k],99)

            if ymin < 0:
                ymin = ymin*1.5
            else:
                ymin=ymin*0.5

            if ymax < 0:
                ymax = ymax*0.5
            else:
                ymax=ymax*1.5

            plt.ylim([ymin,ymax])


        plt.xticks(starting_epoch[::delta_epoch], np.arange(0,n_epochs,delta_epoch))
        plt.xlabel('Number of epochs')
        plt.ylabel(k)
        plt.grid()

        if show:
            plt.show()
        else:
            plt.savefig(join(dirname(filepath), k + '_results.png'))

        plt.close()

    plt.switch_backend(past_backend)
