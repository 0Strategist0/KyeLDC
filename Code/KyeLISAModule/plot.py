"""
Author: Kye Emond
Date: July 8th, 2022


KyeLISA Plotting Module

Useful functions for plotting LISA data easily. 

Methods:
    plot_glitch_span: Plots the span of a glitch
    plot_glitch: Plots a segment of data containing a glitch
    heatmap_corner: Plots a marginalized cornerplot of amplitudes in a grid
    heatmap_corner_intersect: Plots a cornerplot of a cross section of amplitudes in a grid
    heatmap_corner_max: Plots a cornerplot of the cross section containing the maximum amplitude in a grid
    plot_vars: Plots each value of a dictionary in its own subplot
"""


# Imports
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as cl
import numpy as np
from numpy.linalg import inv, norm
from typing import Iterable, Union, Hashable, Callable


# Functions
def plot_glitch_span(indices:np.ndarray, x:np.ndarray=None, padding:int=50, label:bool=True, data_index:np.ndarray=None) -> None:
    """Plot the span of a glitch.
    
    Args: 
        indices (ndarray): A numpy array of indices to include in the span
        x (ndarray): A numpy array of values to be indexed to get the x-values of the span (Default: indices)
        padding (int, optional): The amount of indices on either side of the glitch that are counted as padding (Default: 50)
        data_index (ndarray, optional): Indices of the data to use (Default: all data)
    """

    # Set x if it's not given
    if x is None:
        x = np.arange(indices[-1])
    
    # Use correct data indices
    if data_index is not None:
        x = x[data_index]
    
    # Plot
    plt.axvspan(x[indices[0]], x[indices[0] + padding], label=("Glitch Padding" if label else ""), alpha=0.2, color="yellow")
    plt.axvspan(x[indices[-1] - padding], x[indices[-1]], alpha=0.2, color="yellow")
    plt.axvspan(x[indices[0] + padding], x[indices[-1] - padding], alpha=0.2, color="red", label=("Glitching Region" if label else ""))


def plot_glitch(data:Union[dict, tuple[dict, ...]], 
                indices:np.ndarray, 
                span:bool=True, 
                padding:int=50, 
                extra:int=150, 
                titles:tuple[str]=None, 
                title_keys:bool=True, 
                legend:bool=False, 
                yfunc:Callable=lambda x: x, 
                data_index:np.ndarray=None, 
                plot_kwargs:dict={}) -> None:
    """Plot a glitch in the data more easily.
    
    Args:
        data (dict | tuple[dict, ...]): A dictionary of data times with key "t" and data values with some other key, or a tuple of such dictionaries
        indices (ndarray): A numpy array of indices that are considered glitching
        span (bool, optional): A boolean indicating whether to plot the glitch span or not (Default: True)
        padding (int, optional): An integer giving the region of each glitch that was added due to padding (Default: 50)
        extra (int, optional): An integer giving the amount of extra data points to plot outside the glitch (Default: 150)
        titles (tuple[str, ...], optional): A tuple of strings of the same size as data, if it's a tuple, to be used as titles for the different data dictionaries (Default: None)
        title_keys (bool, optional): Whether to include the keys of the dictionary in the legend labels (Default: True)
        yfunc (function, optional): Function to apply to the y values before plotting (Default: lambda x: x)
        data_index (ndarray, optional): Indices of the data to use (Default: None)
        plot_kwargs (dict, optional): kwargs for the plot function (Default: {})
    """


    # Set up data
    if data_index is not None:
        data = {var: data[var][data_index] for var in data.keys()}

    # Get the range of indices before the data
    pre_indices = np.arange(max(0, indices[0] - extra), indices[0])

    # If passed a tuple of data dictionaries, do stuff
    if type(data) == tuple:
        # Get the titles
        if titles is None:
            titles = range(len(data))

        # Get the range of indices after the data
        post_indices = np.arange(indices[-1] + 1, min(len(data[0]["t"]), indices[-1] + 1 + extra))

        # Get the entire range to show
        all_indices = np.concatenate((pre_indices, indices, post_indices))

        # Iterate through and plot the data
        for index, elem in enumerate(data):
            for key in elem.keys():
                if key != "t":
                    plt.plot(elem["t"][all_indices], yfunc(elem[key][all_indices]), label=str(titles[index]) + (str(key) if title_keys else ""), **plot_kwargs)
        
        # Plot the spans of the glitch
        if span:
            plot_glitch_span(indices, data[0]["t"], padding)
        
    elif type(data) == dict:
        # Get the range of indices after the data
        post_indices = np.arange(indices[-1] + 1, min(len(data["t"]), indices[-1] + 1 + extra))

        # Get the entire range to show
        all_indices = np.concatenate((pre_indices, indices, post_indices))

        # Plot each key of the data
        for key in data.keys():
            if key != "t":
                plt.plot(data["t"][all_indices], yfunc(data[key][all_indices]), label=(str(key) if title_keys else ""), **plot_kwargs)
        
        # Plot the spans of the glitch
        if span:
            plot_glitch_span(indices, data["t"], padding)
    
    # Show the legend
    if legend:
        plt.legend()


def heatmap_corner(data:np.ndarray, 
                   params:np.ndarray=None, 
                   titles:Iterable[str]=None, 
                   truths:Iterable[np.ndarray]=None, 
                   truth_colors:Iterable[Union[str, tuple[float, float, float]]]=None) -> plt.Figure:
    """Generate a corner plot out of heatmaps, where the amplitudes at a given position are given by data, and the positions for each data point in the n-dimensional space are 
    given by positions. Amplitudes are determined by marginalizing over all unused parameters.

    Args:
        data (ndarray): An ndarray of amplitudes in an n-dimensional grid
        params (ndarray[ndarray, ...], optional): An ndarray of ndarrays with the same shape as data with length equal to the number of dimensions in data. Each ndarray gives the 
            parameter value for a given data point (Default: uses data indices)
        titles (Iterable[str], optional): An iterable of strings with a length equal to the data's number of dimensions. Each string corresponds to one dimension of the space 
            (Default: None)
        truths (Iterable[ndarray], optional): An iterable of ndarrays acting as points in the parameter space to be plotted over the heatmaps (Default: None)
        truth_colors (Iterable[str | tuple[double, double, double]], optional): An iterable of colors to plot the truths with (Default: None)

    Returns:
        Figure: The matplotlib figure instance for the corner plot. 
    """ 
    
    # Store the number of dimensions of the data
    ndim = data.ndim
    
    # Initialize the figure to be plotted on
    fig = plt.figure(figsize=(16, 13), constrained_layout=True)
    axes = [None] * (ndim * (ndim + 1) // 2)
    index = 0
    
    # Get the heatmap scale
    marginalized_data = np.array(sum(sum([[list(np.sum(data, axis=tuple(set(range(ndim)) - {x, y})).flatten()) for y in range(x + 1, ndim)] for x in range(ndim)], []), []))
    min_amp_2d = np.min(marginalized_data)
    max_amp_2d = np.max(marginalized_data)
    heat_scale = np.linspace(min_amp_2d, max_amp_2d, int(np.power(data.size, 1.0 / ndim)))

    # Set the parameters to the indices if none given
    if params is None:
        params = np.indices(data.shape)
    if type(params) == np.ndarray and params.shape == (*data.shape, ndim):
        params = np.array(tuple(params[..., index] for index in range(ndim)))
    

    # Data validation
    for param in params:
        assert param.shape == data.shape, "ndarrays in params should be the same shape as data"
    if titles is not None:
        assert len(titles) == ndim, "titles should have one entry for each dimension of data"
    if truths is not None:
        # Set the colors to red if none were given
        if truth_colors is None:
            truth_colors = tuple("red" for _ in truths)
        
        for truth in truths:
            assert truth.size == ndim, "ndarrays in truths should have one entry for each dimension of data"

        assert len(truth_colors) == len(truths), "truth_colors should have one entry for each value of truths"
    
    # Go through and plot each heatmap/histogram
    for x in range(ndim):
        for y in range(x, ndim):
            # Initialize a new subplot
            axes[index] = plt.subplot(ndim, ndim, 1 + x + ndim * y)
            index += 1
            
            # If x == y, plot a histogram
            if x == y:
                # Marginalize over unneeded dimensions
                amplitudes = np.sum(data, axis=tuple(set(range(ndim)) - {x}))
                # Get the positions of the amplitudes
                positions = params[tuple([x] + [0] * x + [slice(None)] + [0] * (ndim - x - 1))]
                # Plot the histogram
                plt.plot(positions, amplitudes)
                # Set x limit
                plt.xlim(positions.min(), positions.max())
                # Plot the true values
                if truths is not None:
                    for truth, color in zip(truths, truth_colors):
                        plt.axvline(truth[x], ls=":", color=color)
                if titles is not None:
                    # Add a title
                    plt.title(titles[x])
                
                
            # Otherwise plot a 2d heatmap
            else:
                # Marginalize over uneeded dimensions
                amplitudes = np.sum(data, axis=tuple(set(range(ndim)) - {x, y}))
                # Get the positions of the amplitudes
                x_positions = params[tuple([x] + [0] * min(x, y) + [slice(None)] + [0] * (max(x, y) - min(x, y) - 1) + [slice(None)] + [0] * (ndim - max(x, y) - 1))]
                y_positions = params[tuple([y] + [0] * min(x, y) + [slice(None)] + [0] * (max(x, y) - min(x, y) - 1) + [slice(None)] + [0] * (ndim - max(x, y) - 1))]
                # Plot the heatmap
                plt.contourf(x_positions, y_positions, amplitudes, heat_scale)
                # Set the x and y limits
                plt.xlim(x_positions.min(), x_positions.max())
                plt.ylim(y_positions.min(), y_positions.max())
                # Plot the true values
                if truths is not None:
                    for truth, color in zip(truths, truth_colors):
                        plt.axvline(truth[x], color=color)
                        plt.axhline(truth[y], color=color)
                        plt.plot(truth[x], truth[y], color=color, marker="o", markersize=8)
                if titles is not None:
                    # Write the axis labels
                    if y == ndim - 1:
                        plt.xlabel(titles[x])
                    if x == 0:
                        plt.ylabel(titles[y])
            
            # Remove axis ticks on the inside plots
            if y != ndim - 1:
                plt.xticks(())
            if x != 0 and x != y:
                plt.yticks(())
                
    # Place a colorbar on the side
    fig.colorbar(cm.ScalarMappable(cl.Normalize(min_amp_2d, max_amp_2d)), ax=axes, ticks=np.linspace(min_amp_2d, max_amp_2d, 10))
    
    # Return the figure
    return fig


def heatmap_corner_intersect(data:np.ndarray, 
                             params:np.ndarray=None, 
                             intersect:np.ndarray=None, 
                             intersect_color:Union[str, tuple[float, ...]]=(1.0, 1.0, 1.0), 
                             titles:Iterable[str]=None, 
                             truths:Iterable[np.ndarray]=None, 
                             truth_colors:Iterable[Union[str, tuple[float, float, float]]]=None,
                             covariance:np.ndarray=None, 
                             close_color:Union[str, tuple[float, float, float]]="lime") -> plt.Figure:
    """Generate a corner plot out of heatmaps, where the amplitudes at a given position are given by data, and the positions for each data point in the n-dimensional space are 
    given by positions. Amplitudes are determined by setting unused parameters to the values at a cross section.

    Args:
        data (ndarray): An ndarray of amplitudes in an n-dimensional grid
        params (ndarray[ndarray], optional): An iterable of ndarrays with the same shape as data with length equal to the number of dimensions in data. Each ndarray gives the 
            parameter value for a given data point. (Default: uses data indices)
        intersect (ndarray, optional): The point at which all the cross sections of the space intersect (Default: maximum value of data)
        titles (Iterable[str], optional): An iterable of strings with a length equal to the data's number of dimensions. Each string corresponds to one dimension of the space
            (Default: None)
        truths (Iterable[ndarray], optional): An iterable of ndarrays acting as points in the parameter space to be plotted over the heatmaps (Default: None)
        truth_colors (Iterable[str | tuple[double, double, double]], optional): An iterable of colors to use when plotting the truths (Default: None)
        covariance (ndarray, optional): Covariance used to calculate the "nearest" truth value. (Default: No calculation)
        close_color (Union[str, tuple[float, float, float]], optional): Color to assign to the truth value nearest the intersection.

    Returns:
        Figure: The matplotlib figure instance for the corner plot. 
    """
    
    # Store the number of dimensions of the data
    ndim = data.ndim
    
    # Initialize the figure to be plotted on
    fig = plt.figure(figsize=(16, 13), constrained_layout=True)
    axes = [None] * (ndim * (ndim + 1) // 2)
    index = 0
    
    # Get the heatmap scale
    min_amp = np.min(data)
    max_amp = np.max(data)
    intersect_position = np.nonzero(data == max_amp) if intersect is None else intersect[:, np.newaxis]
    heat_scale = np.linspace(min_amp, max_amp, int(np.power(data.size, 1.0 / ndim)))
    
    # Add the intersect to be plotted
    if intersect is not None:
        truths = (params[tuple(intersect)], *truths)
        truth_colors = (intersect_color, *truth_colors)
    
    # Set the parameters to the indices if none given
    if params is None:
        params = np.indices(data.shape)
    if type(params) == np.ndarray and params.shape == (*data.shape, ndim):
        params = np.array(tuple(params[..., index] for index in range(ndim)))
    
    # Get the inverse covariance if cov is given
    if covariance is not None:
        inv_cov = inv(covariance)
        closest = [None, np.inf, None]
    
    # Data validation
    for param in params:
        assert param.shape == data.shape, "ndarrays in params should be the same shape as data"
    if titles is not None:
        assert len(titles) == ndim, "titles should have one entry for each dimension of data"
    if truths is not None:
        # Set the colors to red if none were given
        if truth_colors is None:
            truth_colors = ["red"] * len(truths)
        else:
            truth_colors = list(truth_colors)
        
        
        for truth_index, truth in enumerate(truths):
            # Validate truths
            assert truth.size == ndim, "ndarrays in truths should have one entry for each dimension of data"
            
            # Find the closest truth
            if covariance is not None:
                current_len_sq = np.matmul((truth - params[(slice(None), *intersect_position)].squeeze()).transpose(), np.matmul(inv_cov, truth - params[(slice(None), *intersect_position)].squeeze()))
                if current_len_sq < closest[1] and (truth != params[(slice(None), *intersect_position)].squeeze()).any():
                    closest[0] = truth
                    closest[1] = current_len_sq
                    closest[2] = truth_index
        
        # Color the closest truth
        if covariance is not None and closest[2] is not None:
            truth_colors[closest[2]] = close_color
                

        assert len(truth_colors) == len(truths), "truth_colors should have one entry for each value of truths"
    
    # Go through and plot each heatmap/histogram
    for x in range(ndim):
        for y in range(x, ndim):
            # Initialize a new subplot
            axes[index] = plt.subplot(ndim, ndim, 1 + x + ndim * y)
            index += 1
            
            # If x == y, plot a histogram
            if x == y:
                # Get the data along the desired axis
                amplitudes = data[tuple(value[0] if index != x else slice(None) for index, value in enumerate(intersect_position))]
                # Get the positions of the amplitudes
                positions = params[tuple([x] + [0] * x + [slice(None)] + [0] * (ndim - x - 1))]
                # Plot the histogram
                plt.plot(positions, amplitudes)
                # Set x limit
                plt.xlim(positions.min(), positions.max())
                # Plot the true values
                if truths is not None:
                    for truth, color in zip(truths, truth_colors):
                        plt.axvline(truth[x], ls=":", color=color)
                if titles is not None:
                    # Add a title
                    plt.title(titles[x])
                
                
            # Otherwise plot a 2d heatmap
            else:
                # Get the data over the desired axes
                amplitudes = data[tuple(value[0] if index != x and index != y else slice(None) for index, value in enumerate(intersect_position))]
                # Get the positions of the amplitudes
                x_positions = params[tuple([x] + [0] * min(x, y) + [slice(None)] + [0] * (max(x, y) - min(x, y) - 1) + [slice(None)] + [0] * (ndim - max(x, y) - 1))]
                y_positions = params[tuple([y] + [0] * min(x, y) + [slice(None)] + [0] * (max(x, y) - min(x, y) - 1) + [slice(None)] + [0] * (ndim - max(x, y) - 1))]
                # Plot the heatmap
                plt.contourf(x_positions, y_positions, amplitudes, heat_scale)
                # Set the x and y limits
                plt.xlim(x_positions.min(), x_positions.max())
                plt.ylim(y_positions.min(), y_positions.max())
                # Plot the true values
                if truths is not None:
                    for truth, color in zip(truths, truth_colors):
                        plt.axvline(truth[x], color=color)
                        plt.axhline(truth[y], color=color)
                        plt.plot(truth[x], truth[y], color=color, marker="o", markersize=8)
                if titles is not None:
                    # Write the axis labels
                    if y == ndim - 1:
                        plt.xlabel(titles[x])
                    if x == 0:
                        plt.ylabel(titles[y])
            
            # Remove axis ticks on the inside plots
            if y != ndim - 1:
                plt.xticks(())
            if x != 0 and x != y:
                plt.yticks(())
                
    # Place a colorbar on the side
    fig.colorbar(cm.ScalarMappable(cl.Normalize(min_amp, max_amp)), ax=axes, ticks=np.linspace(min_amp, max_amp, 10))
    
    # Return the figure
    return fig


def heatmap_corner_max(data:np.ndarray, 
                       params:np.ndarray=None, 
                       count:int=1, 
                       intersect_color:Union[str, tuple[float, ...]]=(1.0, 1.0, 1.0), 
                       titles:Iterable[str]=None, 
                       truths:Iterable[np.ndarray]=None, 
                       truth_colors:Iterable[Union[str, tuple[float, float, float]]]=None, 
                       covariance:np.ndarray=None, 
                       close_color:Union[str, tuple[float, float, float]]="lime") -> tuple[plt.Figure, ...]:
    """Plot the heatmap_corner_intersect at the maximum of the data

    Args:
        data (ndarray): An ndarray of amplitudes in an n-dimensional grid
        params (ndarray[ndarray], optional): An iterable of ndarrays with the same shape as data with length equal to the number of dimensions in data. Each ndarray gives the 
            parameter value for a given data point. (Default: uses data indices)
        count (int, optional): The number of maxima to plot, from largest to smallest (Default: 1)
        intersect_color (str | tuple[float, float, float]): The color with which to plot the intersection of the cross sections (Default: (1.0, 1.0, 1.0))
        titles (Iterable[str], optional): An iterable of strings with a length equal to the data's number of dimensions. Each string corresponds to one dimension of the space
            (Default: None)
        truths (Iterable[ndarray], optional): An iterable of ndarrays acting as points in the parameter space to be plotted over the heatmaps (Default: None)
        truth_colors (Iterable[str | tuple[double, double, double]], optional): An iterable of colors to use when plotting the truths (Default: None)
        covariance (ndarray, optional): Covariance used to calculate the "nearest" truth value. (Default: No calculation)
        close_color (Union[str, tuple[float, float, float]], optional): Color to assign to the truth value nearest the intersection.

    Returns:
        tuple[plt.Figure, ...]: _description_
    """
    
    assert truths is None or len(truths) == len(truth_colors), "truths and truth_colors should be the same length, if provided."
    # Find the top maxima and return their cornerplots
    sorted_data = np.flip(np.sort(data, None))
    figures = []

    if truths is None:
        truths = ()
    if truth_colors is None:
        truth_colors = ()
    
    for maximum in sorted_data[:count]:
        max_point = tuple(ax[0] for ax in np.nonzero(data == maximum))
        figures.append(heatmap_corner_intersect(data, params, np.array(max_point), intersect_color, titles, truths, truth_colors, covariance=covariance, close_color=close_color))

                
    return tuple(figures)


def hist_vars(data:dict, 
              vars:Iterable[Hashable]=("X", "Y", "Z", "A", "E", "T"), 
              title:str=None, 
              xlabel:str=None, 
              ylabel:str=None, 
              legend_label:str=None, 
              xlim:tuple[float, float]=None, 
              ylim:tuple[float, float]=None, 
              xscale:str=None, 
              yscale:str=None, 
              func:Callable=lambda x: x, 
              hist_kwargs:dict={}) -> None:
    """Create a histogram of the values of a dictionary in several subplots within a large plot

    Args:
        data (dict): The data with which to make the histograms
        vars (Iterable[Hashable], optional): The keys for which to make histograms. Defaults to ("X", "Y", "Z", "A", "E", "T").
        title (str, optional): The string to append to the start of each subplot's title. Defaults to None.
        xlabel (str, optional): The string to use as an x label for all the subplots. Defaults to None.
        ylabel (str, optional): The string to use as a y label for all the subplots. Defaults to None.
        legend_label (str, optional): The string to use for this dataset in the legend. Defaults to None.
        xlim (tuple[float, float], optional): The x limits of the subplots. Defaults to None.
        ylim (tuple[float, float], optional): The y limits of the subplots. Defaults to None.
        xscale (str, optional): The xscale of the subplots. Defaults to None.
        yscale (str, optional): The yscale of the subplots. Defaults to None.
        func (Callable, optional): The function to apply to the data before creating a histogram. Defaults to lambdax:x.
        hist_kwargs (dict, optional): Any additional kwargs to pass to the histogram. Defaults to {}.
    """
    
    for index, var in enumerate(vars):
        plt.subplot(2, 3, index + 1)
        plt.hist(func(data[var]), label=(legend_label if legend_label is not None else ""), **hist_kwargs)
        if title is not None:
            plt.title(f"{title}{var}")
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        if legend_label is not None:
            plt.legend()
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        if xscale is not None:
            plt.xscale(xscale)
        if yscale is not None:
            plt.yscale(yscale)


def plot_vars(data:dict, 
              vars:Iterable[Hashable]=("X", "Y", "Z", "A", "E", "T"), 
              xkey:Hashable="t", 
              title:str=None, 
              xlabel:str=None, 
              ylabel:str=None, 
              legend_label:str=None, 
              xlim:tuple[float, float]=None, 
              ylim:tuple[float, float]=None, 
              xscale:str=None, 
              yscale:str=None, 
              xfunc:Callable=lambda x: x, 
              yfunc:Callable=lambda y: y, 
              indices:Union[np.ndarray, slice]=slice(None),
              plot_kwargs:dict={}) -> None:
    """Plot the values of a dictionary in several subplots within a single large plot.

    Args:
        data (dict): A dictionary containing one ndarray for each variable in vars, and one ndarray with the key xkey
        vars (Iterable[Hashable], optional): An iterable of the keys to plot. Defaults to ("X", "Y", "Z", "A", "E", "T").
        xkey (Hashable, optional): The dictionary key to use for the x axis of the plot. Defaults to "t".
        title (str, optional): The string to append to the start of each subplot's title. Defaults to None.
        xlabel (str, optional): The string to use as an x label for all the subplots. Defaults to None.
        ylabel (str, optional): The string to use as a y label for all the subplots. Defaults to None.
        legend_label (str, optional): The string to use for this dataset in the legend. Defaults to None.
        xlim (tuple[float, float], optional): The x limits of the plot. Defaults to None.
        ylim (tuple[float, float], optional): The y limits of the plot. Defaults to None.
        xscale (str, optional): The xscale of the plot. Defaults to None.
        yscale (str, optional): The yscale of the plot. Defaults to None.
        xfunc (Callable, optional): The function to apply to the x values before plotting. Defaults to lambda x:x.
        yfunc (Callable, optional): The function to apply to the y values before plotting. Defaults to lambda y:y.
        indices (ndarray | slice, optional): The indices or method of slicing the input data.
        plot_kwargs (dict, optional): Any kwargs to pass into the plot function. Defaults to {}.
    """
    
    for index, var in enumerate(vars):
        plt.subplot(2, 3, index + 1)
        plt.plot(xfunc(data[xkey][indices]), yfunc(data[var][indices]), label=(legend_label if legend_label is not None else ""), **plot_kwargs)
        if title is not None:
            plt.title(f"{title}{var}")
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        if legend_label is not None:
            plt.legend()
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        if xscale is not None:
            plt.xscale(xscale)
        if yscale is not None:
            plt.yscale(yscale)