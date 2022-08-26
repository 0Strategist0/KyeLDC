"""
Author: Kye Emond
Date: July 8th, 2022


KyeLISA Analysis Module

Functions to help analyze different tdi variables quickly

Methods:
    uncert: Returns a formatted float and uncertainty
    rolling_quantile: Calculates a quantile at each point using nearby points
    roll_local: Rolls an array towards a neighbour
    find_extrema: Finds extrema in an array
    wrap: Wrap a value between min and max
    amp_pdf: The probability density function of the amplitude
    abs_amp_cdf: The cumulative distribution function of the absolute amplitude
    quad_pdf: The probability density function of the quadratic statistic
    quad_cdf: The cumulative distribution function of the quadratic statistic
    var_func: Applies a function to multiple values in a dictionary
    psd: Returns the psd of each time series in a dictionary
    whiten: Whitens each time series in a dictionary
    multivar_find_glitches: Finds glitches in a time series based off of TDI X, Y, Z and T
    moving_cutoff: Finds a cutoff for glitch detection based off of local quantiles
    get_quad: Gets the quadratic statistic of a numpy array
    find_glitches: Finds glitches simply
    deglitch: Zeros out glitches in a time series
    group: Splits a numpy array around given delimiters
    padded: Adds additional True values around previous ones in numpy arrays
    window: Windows out glitches in a time series
"""

import numpy as np
import scipy.signal
from typing import Union, Callable, Hashable, Optional
from pycbc.types import TimeSeries as cbcTimeSeries
from scipy.signal import welch
from scipy.stats import gamma, norm
from scipy.integrate import quad
from numpy.fft import ifft

# Functions
def uncert(value:float, uncertainty:float) -> str:
    """Return a formatted string of the value with uncertainty

    Args:
        value (float): Value to be formatted
        uncertainty (float): Uncertainty to use formatting
    
    Returns:
        str: Formatted value-uncertainty pair
    """
    
    # Get the index of the first uncertainty digit
    round_index = -int(f"{uncertainty:e}".partition("e")[-1]) + 1
    
    # Return the formatted string
    return f"{round(value, round_index)} ± {round(uncertainty, round_index)}"


def rolling_quantile(array:np.ndarray, quantile:float, extents:int, axis:int=None) -> np.ndarray:
    """Compute the quantile of a region around each point of the data.

    Args:
        array (ndarray): The input array for which to find quantiles.
        quantile (float): Quantile to compute, which must be between 0 and 1 inclusive. 
        extents (int): Extents of the neighbourhood to take the quantile of. For example, extents of 5 would result in quantiles being taken of regions with points up to 5 points away from the main point. 
        axis (int, optional): Axis along which to calculate the quantiles. If None, calculates quantile taking every dimension into account. Defaults to None.
        
    Returns:
        ndarray: The rolling quantile results, of the same shape as the input array
    """
    
    # Initialize the array to return
    return_array = np.zeros(array.shape)
    
    if axis is None:
        # If there is no specified axis, iterate through every index and get the quantile of the region around that index
        for index in range(array.size):
            unravelled_indices = np.unravel_index(index, array.shape)
            return_array[unravelled_indices] = np.quantile(array[tuple((slice(max(unrav_index - extents, 0), 
                                                                              min(unrav_index + extents, ax_len - 1)) 
                                                                        for ax_len, unrav_index in zip(array.shape, unravelled_indices)))], 
                                                           quantile)
    else:
        # If there is a specified axis, iterate along that axis and get the quantile of the region around the index along that axis
        for index in range(array.shape[axis]):
            indices = [slice(None)] * array.ndim
            indices[axis] = index
            indices = tuple(indices)
            
            quantile_indices = [slice(None)] * array.ndim
            quantile_indices[axis] = slice(max(index - extents, 0), min(index + extents, array.shape[axis] - 1))
            quantile_indices = tuple(indices)
            
            return_array[indices] = np.quantile(array[quantile_indices], quantile, axis)
    
    # Return the result
    return return_array


def roll_local(array:np.ndarray, neighbour:int) -> np.ndarray:
    """Roll a numpy array to the neighbour indicated.

    Args:
        array (ndarray): Array to be rolled
        neighbour (int): The index of the neighbour to be rolled towards

    Returns:
        ndarray: The rolled array
    """
    
    # Initialization
    ndim = array.ndim
    rolled_array = array.copy()
    
    # Iterate through each dimension and roll appropriately
    for axis in range(ndim):
        shift = ((neighbour // (3 ** axis)) % 3) - 1
        rolled_array = np.roll(rolled_array, shift, axis)
    
    return rolled_array


def find_extrema(array:np.ndarray, type:str="all", plateaus:bool=False, return_indices:bool=False) -> np.ndarray:
    """Find the extrema in an N-dimensional array by looking at all nearby points and comparing them

    Args:
        array (ndarray): The array in which to find extrema
        type (str, optional): The types of extrema to find. Can be "max", "min", or "all". Defaults to "all".
        plateaus (bool, optional): Whether to count plateaus as extrema. When False, extrema must have all neighbouring points smaller or larger. Defaults to False.
        return_indices (bool, optional): Whether to return indices of extrema. If not, a boolean array is returned. Defaults to False.
    
    Returns:
        ndarray: A boolean ndarray of the same shape as array indicating the locations of detected extrema, or an ndarray containing the indices of detected extrema, with indices grouped by element.
    """
    
    assert type in ("all", "min", "max"), 'type should be "all", "min", or "max"'
    
    # Assign the function used to compare adjacent points
    if plateaus:
        less = np.less_equal
        greater = np.greater_equal
    else:
        less = np.less
        greater = np.greater
    
    # Initialize the boolean arrays of extrema
    extrema = np.full(array.shape, False)
    if type in ("all", "min"):
        minima = np.full(array.shape, True)
    if type in ("all", "max"):
        maxima = np.full(array.shape, True)
    
    for neighbour in set(range(3 ** array.ndim)) - {(3 ** array.ndim) // 2}:
        # Get the rolled array
        rolled_array = roll_local(array, neighbour)
        
        # For each neighbour, if finding minima, confirm that points are minima
        if type in ("all", "min"):
            minima &= less(array, rolled_array)
        
        # For each neighbour, if finding maxima, confirm that points are maxima
        if type in ("all", "max"):
            maxima &= greater(array, rolled_array)
    
    # Combine the extrema
    if type in ("all", "min"):
        extrema |= minima
    if type in ("all", "max"):
        extrema |= maxima
    
    # Return either the indices or the boolean array depending on what was requested
    if return_indices:
        return np.argwhere(extrema)
    else:
        return extrema
        

def wrap(value:float, minimum:float, maximum:float) -> float:
    """Wrap value between min and max such that min <= wrap(x, min, max) < max.

    Args:
        value (float): The value to be wrapped.
        minimum (float): The minimum value to wrap around.
        maximum (float): The maximum value to wrap around.

    Returns:
        float: value wrapped around the range [min, max)
    """
    
    interval_size = maximum - minimum
    
    # Get the amount that value is outside of a multiple of the interval
    fmod_value = np.fmod(value - minimum, interval_size)
    # Keep the value within the span of the interval
    if isinstance(value, float):
        if fmod_value < 0.0:
            fmod_value += interval_size
    else:
        if isinstance(interval_size, float):
            fmod_value[fmod_value < 0.0] += interval_size
        else:
            fmod_value[fmod_value < 0.0] += interval_size[fmod_value < 0.0]
    
    # Return the wrapped value
    return fmod_value + minimum


def amp_pdf(x:float, autocov:np.ndarray=None, psd:np.ndarray=None) -> float:
    """Return the value of the amplitude pdf at x for a distribution with an autocovariance function of autocov or a psd of psd.

    Args:
        x (float): The position at which to evaluate the pdf
        autocov (ndarray, optional): An array of floats giving the evaluations of the autocovariance function of the time series noise. Only needed if psd is not provided. Defaults to None.
        psd (ndarray, optional): The PSD of the time series. Used to calculate an approximation of the autocovariance function. Prefer using autocov if available. If autocov is provided, this will be ignored. Defaults to None.

    Returns:
        float: The probability density of the amplitude at the value x
    """
    
    # Make sure either autocov or psd is provided
    assert autocov is not None or psd is not None, "At least one of autocov or psd should be provided"
    
    # If no autocov is provided, calculate from the psd
    if autocov is None:
        autocov = 1e-1 * np.real_if_close(ifft(psd))
    
    # Return the evaluation
    return norm.pdf(x, scale=np.sqrt(autocov[0]))
    
    
def abs_amp_cdf(x:float, autocov:np.ndarray=None, psd:np.ndarray=None) -> float:
    """Return the value of the amplitude cdf at x for a distribution with an autocovariance function of autocov or a psd of psd.

    Args:
        x (float): The position at which to evaluate the cdf
        autocov (ndarray, optional): An array of floats giving the evaluations of the autocovariance function of the time series noise. Only needed if psd is not provided. Defaults to None.
        psd (ndarray, optional): The PSD of the time series. Used to calculate an approximation of the autocovariance function. Prefer using autocov if available. If autocov is provided, this will be ignored. Defaults to None.

    Returns:
        float: The cumulative distribution of the amplitude at the value x
    """
    
    # Make sure either autocov or psd is provided
    assert autocov is not None or psd is not None, "At least one of autocov or psd should be provided"
    
    # If no autocov is provided, calculate from the psd
    if autocov is None:
        autocov = 1e-1 * np.real_if_close(ifft(psd))
    
    # Return the evaluation
    if x <= 0.0:
        return 0.0
    else:
        return 1.0 - 2.0 * norm.cdf(-x, scale=np.sqrt(autocov[0]))


def quad_pdf(x:Union[float, np.ndarray], autocov:np.ndarray=None, psd:np.ndarray=None) -> float:
    """Return the value of the quadratic statistic pdf at x for a distribution with an autocovariance function of autocov or a psd of psd.

    Args:
        x (float): The position at which to evaluate the pdf
        autocov (ndarray, optional): An array of floats giving the evaluations of the autocovariance function of the time series noise. Only needed if psd is not provided. Defaults to None.
        psd (ndarray, optional): The PSD of the time series. Used to calculate an approximation of the autocovariance function. Prefer using autocov if available. If autocov is provided, this will be ignored. Defaults to None.

    Returns:
        float: The probability density of the quadratic statistic at the value x
    """
    
    # Make sure either autocov or psd is provided
    assert autocov is not None or psd is not None, "At least one of autocov or psd should be provided"
    
    # If no autocov is provided, calculate from the psd
    if autocov is None:
        autocov = 1e-1 * np.real_if_close(ifft(psd))
    
    # Return the evaluation
    if isinstance(x, np.ndarray):
        output = np.zeros(x.shape)
        for index, value in enumerate(x):
            output[index] = quad_pdf(value, autocov)
        
        return output
    else:
        return quad(lambda x, p: gamma.pdf(x, 1/2, scale=(2.0 * autocov[0])) * gamma.pdf(p - x, 1/2, scale=4.0 * (autocov[0] - autocov[2])), 0, x, args=(x,))[0]
        


def quad_cdf(x:float, autocov:np.ndarray=None, psd:np.ndarray=None, point:float=1e-37) -> float:
    """Return the value of the quadratic statistic cdf at x for a distribution with an autocovariance function of autocov or a psd of psd.

    Args:
        x (float): The position at which to evaluate the cdf
        autocov (ndarray, optional): An array of floats giving the evaluations of the autocovariance function of the time series noise. Only needed if psd is not provided. Defaults to None.
        psd (ndarray, optional): The PSD of the time series. Used to calculate an approximation of the autocovariance function. Prefer using autocov if available. If autocov is provided, this will be ignored. Defaults to None.
        point (float, optional): Point at which to break the integral into segments, allowing it to calculate better. Defaults to 1e-37.

    Returns:
        float: The cumulative distribution of the quadratic statistic at the value x
    """
    
    # Make sure either autocov or psd is provided
    assert autocov is not None or psd is not None, "At least one of autocov or psd should be provided"
    
    # If no autocov is provided, calculate from the psd
    if autocov is None:
        autocov = 1e-1 * np.real_if_close(ifft(psd))
    
    # Return the evaluation
    if x == 0.0:
        return 0.0
    else:
        return 1.0 - (quad(quad_pdf, x, point, args=(autocov))[0] + quad(quad_pdf, point, np.inf, args=(autocov))[0])


def var_func(func:Callable, data:dict, skip_keys:Hashable="t", copy:bool=True) -> dict:
    """Apply a function to every element of a dictionary other than skip_keys.
    
    Args:
        func (function): The function to apply to the dictionary elements
        data (dict): The dictionary whose elements func is applied to
        skip_keys (Hashable, optional): An iterable of keys to ignore when applying the function. Defaults to "t"
        copy (bool, optional): Whether to create a copy of the data rather than modify it in-place. Defaults to True
    
    Return:
        Dictionary with the same keys as data with func applied to its values"""

    new_data = {}
    # Apply function to each value
    for key, value in data.items():
        if copy:
            if key in skip_keys:
                new_data[key] = value.copy()
            else:
                new_data[key] = func(value.copy())
        else:
            if key in skip_keys:
                new_data[key] = value
            else:
                new_data[key] = func(value)
    
    return new_data


def psd(data:dict, fs:float, nperseg:int, window:str="hanning") -> dict:
    """Returns the psd of each time series in a dictionary.
    
    Args:
        data (dict): A dictionary of time series for which to find the psds
        fs (float): The sampling frequency of the time series
        nperseg (int): The number of samples to use in each fourier transform
        window (str, optional): The windowing type to use for the fourier transform. Defaults to "hanning"
        
    Returns:
        A dictionary of psds and corresponding frequencies
    """

    psds = {}
    # Find the psd for each value
    for var in data.keys():
        if var != "t":
            psds["f"], psds[var] = welch(np.nan_to_num(data[var]), fs=fs, window=window, nperseg=nperseg)
    
    return psds


def whiten(data:dict, nperseg:int, filter_length:float, dt:float=5.0) -> dict:
    """Whiten the time series in a dictionary. 
    
    Args:
        data (dict): A dictionary of ndarray time series
        nperseg (int): The number of data points for the welch method
        fliter_length (float): The smoothing filter length for the PSD
        dt (float): The period of time between time samples. Defaults to 5.0
    
    Return:
        A dictionary containing the whitened data"""

    white_data = {}
    # Whiten each value
    for var in data.keys():
        if var != "t":
            white_data[var] = cbcTimeSeries(data[var], delta_t=dt).whiten(nperseg, filter_length)
            if var == "X":
                white_data["t"] = white_data["X"].get_sample_times()
            white_data[var] = np.asarray(white_data[var])
    
    return white_data


def multivar_find_glitches(data:dict, cutoffs:Union[np.ndarray, float, dict], padding:int=10, include_nans:bool=False) -> np.ndarray:
    """Return a boolean array indicating the location of glitches. Glitches are detected when data > cutoffs in the TDI T variable, or in two but not three of X, Y, Z. 

    Args:
        data (dict): A dictionary of TDI time series
        cutoffs (ndarray | float | dict): A float, array, or dictionary of arrays or floats of amplitudes at which to label data a glitch
        padding (int, optional): The amount of padding to add to the sides of data > cutoffs points. Defaults to 10.
        include_nans (bool, optional): Whether to count nans as glitches. Defaults to False.

    Returns:
        ndarray: A boolean array indicating the locations of glitches
    """
    mask = np.full(data["t"].shape, False)

    if include_nans:
        mask |= np.isnan(data["X"])
    
    # Mark peaks in "T" as glitches, then cycle through "X", "Y", "Z" and mark any peaks in two but not three as glitches
    if type(cutoffs) != dict:
        mask |= np.abs(data["T"]) > cutoffs
        for A, B, C in tuple(tuple(np.roll(("X", "Y", "Z"), shift)) for shift in range(3)):
            mask |= (padded((np.abs(data[A]) > cutoffs), padding) & padded((np.abs(data[B]) > cutoffs), padding)) & ~padded(np.abs(data[C]) > cutoffs, padding)
    else:
        mask |= np.abs(data["T"]) > cutoffs["T"]
        for A, B, C in tuple(tuple(np.roll(("X", "Y", "Z"), shift)) for shift in range(3)):
            mask |= (padded(np.abs(data[A]) > cutoffs[A], padding) & padded(np.abs(data[B]) > cutoffs[B], padding)) & ~padded(np.abs(data[C]) > cutoffs[C], padding)
    
    return mask


def moving_cutoff(data:np.ndarray, divisions:int=40, multiplier:float=10.0, quantile:float=0.4) -> np.ndarray:
    """Find the cutoffs for glitch amplitudes along a given dataset.

    Args:
        data (ndarray): ndarray of data
        divisions (int, optional): The number of segments to split the data into for quantile estimation. Defaults to 40
        multiplier (float, optional): The value to multiply the quantile by to get local cutoffs. Defaults to 10.0
        quantile (float, optional): The quantile of the data to use. Defaults to 0.4
    
    Return:
        ndarray of same length as data with the cutoffs for each index
    """

    datasize = len(data)
    segments = np.linspace(0, datasize, divisions + 1, dtype=int)

    cutoffs = np.zeros(data.shape)
    # Iterate through the array and find the sample quantile around that area
    prev_index = None
    for current_index in segments:
        if prev_index is not None:
            cutoffs[prev_index:current_index] = multiplier * np.nanquantile(data[prev_index:current_index], quantile)
        prev_index = current_index

    return cutoffs


def get_quad(data:np.ndarray, potential_coef:float=1.0, kinetic_coef:float=1.0) -> np.ndarray:
    """Return the quad statistic of the data.

    Args:
        data (ndarray): Array for which to find the quad statistic
        potential_coef (float, optional): Coefficient of the potential component. Defaults to 1.0.
        kinetic_coef (float, optional): Coefficient of the kinetic component. Defaults to 1.0.

    Returns:
        ndarray: Array of quad statistics at each point
    """
    
    assert len(data) >= 3, "data should have a length of 3 or more, since that's the minimum needed to generate a slope"

    # Add potential quad
    quad = potential_coef * data ** 2.0

    # Add kinetic quad
    velocity = data[2:] - data[:-2]
    kinetic_quad = kinetic_coef * velocity ** 2.0
    kinetic_quad = np.concatenate(([kinetic_quad[0]], kinetic_quad, [kinetic_quad[-1]]))

    quad += kinetic_quad

    return quad


def find_glitches(data:np.ndarray, padding:int=0, glitch_tolerance:float=6e-20, include_gaps:bool=True) -> np.ndarray:
    """Find glitches in a time series purely by comparing the amplitude of the time series to the glitch_tolerance.

    Args:
        data (ndarray): An array in which to find glitches
        padding (int, optional): The amount of padding to add around glitches. Defaults to 0.
        glitch_tolerance (float, optional): The amplitude at which to decide a point is a glitch. Defaults to 6e-20.
        include_gaps (bool, optional): Whether to include nans as glitches. Defaults to True.

    Returns:
        ndarray: A boolean array indicating the positions of found glitches
    """


    # Find the locations of glitches
    if include_gaps:
        base_mask = (np.abs(data) > glitch_tolerance) | np.isnan(data)
    else:
        base_mask = np.abs(data) > glitch_tolerance

    if padding != 0:
        glitch_mask = base_mask.copy()

        for shift in range(-padding, padding + 1):
            glitch_mask |= np.roll(base_mask, shift=shift)


        # Return those locations
        return glitch_mask
    else:
        return base_mask


def deglitch(data:np.ndarray, data_mask:np.ndarray=None, padding:int=0, glitch_tolerance:float=6e-20, value:float=0.0, copy:bool=True) -> np.ndarray:
    """Remove glitches from an array by setting them to a given value.

    Args:
        data (ndarray): An ndarray of points to be deglitched
        data_mask (ndarray, optional): An alternate set of data points to use for glitch identification. Defaults to None.
        padding (int, optional): The amount of padding to add around the glitches. Defaults to 0.
        glitch_tolerance (float, optional): Amplitude at which to start labelling data as glitches. Defaults to 6e-20.
        value (float, optional): Glitches are set to this value. Defaults to 0.0.
        copy (bool, optional): Whether to create a new copy or edit the array in-place. Defaults to True.

    Returns:
        ndarray: The data with the glitches set to value
    """
    
    if data_mask is None:
        data_mask = data.copy()

    # Set glitches to value in passed data
    if copy:
        return_data = data.copy()
        return_data[find_glitches(data_mask, padding, glitch_tolerance)] = value
        return return_data
    else:
        data[find_glitches(data_mask, padding, glitch_tolerance)] = value
        return data


def group(delimiters:np.ndarray, data:np.ndarray=None) -> list[np.ndarray]:
    """Take either data, or the indices of delimiters. Split this array around the delimiters (delimiter exclusive) and return the split arrays in a list.

    Args:
        delimiters (ndarray): Boolean mask or array of indices at which to split the data, leaving out the delimiters from the split
        data (ndarray, optional): Data to be split. Defaults to delimiters indices

    Returns:
        list[ndarray]: A list of ndarrays containing the split data.
    """
    
    # Data validations
    assert delimiters.dtype == bool or (delimiters.dtype == int and data is not None), "delimiters should be boolean or integer. If integer, data should not be None."
    
    # Find the indices to split at
    if delimiters.dtype == bool:
        split_indices = np.arange(delimiters.size)[delimiters]
    else:
        split_indices = delimiters
    
    # If there's no data, just return grouped indices
    if data is None:
        groups = np.split(np.arange(delimiters.size), split_indices)
        groups = ([groups[0]] if groups[0].size > 1 else []) + [array[1:] for array in groups[1:] if array.size > 1]
    # Otherwise, group the data
    else:
        groups = np.split(data, split_indices)
        groups = ([groups[0]] if groups[0].size > 1 else []) + [array[1:] for array in groups[1:] if array.size > 1]
    
    # Return the groups
    return groups


def padded(array:np.ndarray, padding:int, copy:bool=True) -> np.ndarray:
    """Adds extra True values as padding around True values in a 1D boolean numpy array. 
    
    Args:
        array (ndarray): Array to pad.
        padding (int): Number of True values to add as padding on each side of pre-existing True values.
        copy (bool, optional): Whether to create a copy of the array or modify it directly. Defaults to True.
    
    Returns:
        ndarray: The array with padding added
    """
    
    # Initialize the arrays
    if copy:
        return_array = array.copy()
    else:
        return_array = array
    
    rolling_array = return_array.copy()

    # Roll the array around, spreading the True values to nearby elements
    for shift in range(-padding, padding + 1):
        return_array |= np.roll(rolling_array, shift=shift)
    
    return return_array


def window(data:np.ndarray, glitch_tolerance:float=1e-19, glitch_mask:np.ndarray=None, padding:int=10, window_period:float=2e3, return_window:bool=False, copy:bool=True) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Window out the glitches in a time series dataset, and optionally return the window function as well.

    Args:
        data (ndarray): The data to window out
        glitch_tolerance (float, optional): The amplitude at which to label data a glitch. Has no effect if glitch_mask is provided. Defaults to 1e-19.
        glitch_mask (ndarray, optional): A boolean array indicating the location of glitches. Overrides glitch_tolerance. Defaults to None.
        padding (int, optional): The amount of padding to add to glitches. Defaults to 10.
        window_period (float, optional): The number of data points affected by the windowing for both sides of a glitch combined. Defaults to 2e3.
        return_window (bool, optional): Whether to return the window function. Defaults to False.
        copy (bool, optional): Whether to create a copy of the data array or modify it in-place. Defaults to True.

    Returns:
        ndarray: The windowed version of the data
        ndarray (optional): The window function used. Only returned if return_window is True
    """

    # Get the correct glitch mask and pad it appropriately
    if glitch_mask is None:
        glitch_mask = find_glitches(data, padding=padding, glitch_tolerance=glitch_tolerance)
    else:
        glitch_mask = padded(glitch_mask, padding)
    
    # Initialize the array to be returned
    if copy:
        return_data = data.copy()
    else:
        return_data = data
    
    # Initialize the window function
    window_function = np.ones(return_data.shape)

    # Zero out glitches
    window_function[glitch_mask] = 0.0
    return_data[glitch_mask] = 0.0

    # Get an tuple of the valid windows
    windows = group(glitch_mask)

    # Go through each window of valid indices and window it properly
    for window in windows:
        tukey_coef = window_period / window.size
        if tukey_coef > 1.0:
            window_function[window] = 0.0
        else:
            tukey_window = scipy.signal.windows.tukey(window.size, tukey_coef, True)
            window_function[window] *= tukey_window
    
    # Window the data with the function created
    return_data *= window_function

    # Return the correct values
    if return_window:
        return return_data, window_function
    else:
        return return_data