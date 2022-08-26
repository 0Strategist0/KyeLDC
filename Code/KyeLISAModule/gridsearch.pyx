"""
Author: Kye Emond
Date: July 8th, 2022


KyeLISA Gridsearch Module

Functions and classes to help run a gridsearch quickly and efficiently. 

Methods:
    full_gridsearch_python: Slower python gridsearch function
    _evaluate_func: Private function to evaluate a function on an array in parallel
    full_gridsearch: Fast C multiprocessing gridsearch function

Classes:
    _Copier: Private class to let you pass lambda functions into multiprocessing pools
"""

# Imports
import cython
from libc.math cimport fmod
import numpy as np
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from typing import Iterable, Union, Callable

MEMORY_NAMES = ("result", "grid")

@cython.boundscheck(False)
@cython.cdivision(True)
def full_gridsearch_python(func: Callable, ndim: int, bounds: tuple[np.ndarray, np.ndarray], gridshape: Union[int, np.ndarray], end_spacing: tuple[float, ...], verbose: cython.bint = False, func_return_shape: Iterable = (), func_args: tuple = (), func_kwargs: dict = {}) -> tuple:
    """A function that searches for the maximum of a function by evalutating the function over finer and finer grids. 

    Args:
        func (function): The function to be maximized. Must take an iterable of parameters, followed by extra arguments. Must return a double or double-like.
        ndim (int): The number of dimensions in the grid's space.
        bounds (Iterable[Iterable[double, ...], Iterable[double, ...]]): A tuple of ndarrays, the first one indicating the lower bounds of the grid, the second indicating the upper bounds. The lengths of the ndarrays must match ndim. 
        gridshape (int, Iterable): An integer or 1d array of the number of gridlines that should be along each axis. If an integer is given, it takes that to be the number of gridpoints along all the dimensions. If an array is given, it uses each number as the gridpoints along a given axis. 
        end_spacing (Iterable[double, ...]): A 1d array giving the grid spacing along each dimension at which to terminate. The length must match ndim.
        verbose (bool, optional): Whether to print out progress or not. Defaults to False.
        func_return_shape (int, optional): The shape of the function's return value. Defaults to double. If multiple returns, the first value is maximized.
        func_args (tuple, optional): Any extra arguments to pass in to func.
        func_kwargs (dict, optional): Any extra kwargs to pass in to func.

    Returns:
        tuple: A list of tuples of ndarrays, each tuple containing a grid with the function evaluated along it, and a grid with the parameter values in the last axis. 
    """
    
    # Data validation
    assert len(bounds[0]) == ndim and len(bounds[1]) == ndim, "bounds should have a length of ndim"
    assert len(end_spacing) == ndim, "end_spacing should have a length of ndim"
    assert type(gridshape) == int or len(gridshape) == ndim, "start_gridnumber should have a length of ndim or be an integer"
    
    # Set up gridshape properly
    if type(gridshape) == int:
        gridshape = tuple([gridshape] * ndim)
    
    # Initialize a return variable
    return_grids = []
    
    # Set up some variables for looping over the grid
    current_bounds = bounds
    current_gridshape = np.array(gridshape)
    grid_spacing = current_bounds[1] - current_bounds[0]
    
    # If you want a progress report, get the number of cycles that will be needed
    # if verbose:
    cdef int cycles_total = int(np.ceil(np.log(grid_spacing / np.array(end_spacing)) / np.log(np.array(gridshape) / 2.0)).max())
    cdef int cycle_number = 0
    cdef int gridsize, flat_index
    cdef double PERCENT_TO_REPRINT = 1.0
    cdef double percent, prev_percent
    # Keep searching over finer and finer grids until the grid is finer than end_spacing
    while (grid_spacing > end_spacing).any():
        # If verbose, print the cycle progress
        if verbose:
            cycle_number += 1
            prev_percent = 0.0
        
        # Set the new grid_spacing based off of the new bounds
        grid_spacing = (current_bounds[1] - current_bounds[0]) / current_gridshape
        
        # Initialize the first (faster) grid
        speedy_grid = tuple(tuple(np.linspace(current_bounds[0][dim], current_bounds[1][dim], current_gridshape[dim])) for dim in range(ndim))
        
        # Initialize a grid to store the function values in
        search_results = (np.zeros((*gridshape, *func_return_shape)), np.zeros((*gridshape, ndim)))
        
        # Iterate through the grid
        gridsize = np.prod(current_gridshape)
        for flat_index in range(gridsize):
            # If verbose, give progress reports every 10%
            if verbose:
                percent = 100.0 * flat_index / gridsize
                if fmod(percent, PERCENT_TO_REPRINT) < fmod(prev_percent, PERCENT_TO_REPRINT):
                    print(f"Gridsearch {cycle_number} of {cycles_total}: {percent:.1f}%", end="\r")
                prev_percent = percent
            
            # Evaluate function at each point
            fancy_index = tuple((flat_index // current_gridshape[:dim].prod()) % current_gridshape[dim] for dim in range(ndim))
            parameters = tuple(speedy_grid[dim][fancy_index[dim]] for dim in range(ndim))
            search_results[0][fancy_index] = func(parameters, *func_args, **func_kwargs)
            search_results[1][fancy_index] = parameters
            
        
        # Add the results into a list to return
        return_grids.append(search_results)
        
        # Find the max value, and the params where that max value are found
        max_indices = np.unravel_index(search_results[0].argmax(), search_results[0].shape)
        max_params = search_results[1][max_indices]
        
        # Set the new bounds for the finer gridsearch
        current_bounds = (np.clip(max_params - grid_spacing, current_bounds[0], current_bounds[1]), np.clip(max_params + grid_spacing, current_bounds[0], current_bounds[1]))
    # Return the array of grids
    return tuple(return_grids)


def _evaluate_func(int index, func, int func_return_size, result_arr_shape, grid_arr_shape, int gridsize=0, int cycle_number=0, int cycles_total=0, bint verbose=False, func_args=(), func_kwargs={}) -> None:
    """A helper function to run in subprocesses in order to fill out the SharedMemory results array
    
    Args:
        func (function): The function to be evaluated into the SharedMemory. Must take an iterable of parameters, followed by extra arguments.
        result_indices (int, slice): The indices of the results array to fill out.
        param_index (int): The index in the param grid array to get parameters from.
        result_arr_shape (tuple | int): The shape of the results array.
        grid_arr_shape (tuple): The shape of the grid array.
        gridsize (int, optional)
        verbose (bool, optional): Whether to print progress.
        func_args (tuple, optional): Any extra arguments to pass in to func.
        func_kwargs (dict, optional): Any extra keyword arguments to pass into func.
    """
    # Get the memory
    result_mem = SharedMemory(name=MEMORY_NAMES[0], create=False)
    grid_mem = SharedMemory(name=MEMORY_NAMES[1], create=False)

    # Make arrays using the memory
    result_arr = np.ndarray(result_arr_shape, np.float64, result_mem.buf)
    grid_arr = np.ndarray(grid_arr_shape, np.float64, grid_mem.buf)

    # Edit the correct array values
    result_arr[func_return_size * index : func_return_size * (index + 1)] = func(grid_arr[index], *func_args, **func_kwargs).flatten()

    # If verbose, print out progress
    if verbose:
        if (index % (gridsize // 100)) == 0:
            percent = 100.0 * index / gridsize
            print(f"Gridsearch {cycle_number} of {cycles_total}: {percent:.1f}%", end="\r")
    
    # Free the memory
    result_mem.close()
    grid_mem.close()

class _Copier(object):
    """A helper class to allow you to pass lambda functions into multiprocessing.Pool.map"""

    def __init__(self, args, func_args=(), func_kwargs={}):
        self.args = args
        self.func_args = func_args
        self.func_kwargs = func_kwargs
    
    def __call__(self, index):
        _evaluate_func(index, *self.args, func_args=self.func_args, func_kwargs=self.func_kwargs)

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def full_gridsearch(func, ndim, bounds, gridshape, end_spacing, bint verbose=False, processes=1, func_return_shape=(), func_args=(), func_kwargs={}):
    """A function that searches for the maximum of a function by evalutating the function over finer and finer grids. 

    The function has two modes: C mode and multiprocessor mode. 
    C mode is active when processes = 1. Multiprocessor mode activates when processes > 1. 
    
    C mode runs much faster with less overhead, but it only runs one process at a time. It performs well when the function 
    evaluated across the grid is fast. 
    
    Multiprocessor mode has more overhead and thus the full_gridsearch function takes more time to run, but it allows the 
    search to run in parallel with "processes" number of subprocesses. If the function passed in to be evaluated over the
    grid is the bottleneck, multiprocessor mode can accelerate the evaluation by several times, with greater improvements 
    the slower the passed in function and the more processes given to "processes". You get diminishing returns when 
    passing values larger than the number of cores on your computer to "processes".

    Args:
        func (function): The function to be maximized. Must take an iterable of parameters, followed by extra arguments 
                         and keyword arguments. Must return a double if func_return_shape is empty, or an array of doubles 
                         with a shape of func_return_shape.
        ndim (int): The number of dimensions in the grid's space.
        bounds (Iterable[ndarray, ndarray]): An iterable of ndarrays, the first one indicating the lower bounds of the grid, 
                                             the second indicating the upper bounds. The lengths of the ndarrays must match ndim. 
        gridshape (int, Iterable): An integer or iterable of integers giving of the number of gridlines that should be along each 
                                   axis. If an integer is given, it is taken to be the number of gridpoints along each dimension. 
                                   If an iterable is given, it must be of length ndim, and each number in the iterable is
                                   used as the number of gridpoints along its respective axis. 
        end_spacing (Iterable): An iterable of floats giving the grid spacing along each dimension at which to terminate. 
                                The length must match ndim.
        verbose (bool, optional): Whether to print out progress or not. Defaults to False.
        processes (int, optional): The number of subprocesses to use in parallel for grid calculations. Defaults to 1.
        func_return_shape (Iterable[int, ...], optional): The shape of the function's return value. Defaults to float. 
                                                          If multiple returns, the first value is maximized.
        func_args (tuple, optional): Any extra arguments to pass in to func.
        func_kwargs (dict, optional): Any extra keyword arguments to pass in to func.

    Returns:
        tuple: A list of tuples of ndarrays, each tuple containing a grid with the function evaluated along it, 
               and a grid with the parameter values in the last axis. 
    """
    
    # Data validation
    assert len(bounds[0]) == ndim and len(bounds[1]) == ndim, "bounds should have a length of ndim"
    assert (bounds[0] < bounds[1]).all(), "first entry to bounds should be smaller than second entry"
    assert len(end_spacing) == ndim, "end_spacing should have a length of ndim"
    assert type(gridshape) == int or len(gridshape) == ndim, "gridshape should have a length of ndim or be an integer"
    assert processes >= 1 and type(processes) == int, "processes should be an integer >= 1"
    
    # Convert parameters to correct types
    if type(gridshape) == int:
        gridshape = np.array([gridshape] * ndim)
    else:
        gridshape = np.array(gridshape)
    end_spacing = np.asarray(end_spacing)
    if type(func_return_shape) == int:
        func_return_shape = (func_return_shape,)
    else:
        func_return_shape = tuple(func_return_shape)

    # Initialize a return variable
    return_grids = []
    
    # Set up some variables for looping over the grid
    current_bounds = bounds
    grid_spacing = current_bounds[1] - current_bounds[0]
    
    # Initialize cython typed variables for optimized looping over the grid
    cdef int cycles_total = int(np.ceil(np.log(np.array(end_spacing) / (grid_spacing / (gridshape - 1))) 
                                / np.log(2.0 / np.array(gridshape - 1))).max()) + 1
    cdef int cycle_number = 0
    cdef int gridsize = gridshape.prod()
    cdef int flat_index
    cdef double PERCENT_TO_REPRINT = 1.0
    cdef double percent, prev_percent
    cdef int func_return_dim = len(func_return_shape)
    cdef int func_return_size = np.array(func_return_shape).prod()
    cdef double [:] func_result
    cdef double [:, :] speedy_grid
    cdef double [:, :] temp_grid

    # If multiprocessing, set up the shared memory for the memoryview
    if processes > 1:
        # Set up result shared memory
        init_array = np.zeros((gridsize * func_return_size))
        result_mem = SharedMemory(name=MEMORY_NAMES[0], create=True, size=init_array.nbytes)
        result_array = np.ndarray(init_array.shape, init_array.dtype, result_mem.buf)
        result_array[:] = init_array[:]

        # Set up grid shared memory
        init_array = np.zeros((gridsize, ndim))
        grid_mem = SharedMemory(name=MEMORY_NAMES[1], create=True, size=init_array.nbytes)
        grid_array = np.ndarray(init_array.shape, init_array.dtype, grid_mem.buf)
        grid_array[:] = init_array[:]

        # Set up a pool for parallel processing
        pool = mp.Pool(processes)

        del init_array
    else:
        result_array = np.zeros((gridsize * func_return_size))
        grid_array = np.zeros((gridsize, ndim))

    func_result = result_array
    speedy_grid = grid_array
    
    # Keep searching over finer and finer grids until the grid is finer than end_spacing
    while (grid_spacing > end_spacing).any():
        # If verbose, get the cycle progress
        if verbose:
            cycle_number += 1
            prev_percent = 0.0
        
        # Set the new grid_spacing based off of the new bounds
        grid_spacing = (current_bounds[1] - current_bounds[0]) / (gridshape - 1)
        
        # Initialize an array of parameter values grid
        coord_vectors = tuple(np.linspace(current_bounds[0][dim], current_bounds[1][dim], gridshape[dim]) for dim in range(ndim))
        temp_grid = np.swapaxes(np.array(np.meshgrid(*coord_vectors)), 1, 2).transpose().reshape((gridsize, ndim))
        speedy_grid[...] = temp_grid

        # Iterate through the grid
        if processes > 1:
            process_func = _Copier(
                                   (    
                                        func, 
                                        func_return_size,  
                                        result_array.shape, 
                                        grid_array.shape, 
                                        gridsize, 
                                        cycle_number, 
                                        cycles_total, 
                                        verbose
                                   ),  
                                   func_args, 
                                   func_kwargs)
            pool.map(process_func, range(gridsize))
        elif func_return_size != 1:
            for flat_index in range(gridsize):
                # If verbose, give progress reports every 1%
                if verbose:
                    percent = 100.0 * flat_index / gridsize
                    if fmod(percent, PERCENT_TO_REPRINT) < fmod(prev_percent, PERCENT_TO_REPRINT):
                        print(f"Gridsearch {cycle_number} of {cycles_total}: {percent:.1f}%", end="\r")
                    prev_percent = percent
                
                # Evaluate function at each point
                func_result.base[func_return_size * flat_index 
                                 : func_return_size * (flat_index + 1)] = func(speedy_grid[flat_index], 
                                                                               *func_args, 
                                                                               **func_kwargs).flatten()
        else:
            for flat_index in range(gridsize):
                # If verbose, give progress reports every 1%
                if verbose:
                    percent = 100.0 * flat_index / gridsize
                    if fmod(percent, PERCENT_TO_REPRINT) < fmod(prev_percent, PERCENT_TO_REPRINT):
                        print(f"Gridsearch {cycle_number} of {cycles_total}: {percent:.1f}%", end="\r")
                    prev_percent = percent
                
                # Evaluate function at each point
                func_result[flat_index] = func(speedy_grid[flat_index], *func_args, **func_kwargs)
        
        # Reshape speedy_grid into a grid
        reshaped_param_results = np.moveaxis(np.asarray(speedy_grid).reshape(np.flip((ndim, *gridshape))), ndim, 0).transpose().copy()
        
        # Reshape function return array into a grid
        func_result_shape = (*np.flip(gridshape), *(func_return_shape if func_return_dim > 0 else ()))
        reshaped_func_result = np.asarray(func_result).reshape(func_result_shape).transpose().copy()
        if func_return_dim > 0:
            reshaped_func_result = np.moveaxis(reshaped_func_result, range(func_return_dim), range(ndim + func_return_dim - 1, ndim - 1, -1))
        
        # Add the results into a list to return
        return_grids.append((reshaped_func_result, reshaped_param_results))
        
        # Find the max value, and the params where that max value are found
        max_flat_index = reshaped_func_result[tuple([slice(None)] * ndim + [0] * func_return_dim)].argmax()
        max_indices = tuple(np.unravel_index(max_flat_index, gridshape))
        max_params = reshaped_param_results[max_indices]
        
        # Set the new bounds for the finer gridsearch
        current_bounds = (np.clip(max_params - grid_spacing, current_bounds[0], current_bounds[1]), 
                          np.clip(max_params + grid_spacing, current_bounds[0], current_bounds[1]))
    
        # If verbose, print the end of the loop
        if verbose:
            print(f"Gridsearch {cycle_number} of {cycles_total}: 100.0%", end="\n")
    
    # If multiprocessing, free shared memory
    if processes > 1:
        result_mem.unlink()
        grid_mem.unlink()

    # Return the array of grids
    return return_grids