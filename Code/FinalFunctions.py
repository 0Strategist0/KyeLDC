"""
Author: Kye Emond
Date: August 25th, 2022


FinalFunctions

File containing several probability functions for Funal_Notebook Spritz Analysis.
Generally poorly to moderately optimized. Could definitely be sped up. 

Variables:
    obs_tdi_t (h5py.File): Observed TDI data
    dt (float): The time interval between adjacent time series samples
    T_OBS (float): The period of observation
    lisa_orbits (ldc.lisa.orbits.Orbits): An Orbits object to store the orbit of LISA for our data
    GB (fastGB.FastGB): A FastGB object holding information about LISA's observation of binaries
    dataf (dict): A dictionary containing A and E TDI ndarrays of frequency-domain data
    fd_window_function (ndarray): Any windowing applied to dataf to remove glitches
    
Methods:
    set_GB: Easy modification of GB from outside module
    circ_conv: Circularly convolve two discrete signals
    window_loglike: The log-likelihood including data windowing
    window_log_posterior: The log-posterior using window_loglike
    full_fstat: Maximize log-likelihood over four parameters
"""

# Imports
import os
import h5py
from ldc.lisa import orbits
import ldc.waveform.fastGB as fastGB
from astropy import units as un
import numpy as np
from numpy.fft import fft, ifft
from typing import Callable

# Load the binary dataset from its file
with h5py.File(f"{os.path.dirname(__file__)}/Data/VGB_Spritz.h5") as file:
    # Get the observed time series
    obs_tdi_t = file["obs/tdi"]
    # Get the sample period
    dt = obs_tdi_t.attrs["dt"]
    # Get the total observation time
    T_OBS = obs_tdi_t["t"].size * dt
    # Define an orbit based off of the data from the file
    lisa_orbits = orbits.Orbits.type(dict({"nominal_arm_length":file["obs/config/nominal_arm_length"][()] * 1e-3 * un.km,
                                        "initial_rotation":file["obs/config/initial_rotation"][()] * un.rad,
                                        "initial_position":file["obs/config/initial_position"][()] * un.rad,
                                        "orbit_type":"analytic"}))
    # Get the galactic binary waveform generator
    GB = fastGB.FastGB(delta_t=dt, T=T_OBS, orbits=lisa_orbits)

# Initialize some global variables for fast function parallelization
dataf = None
fd_window_function = None


# Methods

def set_GB(delta_t:float=dt, T:float=T_OBS, orbits:orbits.Orbits=lisa_orbits):
    """Set GB to parameters for execution of fstat"""
    
    global GB
    GB = fastGB.FastGB(delta_t=delta_t, T=T, orbits=orbits)


def circ_conv(signal1:np.ndarray, signal2:np.ndarray) -> np.ndarray:
    """Calculate the circular convolution of signal1 and signal2 using the fast fourier transform

    Args:
        signal1 (ndarray): 1-dimensional ndarray to be convolved
        signal2 (ndarray): 1-dimensional ndarray to be convolved. Must be same shape as signal1

    Returns:
        ndarray: Circular convolution of signal1 and signal2, same length as signal1 and signal2
    """
    
    return ifft(fft(np.asarray(signal1)) * fft(np.asarray(signal2))) / len(signal1)


def window_loglike(params:np.ndarray, noise_psd:Callable) -> float:
    """Return the natural log of the likelihood function for the given parameters, data and window function, up to an additive constant.
    This function will throw an error if dataf and fd_window_function are not numpy arrays of the same size. 
    The only reason they're not required parameters of this function is to optimize parallelization, 
    since Python multiprocessing pickles every single function argument for every call to the function. 
    
    dataf must be a dictionary containing the frequency domain A and E TDI variables. 
    
    fd_window_function must be the Fourier Transform of the window function applied to dataf (without any windowing applied to itself when transformed). 
    It must contain positive and negative frequencies, in the same format as returned by numpy.fft.fft. 

    Args:
        params (ndarray): ndarray of parameters:
                                log10(Amplitude) [None]
                                Frequency [mHz]
                                FrequencyDerivative [Hz/s]
                                sin(EclipticLatitude) [None]
                                EclipticLongitude [rad]
                                cos(Inclination) [None]
                                PolarizationAngle [rad]
                                InitialPhase [rad]
        noise_psd (Callable): A callable that returns the power spectral density of the noise. 

    Returns:
        float: The natural log of the likelihood, up to an additive constant
    """
    
    # Stick the params into a dictionary for model generation
    waveform_params = {"Amplitude": 10.0 ** params[0], 
                       "Frequency": 1e-3 * params[1], 
                       "FrequencyDerivative": params[2], 
                       "EclipticLatitude": np.arcsin(params[3]), 
                       "EclipticLongitude": params[4], 
                       "Inclination": np.arccos(params[5]), 
                       "Polarization": params[6], 
                       "InitialPhase": params[7]}

    # Generate the waveforms
    X, Y, Z = GB.get_fd_tdixyz(template=waveform_params, oversample=4, tdi2=True)
    
    # Get the indices of the waveform
    indices = slice(X.kmin, X.kmin + len(X))
    negative_indices = slice(-X.kmin - len(X) + 1, -X.kmin + 1)
    
    # Initialize an array to store the A waveform
    A_no_window = np.zeros(len(fd_window_function), dtype=np.complex128)
    # Add in the waveform for positive and negative indices
    A_no_window[indices] = (Z - X) / np.sqrt(2.0)
    A_no_window[negative_indices] = np.conjugate(np.flip((Z - X) / np.sqrt(2.0)))
    # Initialize an array to store the E waveform
    E_no_window = np.zeros(len(fd_window_function), dtype=np.complex128)
    # Add in the waveform for positive and negative indices
    E_no_window[indices] = (X - 2.0 * Y + Z) / np.sqrt(6.0)
    E_no_window[negative_indices] = np.conjugate(np.flip((X - 2.0 * Y + Z) / np.sqrt(6.0)))
    
    # Cicularly convolve the waveforms and the window function to get the modified model
    A = circ_conv(A_no_window, fd_window_function)[indices]
    E = circ_conv(E_no_window, fd_window_function)[indices]
    
    # Get the data over the length of the waveform
    wave_data = {var: dataf[var][indices] for var in ("A", "E")}
    
    # Get the values of the PSD at the frequency
    frequency = 1e-3 * params[1]
    SnA = noise_psd(freq=frequency, option="A")
    SnE = noise_psd(freq=frequency, option="E")
    
    # Calculate the log likelihood up to an additive constant
    return (4.0 / GB.T) * np.real(np.sum(wave_data["A"] * np.conjugate(A)) / SnA
                                  + np.sum(wave_data["E"] * np.conjugate(E)) / SnE
                                  - 0.5 * (np.sum(np.abs(A) ** 2.0) / SnA
                                           + np.sum(np.abs(E) ** 2.0) / SnE))


def window_log_posterior(params:np.ndarray, noise_psd:Callable, a:np.ndarray, b:np.ndarray) -> float:
    """Return the log of the posterior distribution, up to an additive constant. 

    Args:
        params (ndarray): A numpy array of sampling parameters:
                                log10(Amplitude) [None]
                                Frequency [mHz]
                                FrequencyDerivative [log10(Hz/s)]
                                sin(EclipticLatitude) [None]
                                EclipticLongitude [rad]
                                cos(Inclination) [None]
                                PolarizationAngle [rad]
                                InitialPhase [rad]
        noise_psd (Callable): A callable that returns the power spectral density of the noise. 
        a (ndarray): The minimum of the uniform distribution. 
        b (ndarray): The maximum of the uniform distribution.

    Returns:
        float: The log of the posterior distribution, up to and additive constant
    """
    
    # If the parameters are within their bounds, return the log likelihood, otherwise, -inf
    if np.all(a <= params) and np.all(b >= params):
        return window_loglike(params, noise_psd)
    else:
        return -np.inf


def full_fstat(params:np.ndarray, dataf:dict, Tobs:float, psd:Callable, fdot:float=1e-17):
    """Calculate the natural logarithm of the likelihood function, maximized over 
    amplitude, inclination, polarization angle, and initial phase for any given 
    frequency, frequency derivative, and sky position, as well as the maximized values of those parameters. 

    Args:
        params (ndarray): ndarray of parameters at which to maximize the likelihood over the remaining parameters:
                                Initial Frequency [mHz]
                                Ecliptic Latitude [rad]
                                Ecliptic Longitude [rad]
        dataf (dict): A dictionary containing A, E, and T TDI waveforms in the frequency domain
        Tobs (float): The period of observation
        psd (Callable): A callable that returns the power spectral density of the noise at a given frequency for each TDI variable
        fdot (float, optional): The frequency derivative parameter. Defaults to 1e-17.

    Returns:
        ndarray: An ndarray consisting of:
                        Maximized log-likelihood (F-Statistic) [None]
                        Maximized Log10 of Amplitude [None]
                        Initial Frequency passed to function [mHz]
                        Frequency Derivative passed to function [Hz/s]
                        Ecliptic Latitude passed to function [rad]
                        Ecliptic Longitude passed to function [rad]
                        Maximized Cosine of Inclination Angle [None]
                        Maximized Polarization Angle [rad]
                        Maximized Initial Phase [rad]
    """
    
    # Get the individual parameters from the whole array
    fr_i = params[0]
    bet_i = params[1]
    lam_i = params[2]
    
    # Get the frequency change between adjacent samples
    df = 1.0 / Tobs

    # Pick an arbitrary amplitude to use for waveform generation
    amp = 1.e-21
    lAmp = np.log10(amp)
    
    # Generate a waveform assuming + polarization
    pgb1 = {'Amplitude': amp,# "strain"
            'EclipticLatitude': bet_i, # "radian"
            'EclipticLongitude': lam_i,# "radian"
            'Frequency': fr_i*1.e-3, #"Hz"
            'FrequencyDerivative': fdot,# "Hz^2"
            'Inclination': 0.5*np.pi,# "radian"
            'InitialPhase': 0.0, #"radian"
            'Polarization': 0.0} #"radian"
    Xf1, Yf1, Zf1 = GB.get_fd_tdixyz(template=pgb1, oversample=4)
    
    # Convert the + polarized waveform to A, E and T TDI variables
    Af1 = (Zf1 - Xf1) / np.sqrt(2.0)
    Ef1 = (Xf1 - 2.0 * Yf1 + Zf1) / np.sqrt(6.0)
    Tf1 = (Xf1 + Yf1 + Zf1) / np.sqrt(3.0)
    
    # Generate a waveform assuming x polarization
    pgb2 = dict({'Amplitude': amp,# "strain"
            'EclipticLatitude': bet_i, # "radian"
            'EclipticLongitude': lam_i,# "radian"
            'Frequency': fr_i*1.e-3, #"Hz"
            'FrequencyDerivative': fdot,# "Hz^2"
            'Inclination': 0.5*np.pi,# "radian"
            'InitialPhase': 0.0, #"radian"
            'Polarization': 0.25*np.pi}) #"radian"
    Xf2, Yf2, Zf2 = GB.get_fd_tdixyz(template=pgb2, oversample=4)
    
    # Convert the polarized waveform to A, E and T TDI variables
    Af2 = (Zf2 - Xf2) / np.sqrt(2.0)
    Ef2 = (Xf2 - 2.0 * Yf2 + Zf2) / np.sqrt(6.0)
    Tf2 = (Xf2 + Yf2 + Zf2) / np.sqrt(3.0)
    
    # Get the power spectral density for each TDI variable at the frequency we're interested in
    SnA = psd(freq=fr_i * 1.e-3, option='A')
    SnE = psd(freq=fr_i * 1.e-3, option='E')
    SnT = psd(freq=fr_i * 1.e-3, option='T')

    # Grab the indices of the waveform in frequency space
    ib = Xf1.kmin
    ie = Xf1.kmin + len(Xf1)
    
    # Calculate a bunch of terms to be used in the final F-Stat and parameter calculations
    U = np.sum(np.absolute(Af1) ** 2) / SnA + np.sum(np.absolute(Ef1) ** 2) / SnE + np.sum(np.absolute(Tf1) ** 2) / SnT
    V = np.sum(np.absolute(Af2) ** 2) / SnA + np.sum(np.absolute(Ef2) ** 2) / SnE + np.sum(np.absolute(Tf2) ** 2) / SnT
    W = np.sum(Af1 * np.conjugate(Af2)) / SnA + np.sum(Ef1 * np.conjugate(Ef2)) / SnE + np.sum(Tf1 * np.conjugate(Tf2)) / SnT
    Nu = np.sum(dataf["A"][ib:ie] * np.conjugate(Af1)) / SnA + np.sum(dataf["E"][ib:ie] * np.conjugate(Ef1)) / SnE + np.sum(dataf["T"][ib:ie] * np.conjugate(Tf1)) / SnT
    Nv = np.sum(dataf["A"][ib:ie] * np.conjugate(Af2)) / SnA + np.sum(dataf["E"][ib:ie] * np.conjugate(Ef2)) / SnE + np.sum(dataf["T"][ib:ie] * np.conjugate(Tf2)) / SnT
    Del = U*V - np.absolute(W)**2
    
    # Calculate the maximized log likelihood, up to an additive constant
    Fstat = 2.0 * df * (V * np.absolute(Nu) ** 2 + U * np.absolute(Nv) ** 2 - 2.0 * np.real(W * Nu * np.conjugate(Nv))) / (Del)

    # Go through some math to calculate the amplitudes of the different signal components
    a1c = (V * Nu - np.conjugate(W) * Nv) / Del
    a2c = (U * Nv - W * Nu) / Del

    a1 = np.real(a1c)
    a3 = np.imag(a1c)
    a2 = np.real(a2c)
    a4 = np.imag(a2c)

    A = a1 ** 2 + a2 ** 2 + a3 ** 2 + a4 ** 2
    D = a1 * a4 - a2 * a3

    # Get the amplitudes of the + and x polarized waveforms
    Ap = 0.5*(np.sqrt(A + 2.0*D) + np.sqrt(A - 2.0*D))
    Ac = 0.5*(np.sqrt(A - 2.0*D) - np.sqrt(A + 2.0*D))

    # Use the calculated values to determine the maximum in each of the parameters
    Amp = 0.5 * (Ap + np.sqrt(Ap * Ap - Ac * Ac))
    cos_i = 0.5 * Ac / Amp
    phi0 = 0.5 * np.arctan2(2.0 * (a1 * a3 + a2 * a4), (a1 * a1 + a2 * a2 - a3 * a3 - a4 * a4))
    psi = 0.25 * np.arctan2(2.0 * (a1 * a2 + a3 * a4), (a1 * a1 + a3 * a3 - a2 * a2 - a4 * a4))
    psi = psi - np.pi
    phi0 = phi0 - np.pi
    if (psi < 0.0):
        psi += 2.0 * np.pi
    if (phi0 < 0.0):
        phi0 += 2.0 * np.pi

    l_Amp = np.log10(Amp) + lAmp

    # Return an array of the log likelihood and the other source parameters
    return (np.array([Fstat, l_Amp, fr_i, fdot, bet_i, lam_i, cos_i, psi, phi0]))