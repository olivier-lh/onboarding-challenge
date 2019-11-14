from pylsl import StreamInlet, resolve_byprop
from muselsl.constants import LSL_SCAN_TIMEOUT, LSL_EEG_CHUNK
from time import time
from collections import OrderedDict
from csv import reader


def read_eeg_file(csv_file_path):
    eeg_data = OrderedDict()

    with open(csv_file_path, newline='') as eeg_file:
        csv_reader = reader(eeg_file, delimiter=',')
        eeg_data = OrderedDict({key:[] for key in next(csv_reader)})  # Use the headers as key of eeg_data dict
        for samples in csv_reader:
            for i, key in enumerate(eeg_data.keys()):
                eeg_data[key].append(float(samples[i])) # Add the sample to its channel list
    
    return eeg_data


def debounce_signal(signal, debouncing_time=0.1, period=1/256):
    rising_edges = []
    falling_edges = []
    last_value = signal[0]

    # Find the edge indexes.
    for i, data in enumerate(signal):
        if data is not last_value:
            if last_value:
                falling_edges.append(i)
            else:
                rising_edges.append(i)
        last_value = data

    for i in falling_edges:
        next_rising_edge = next((index for index in rising_edges if index > i), None)
        if next_rising_edge is None:
            break
        
        # If the next rising edge is within the debouncing period, then set the blink signal to 1.
        falling_to_rising_idx_span = next_rising_edge - i
        if falling_to_rising_idx_span * period < debouncing_time:
            signal[i: next_rising_edge] = [True] * falling_to_rising_idx_span
    return signal

def scale_duty_cycle(signal, factor):
    rising_edges = []
    falling_edges = []

    # Look for all pairs of rising and falling edges.
    for i, is_blinking in enumerate(signal):
        if is_blinking and not signal[i-1]:
            rising_edges.append(i)
        elif not is_blinking and signal[i-1]:
            falling_edges.append(i)

    duty_domains = [(rising_edges[i], falling_edges[i]) for i in range(len(rising_edges))]

    for domain in duty_domains:
        domain_span = domain[1] - domain[0]
        scaled_domain_span = domain_span * factor
        span_to_set = int((scaled_domain_span - domain_span)/2)
        upper_bound = domain[1]+span_to_set
        lower_bound = domain[0]-span_to_set
        signal[lower_bound: upper_bound] = [True]*(upper_bound-lower_bound)
    return signal


def print_eeg_callback(timestamps, eeg_data):
    for i, data in enumerate(eeg_data):
        print(timestamps[i], data)


def acquire_eeg(duration, callback=print_eeg_callback, eeg_chunck=LSL_EEG_CHUNK):

    DATA_SOURCE = "EEG"

    print("Looking for a %s stream..." % (DATA_SOURCE))
    streams = resolve_byprop('type', DATA_SOURCE, timeout=LSL_SCAN_TIMEOUT)

    if len(streams) == 0:
        print("Can't find %s stream." % (DATA_SOURCE))
        return

    print("Started acquiring data.")
    inlet = StreamInlet(streams[0], max_chunklen=eeg_chunck)

    info = inlet.info()
    description = info.desc()
    Nchan = info.channel_count()

    ch = description.child('channels').first_child()
    ch_names = [ch.child_value('label')]
    for i in range(1, Nchan):
        ch = ch.next_sibling()
        ch_names.append(ch.child_value('label'))

    timestamps = []
    t_init = time()
    time_correction = inlet.time_correction()

    print('Start acquiring at time t=%.3f' % t_init)
    print('Time correction: ', time_correction)

    while (time() - t_init) < duration:
        try:
            chunk, timestamps = inlet.pull_chunk(timeout=1.0,
                                                 max_samples=eeg_chunck)

            if timestamps:
                samples = {key: [sample[i] for sample in chunk] for i, key in enumerate(ch_names)}
                callback(timestamps, samples)
        except KeyboardInterrupt:
            break

    print('Acquisition is done')

# -*- coding: utf-8 -*-
"""
Muse LSL Example Auxiliary Tools
These functions perform the lower-level operations involved in buffering,
epoching, and transforming EEG data into frequency bands
@author: Cassani
"""

import os
import sys
from tempfile import gettempdir
from subprocess import call

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from scipy.signal import butter, lfilter, lfilter_zi


NOTCH_B, NOTCH_A = butter(4, np.array([55, 65]) / (256 / 2), btype='bandstop')


def epoch(data, samples_epoch, samples_overlap=0):
    """Extract epochs from a time series.
    Given a 2D array of the shape [n_samples, n_channels]
    Creates a 3D array of the shape [wlength_samples, n_channels, n_epochs]
    Args:
        data (numpy.ndarray or list of lists): data [n_samples, n_channels]
        samples_epoch (int): window length in samples
        samples_overlap (int): Overlap between windows in samples
    Returns:
        (numpy.ndarray): epoched data of shape
    """

    if isinstance(data, list):
        data = np.array(data)

    n_samples, n_channels = data.shape

    samples_shift = samples_epoch - samples_overlap

    n_epochs = int(
        np.floor((n_samples - samples_epoch) / float(samples_shift)) + 1)

    # Markers indicate where the epoch starts, and the epoch contains samples_epoch rows
    markers = np.asarray(range(0, n_epochs + 1)) * samples_shift
    markers = markers.astype(int)

    # Divide data in epochs
    epochs = np.zeros((samples_epoch, n_channels, n_epochs))

    for i in range(0, n_epochs):
        epochs[:, :, i] = data[markers[i]:markers[i] + samples_epoch, :]

    return epochs


def compute_band_powers(eegdata, fs):
    """Extract the features (band powers) from the EEG.
    Args:
        eegdata (numpy.ndarray): array of dimension [number of samples,
                number of channels]
        fs (float): sampling frequency of eegdata
    Returns:
        (numpy.ndarray): feature matrix of shape [number of feature points,
            number of different features]
    """
    # 1. Compute the PSD
    winSampleLength, nbCh = eegdata.shape

    # Apply Hamming window
    w = np.hamming(winSampleLength)
    dataWinCentered = eegdata - np.mean(eegdata, axis=0)  # Remove offset
    dataWinCenteredHam = (dataWinCentered.T * w).T

    NFFT = nextpow2(winSampleLength)
    Y = np.fft.fft(dataWinCenteredHam, n=NFFT, axis=0) / winSampleLength
    PSD = 2 * np.abs(Y[0:int(NFFT / 2), :])
    f = fs / 2 * np.linspace(0, 1, int(NFFT / 2))

    # SPECTRAL FEATURES
    # Average of band powers
    # Delta <4
    ind_delta, = np.where(f < 4)
    meanDelta = np.mean(PSD[ind_delta, :], axis=0)
    # Theta 4-8
    ind_theta, = np.where((f >= 4) & (f <= 8))
    meanTheta = np.mean(PSD[ind_theta, :], axis=0)
    # Alpha 8-12
    ind_alpha, = np.where((f >= 8) & (f <= 12))
    meanAlpha = np.mean(PSD[ind_alpha, :], axis=0)
    # Beta 12-30
    ind_beta, = np.where((f >= 12) & (f < 30))
    meanBeta = np.mean(PSD[ind_beta, :], axis=0)

    feature_vector = np.concatenate((meanDelta, meanTheta, meanAlpha,
                                     meanBeta), axis=0)

    feature_vector = np.log10(feature_vector)

    return feature_vector


def nextpow2(i):
    """
    Find the next power of 2 for number i
    """
    n = 1
    while n < i:
        n *= 2
    return n


def compute_feature_matrix(epochs, fs):
    """
    Call compute_feature_vector for each EEG epoch
    """
    n_epochs = epochs.shape[2]

    for i_epoch in range(n_epochs):
        if i_epoch == 0:
            feat = compute_band_powers(epochs[:, :, i_epoch], fs).T
            # Initialize feature_matrix
            feature_matrix = np.zeros((n_epochs, feat.shape[0]))

        feature_matrix[i_epoch, :] = compute_band_powers(
            epochs[:, :, i_epoch], fs).T

    return feature_matrix


def get_feature_names(ch_names):
    """Generate the name of the features.
    Args:
        ch_names (list): electrode names
    Returns:
        (list): feature names
    """
    bands = ['delta', 'theta', 'alpha', 'beta']

    feat_names = []
    for band in bands:
        for ch in range(len(ch_names)):
            feat_names.append(band + '-' + ch_names[ch])

    return feat_names


def update_buffer(data_buffer, new_data, notch=False, filter_state=None):
    """
    Concatenates "new_data" into "data_buffer", and returns an array with
    the same size as "data_buffer"
    """
    if new_data.ndim == 1:
        new_data = new_data.reshape(-1, data_buffer.shape[1])

    if notch:
        if filter_state is None:
            filter_state = np.tile(lfilter_zi(NOTCH_B, NOTCH_A),
                                   (data_buffer.shape[1], 1)).T
        new_data, filter_state = lfilter(NOTCH_B, NOTCH_A, new_data, axis=0,
                                         zi=filter_state)

    new_buffer = np.concatenate((data_buffer, new_data), axis=0)
    new_buffer = new_buffer[new_data.shape[0]:, :]

    return new_buffer, filter_state


def get_last_data(data_buffer, newest_samples):
    """
    Obtains from "buffer_array" the "newest samples" (N rows from the
    bottom of the buffer)
    """
    new_buffer = data_buffer[(data_buffer.shape[0] - newest_samples):, :]

    return new_buffer
