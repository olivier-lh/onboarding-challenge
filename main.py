
# -*- coding: utf-8 -*-
"""
Estimate Relaxation from Band Powers
This example shows how to buffer, epoch, and transform EEG data from a single
electrode into values for each of the classic frequencies (e.g. alpha, beta, theta)
Furthermore, it shows how ratios of the band powers can be used to estimate
mental state for neurofeedback.
The neurofeedback protocols described here are inspired by
*Neurofeedback: A Comprehensive Review on System Design, Methodology and Clinical Applications* by Marzbani et. al
Adapted from https://github.com/NeuroTechX/bci-workshop
"""

import numpy as np  # Module that simplifies computations on matrices
import matplotlib.pyplot as plt  # Module used for plotting
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
from utils.utils import update_buffer, get_last_data, compute_band_powers, read_eeg_file

# Handy little enum to make code more readable

class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3

""" EXPERIMENTAL PARAMETERS """
# Modify these to change aspects of the signal processing

# Length of the EEG data buffer (in seconds)
# This buffer will hold last n seconds of data and be used for calculations
BUFFER_LENGTH = 5

# Length of the epochs used to compute the FFT (in seconds)
EPOCH_LENGTH = 1

# Amount of overlap between two consecutive epochs (in seconds)
OVERLAP_LENGTH = 0.8

# Amount to 'shift' the start of each next consecutive epoch
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

# Index of the channel(s) (electrodes) to be used
# 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
INDEX_CHANNEL = [0]

# Thresholds to consider a blink
UPPER_BLINK_LIMIT = 110
LOWER_BLINK_LIMIT = -300

if __name__ == "__main__":

    """ 1. CONNECT TO EEG STREAM """

    # Search for active LSL streams
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    # Get the stream info and description
    info = inlet.info()
    description = info.desc()

    # Get the sampling frequency
    # This is an important value that represents how many EEG data points are
    # collected in a second. This influences our frequency band calculation.
    # for the Muse 2016, this should always be 256
    fs = int(info.nominal_srate())

    """ 2. INITIALIZE BUFFERS """

    # Initialize raw EEG data buffer
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
    filter_state = None  # for use with the notch filter

    # Compute the number of epochs in "buffer_length"
    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                              SHIFT_LENGTH + 1))

    # Initialize the band power buffer (for plotting)
    # bands will be ordered: [delta, theta, alpha, beta]
    band_buffer = np.zeros((n_win_test, 4))

    """ 3. GET DATA """

    # The try/except structure allows to quit the while loop by aborting the
    # script with <Ctrl-C>
    print('Press Ctrl-C in the console to break the while loop.')

    try:
        # The following loop acquires data, computes band powers, and calculates neurofeedback metrics based on those band powers
        debouncer = 0
        debouncing = False
        while True:
            """ 3.1 ACQUIRE DATA """
            # Obtain EEG data from the LSL stream
            eeg_data, timestamp = inlet.pull_chunk(
                timeout=1, max_samples=int(SHIFT_LENGTH * fs))

            # Only keep the channel we're interested in
            ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]

            # Update EEG buffer with the new data
            eeg_buffer, filter_state = update_buffer(
                eeg_buffer, ch_data, notch=True,
                filter_state=filter_state)
            last = eeg_buffer[len(eeg_buffer) - 1]
            if (last < 10 or last > 60) and not debouncing:
                print("blink!")
                debouncing = True
            elif debouncing:
                debouncer += 1
            if debouncer > 3:
                debouncing = False
                debouncer = 0



    except KeyboardInterrupt:
        print('Closing!')

# Function to find blinks in csv record
def find_blink_in_record() -> None:
    blinking = []
    blink = False
    eeg_data = read_eeg_file("./eeg-records/eeg_and_eye_blinks_recording.csv")
    timestamps = eeg_data['timestamps']
    tp10 = eeg_data['TP10']
    i = 0
    normal_count = 0
    while i < len(tp10):
        if tp10[i] < -300 or tp10[i] > 110:
            if not blink:
                blink = True
            else:
                normal_count = 0
        elif (50 > tp10[i] > -50) and blink:
            normal_count += 1
            if normal_count > 100:
                blink = False
                normal_count = 0
        blinking.append(1) if blink else blinking.append(0)
        i += 1
