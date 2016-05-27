import pdb
import sys
import pandas as pd
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib
import scipy.signal as sg
import math
import scipy as sp
import socket
import os
import wave
import struct


def band_pass_filter(chunk, hp_b, hp_a, lp_b, lp_a):
    chunk_hi = sg.filtfilt(hp_b, hp_a, chunk)
    chunk_filt = sg.filtfilt(lp_b, lp_a, chunk_hi)


def main():
    comp_name=socket.gethostname()
    print 'Computer: ' + comp_name
    if comp_name == 'chim':
        sys.path.append('C:\Users\GentnerLab\Documents\Experiment\scripts\sound_tools')
        experiment_folder = os.path.join('C:\Users\GentnerLab\Documents\Experiment')
        r = os.path.join('C:\Users\GentnerLab\Documents\Experiment')
        raw_data_folder = os.path.join(experiment_folder, 'raw_data')

    #file structure
    bird_id = 'zk_zf_001'
    sess = 2
    rec = 'a'
    run = 1

    raw_file_folder = os.path.join(raw_data_folder, bird_id, str(sess).zfill(3))
    raw_file_name = rec + '_' + str(run).zfill(2) + '_song.wav'
    raw_file_path = os.path.join(raw_file_folder, raw_file_name)

    #test soundtools
    from soundtools import WavData, Chunk
    sound = WavData(raw_file_path)

    #get a piece, filter it, plot it
    window_t = 234700 #time window in msec
    window_size = 1000 # window len in msec

    window_t_samples = int(round(window_t*sound.s_f/1000.))
    window_samples = int(round(window_size*sound.s_f/1000.))
    window_samples = min(window_samples, sound.n_samples - window_t_samples)

    window = Chunk(sound, segment=[window_t_samples, window_t_samples + window_samples])
    # plt.plot(window.data)
    # window2 = window2 = Chunk(sound, segment=[window_t_samples, window_t_samples + window_samples], chan_list = [0, 1])
    # plt.plot(window2.data);

    s_f = window.sound.s_f

    filt_lo = 22000 #Hz
    filt_hi = 50 #Hz
    hp_b, hp_a = sg.butter(4, filt_hi/(s_f/2.), btype='high')
    lp_b, lp_a = sg.butter(4, filt_lo/(s_f/2.), btype='low')

    print window_samples
    window.apply_filter(band_pass_filter, hp_b, hp_a, lp_b, lp_a)

if __name__=="__main__":
    main()