# Wav files operations module
# These are functions that operate on wav files
from scipy.io import wavfile
from scipy.signal import resample
import numpy as np
import os


def resample_wav(in_file, rate=0, out_path='', out_file='', mono_output=True):

    """
    in_file : complete (file, path) input file
    rate    : resampled file rate (samples/sec) (default 0: do not resample)

    out_path : path to output of file (default input path)
    out_file : output file name (default input_file_name_(rate).extension_of_input_file)
    mono_output: force single channel output file
    """
    in_file_name = os.path.split(in_file)[1]
    in_path = os.path.split(in_file)[0]

    sf, data = wavfile.read(in_file)
    new_len = int(round(data.shape[0] * rate / sf))

    if mono_output:
        data = data[:, 0] if len(data.shape) > 1 else data

    if rate == 0:
        rate = sf
        data_out = np.array(np.round(data), dtype=np.int16)
    else:
        data_out = np.array(np.round(resample(data, new_len)), dtype=np.int16)

    if out_path == '':
        out_path = in_path
    if out_file == '':
        out_file = in_file_name.split('.')[0] + '_' + str(rate).zfill(4) + '.' + in_file_name.split('.')[1]

    print os.path.join(out_path, out_file)
    wavfile.write(os.path.join(out_path, out_file), rate, data_out)
