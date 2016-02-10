# Objects and helper functions to do stuff with sound
# The logic of using wave is to be able to page through long files.
import numpy as np
import wave
import struct

class WavData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw = wave.open(file_path, 'rb')
        self.n_samples = self.raw.getparams()[3]
        self.s_f = self.raw.getparams()[2]
        self.frame_size = self.raw.getparams()[1]
        self.n_chans = self.raw.getparams()[0]

    def get_chunk(self, start, end, chan_list=[0]):
        frame_to_type = { '2' : 'h', '4' : 'i'}
        # returns a vector with a channel
        n_chans = self.n_chans
        self.raw.setpos(start)
        chunk_bit = self.raw.readframes(end - start)

        data_type = frame_to_type[str(self.frame_size)]
        data = np.zeros((len(chan_list), end - start), dtype=np.dtype(data_type))

        for i, channel in enumerate(chan_list):
            data[i, :] = struct.unpack('<' + str((end -start)*n_chans) + data_type, chunk_bit)[channel::n_chans]
        return data


# class of methods for chunks of a signal
# A chunk is a part of a signal and it is referenced to that signal.
class Chunk:
    '''
    sound:

    chan_list:
    segment:
    '''
    def __init__(self, sound, chan_list=[0], segment=[0, None]):
        '''
        :param sound: Sound where it comes from. Sound has to have methods that return
        n_samples (int, total number of samples)
        s_f (int, sampling frequency in KHZ)
        chunk: a slice of a set of samples across a list of channels.
        :type sound: object
        :param chan_list: list of channels to extract
        :type chan_list: int list
        :param segment: begin and end of segment to extract (in samples)
        :type segment: list
        :return:
        :rtype:
        '''

        self.sound = sound
        self.start = segment[0]
        self.end = segment[1] if segment[1] is not None else sound.n_samples
        self.samples = self.end - self.start
        self.chan_list = chan_list

        # Read the data
        self.data = sound.get_chunk(self.start, self.end, chan_list=chan_list)

    def apply_filter(self, filter_func, *args, **kwargs):
        # Apply some filter function to the chunk of data
        self.data = filter_func(self.data, *args, **kwargs)

    def get_f0(self):
        pass
