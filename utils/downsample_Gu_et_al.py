import h5py
from scipy.signal import decimate
import numpy as np
import os


if __name__ == '__main__':

    sjNum = int(input('Sj Number: '))
    rootDir = 'D:/EEG_Forecast/Data/Gu_et_al_2024_SSVEP_dataset/Data/'
    dataDir = os.path.join(rootDir, f'data_s{sjNum:d}_64.mat')

    data_file = h5py.File(dataDir)

    # block x stimulation frequency x time x channels x conditions (i.e., modulation depths; low and high luminance ratios)
    # stimulation frequencies are from 1-60 in increments of 1 Hz
    data = data_file['datas']

    # downsample to 250 Hz from 1000 Hz
    resampled_data = decimate(data, q=4, axis=2, ftype='iir')

    # save as a numpy file
    np.save(os.path.join(rootDir, f'data_s{sjNum:d}_64_down.npy'), resampled_data)



