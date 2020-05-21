# Author:       Brian Murphy
# Date started: 29/04/2020
# Last updated: <21/05/2020 08:15:31 (BrianM)>

import pandas as pd
import numpy as np
import xgboost as xgb
import mne  # Read the edf file
from scipy.signal import filtfilt, butter, resample_poly
import matlab.engine


def make_tfd_dataframe(tfd_data_flat):
    """
    This is just a quick function to make a nice DataFrame from the data. It makes it easy to manage.

    Syntax: tfd_data_df = make_tfd_dataframe(tfd_data_flat)

    Inputs:
        tfd_data_flat    - Dictionary containing information about the file and channel along with TFD annotations and
                           TFD values

    Outputs:
        tfd_data_df     - A pandas DataFrame of all the data combined

    Example:

    """
    columns = ['baby_ID', 'channel', 'Fs', 'anno']
    for f in range(1, tfd_data_flat['tfd'].shape[1] + 1):
        columns.append('tfd_f' + str(f))
    tfd_data_df = pd.DataFrame(columns=columns, index=np.arange(0, tfd_data_flat['tfd'].shape[0]))
    tfd_data_df.loc[:, ['baby_ID', 'channel', 'Fs']] = [tfd_data_flat['baby_ID'], tfd_data_flat['channel'],
                                                        tfd_data_flat['Fs']]

    tfd_data_df.loc[:, 'anno'] = tfd_data_flat['anno']

    tfd_data_df.loc[:, columns[4:]] = tfd_data_flat['tfd']

    return tfd_data_df



def resample_down_BM(anno, fs_ratio):
    """
    Syntax: anno = resample_down_BM(anno, fs_ratio)

    Inputs:
        anno        - Numpy array of annotations
        fs_ratio    - The downsample ratio

    Outputs:
        anno        - The downsampled annotations

    Example:

    """

    if any(np.isnan(anno)):
        anno[np.isnan(anno)] = 0.5
    anno = resample_poly(anno, 1, fs_ratio)
    anno = np.round(anno)
    return anno


def prepare_tfd_data(tfd_data, params):
    """
    The purpose of this function is to unpack the annotations and TFDs from lists of lists to single numpy arrays.

    Syntax: tfd_data_flat = prepare_tfd_data(tfd_data, params)

    Inputs:
        tfd_data        - Dinctionary containing file/ channel details along with the eeg annotations, eeg_epochs and
                          the generated TFDs
        params          - Dictionary containing parameters used to generate the TFD

    Outputs:
        tfd_data_flat   - Dictionary where annotations and TFD and now in single numpy arrays

    Example:

    """

    L_epoch = params['epoch_seconds'] * tfd_data['Fs']
    L_tfd_time = np.shape(tfd_data['tfds_epoch'][0])[0]
    fs_ratio = int(L_epoch / L_tfd_time)


    # ---------------------------------------------------------------------
    #  resample annotation to lower value
    # ---------------------------------------------------------------------

    # ignore the last epoch as it was most likely padded with zeros:
    anno = tfd_data['anno'][:, :-1]
    # concatenate in time:
    anno = anno.flatten('F')

    # join all the TFDs into a single numpy array
    tfd = np.concatenate(tfd_data['tfds_epoch'])

    # downsample annotations so they align with the generated TFDs
    anno = resample_down_BM(anno, fs_ratio)



    # transfer to array structure
    tfd_data_flat = dict()
    tfd_data_flat['baby_ID'] = tfd_data['baby_ID']
    tfd_data_flat['channel'] = tfd_data['channel']
    tfd_data_flat['Fs'] = tfd_data['Fs']
    tfd_data_flat['anno'] = anno
    tfd_data_flat['tfd'] = tfd

    return tfd_data_flat




def padwin(w, Npad):
    """
    Pad window to Npad.
    Presume that positive window indices are first.
     USE: w_pad=padWin(w,Npad)

     INPUT:
           w    = window (vector) of length N
           Npad = pad window to length N (Npad>N)
     OUTPUT:
           w_pad = window of length N zeropadded to length Npad

    When N is even use method described in [1]
      References:
        [1] S. Lawrence Marple, Jr., Computing the discrete-time analytic
        signal via FFT, IEEE Transactions on Signal Processing, Vol. 47,
        No. 9, September 1999, pp.2600--2603.
    """
    w_pad = np.zeros(Npad).astype(w.dtype)
    N = len(w)
    Nh = int(np.floor(N / 2))

    if Npad < N:
        raise ValueError('npad is less than n')

    # Trivial case:
    if N == Npad:
        return w

    # For N odd:
    if N % 2 == 1:
        n = np.arange(0, Nh + 1)
        w_pad[n] = w[n]
        n = np.arange(1, Nh + 1)
        w_pad[Npad - n] = w[N - n]

    # For N even:
    # split the Nyquist frequency in two and distribute over positive
    # and negative indices.
    else:
        n = np.arange(0, Nh)
        w_pad[n] = w[n]
        w_pad[Nh] = w[Nh] / 2

        n = np.arange(1, Nh)
        w_pad[Npad - n] = w[N - n]
        w_pad[Npad - Nh] = w[Nh] / 2

    return w_pad



def gen_tfd(x, Fs=64, do_diff=1, zero_pad_seconds=0, oversample=0, d_win_size=31, l_win_size=171, Nfreq_in=256, path_to_TFD_package=None):
    """
    This is the function that actually generates the TFD from the eeg epoch x

    Syntax: tfd = gen_tfd(x, Fs=64, do_diff=1, zero_pad_seconds=0, oversample=0, d_win_size=31,
                                    l_win_size=171, Nfreq_in=256, path_to_TFD_package=None)

    Inputs:
        x                   - 1-D numpy array containing the EEG data
        Fs                  - Sampling rate of the EEG (default = 64)
        do_diff             - Flag used to decide to do pre-whitening filter (default = 1)
        zero_pad_seconds    - Seconds to zero pad the signal (default = 0)
        oversample          - (default = 0)
        d_win_size          - Doppler window length (default = 31)
        l_win_size          - Lag window length (default = 171)
        Nfreq_in            - (default = 256)
        path_to_TFD_package - Path to package that computes the TFD (default = None, should cause error)

    Outputs:
        tfd                 - 2-D numpy array containing the TFD corresponding to epoch x

    Example:

    """


    zero_pad_frequency = 1  # To remove wrap-around effects in the frequency domain

    epoch_length = 32 + 4  # assuming epoch length is 32 seconds


    if do_diff:
        y = np.concatenate(([0], np.diff(x)))
    else:
        y = x

    # zero-pad in the time-domain?
    if zero_pad_seconds:
        y = np.concatenate((np.zeros(zero_pad_seconds * Fs), y, np.zeros(zero_pad_seconds * Fs)))

    doppler_win = [d_win_size, 'tukey', 0.9]
    lag_win = [l_win_size, 'hann']


    # oversample or not
    if oversample:
        fs_time = 2 * Fs
    #     Nfreq = N;
    else:
        fs_time = 4
        #     Nfreq = 32;
        Nfreq = Nfreq_in

    Ntime = (epoch_length + 2 * zero_pad_seconds) * fs_time

    if zero_pad_frequency:
        y = np.real_if_close(np.fft.ifft(padwin(np.fft.fft(y), len(y) * 2))) * 2
        Nfreq = 2 * Nfreq



    # generate Q-TFD with separable kernel:
    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath(path_to_TFD_package))

    y_mat = matlab.double(y.tolist())
    doppler_win_mat = (float(doppler_win[0]), doppler_win[1], float(doppler_win[2]))
    lag_win_mat = (float(lag_win[0]), lag_win[1])
    Ntime_mat = float(Ntime)
    Nfreq_mat = float(Nfreq)
    tfd = np.array(eng.sep_gdtfd(y_mat, doppler_win_mat, lag_win_mat, Ntime_mat, Nfreq_mat))


    if zero_pad_frequency:
        tfd = tfd[:, 0:(int(Nfreq/2) + 1)]

    # careful here as not generalised (assuming TFD fs_time=2 Hz):
    if zero_pad_seconds:
        tpad = zero_pad_seconds * fs_time
        tfd = tfd[tpad:(tfd.shape[0] - tpad), :]



    # ---------------------------------------------------------------------
    # percentage below zero:
    # ---------------------------------------------------------------------
    PRC_ZERO = 0
    if PRC_ZERO:
        N_tots = np.size(tfd)

        prc_zeros = 100 * len(np.where(tfd < 0)[0]) / N_tots
        all_zero = np.sum(tfd)
        print('Percentage of <0 in TFD = %f ' % prc_zeros)
        print('Total TFD energy = %f' % all_zero)

    return tfd



def buffer(x, n, p=0):
    """
    Parameters
    ----------
    x: ndarray
        Signal array
    n: int
        Number of data segments
    p: int
        Number of values to overlap

    Returns
    -------
    result : (n,m) ndarray
        Buffer array created from X
    """

    d = n - p
    m = len(x)//d

    if m * d != len(x):
        m = m + 1

    Xn = np.zeros(d*m)
    Xn[:len(x)] = x

    Xn = np.reshape(Xn, (m, d))
    Xne = np.concatenate((Xn, np.zeros((1, d))))
    Xn = np.concatenate((Xn, Xne[1:, 0:p]), axis=1)

    return np.transpose(Xn)


def gen_TFD_EEG_epochs(d_win_len=31, l_win_len=171, Nfreq_in=256, data=None, params=None, path_to_TFD_package=None):
    """
    This is the function is used to arrange the EEG data and generate the TFDs after the EEG is divided into epochs.

    Syntax: tfd = gen_TFD_EEG_epochs(d_win_len=31, l_win_len=171, Nfreq_in=256, data=None, params=None,
                                     path_to_TFD_package=None)

    Inputs:
        d_win_len          - Doppler window length (default = 31)
        l_win_len          - Lag window length (default = 171)
        Nfreq_in            - (default = 256)
        data                - Dictionary containing eeg_data (numpy array), anno (numpy array), Fs, channel, baby_ID
        path_to_TFD_package - Path to package that computes the TFD (default = None, should cause error)

    Outputs:
        tfd_data                 - Dictionary with the following: ptient id, channel used for tfd, sampling frequency,
                                   eeg epochs, annotations and the TFDs

    Example:

    """

    Fs = data['Fs']
    x = data['eeg_data']
    anno = data['anno']
    l_epoch = int(params['epoch_seconds'] * Fs)
    n_pad = np.arange(0, int(params['tfd_pad_seconds'] * Fs))


    # Bandpass filter
    b, a = butter(5, 30 / (Fs / 2), 'lowpass')  # Low pass filter
    y = filtfilt(np.array(b), np.array(a), x)
    b, a = butter(5, 0.5 / (Fs / 2), 'highpass')  # High pass filter
    y = filtfilt(np.array(b), np.array(a), y)

    # Split the signal into epochs
    x_epoch = buffer(y, l_epoch)
    x_epoch_anno = buffer(anno, l_epoch)



    # Split into short-duration epochs (32 seconds)
    tfds = []

    for n in np.arange(0, x_epoch.shape[1] - 1):
        # pad the signal either side to avoid time-wrapping
        # in the TFD
        if n > 0:
            x = np.concatenate((x_epoch[-len(n_pad):, n - 1], x_epoch[:, n], x_epoch[n_pad, n + 1]))
        else:
            x = np.concatenate((np.zeros(len(n_pad)), x_epoch[:, n], x_epoch[n_pad, n + 1]))


        # ---------------------------------------------------------------------
        # calculate the TFD for each epoch:
        # ---------------------------------------------------------------------

        tfdd = gen_tfd(x, Fs=Fs, do_diff=1, d_win_size=d_win_len, l_win_size=l_win_len, Nfreq_in=Nfreq_in, path_to_TFD_package=path_to_TFD_package)

        # assuming that sampling frequency in time for TFD is 2 Hz:
        fs_tfd_time = 4
        # subtract padding:
        tfds.append(tfdd[(params['tfd_pad_seconds'] * fs_tfd_time): (tfdd.shape[0] - (params['tfd_pad_seconds'] * fs_tfd_time)), :])


    # ---------------------------------------------------------------------
    #  collect in structure:
    # ---------------------------------------------------------------------
    tfd_data = dict()
    tfd_data['baby_ID'] = data['baby_ID']
    tfd_data['channel'] = data['channel']
    tfd_data['Fs'] = Fs
    tfd_data['eeg_epochs'] = x_epoch
    tfd_data['anno'] = x_epoch_anno
    tfd_data['tfds_epoch'] = tfds

    return tfd_data



def prepare_EEG_data(data, annotations=None, ch_to_use='F4_C4', Fs=None, baby_id=None):
    """
    This function would be used to prepare the EEG data prior to edf generation.

    The demo data included in this example is an edf file that is already sampled at 64 Hz
    and is already in the bi-polar format.

    For use with own EEG data, please ensure the data is filtered and downsampled to 64 Hz and
    the bipolar montage ( F4–C4, C4–O2, F3–C3, C3–O1, T4–C4, C4–Cz, Cz–C3 and C3–T3) is applied.

    The paper evaluates only single channels therefore the output is a single channel


    Syntax: data_dict = prepare_EEG_data(data, annotations=None, ch_to_use='F4_C4', Fs=64, baby_id=None)

    Inputs:
        data           - EEG data in DataFrame format
        annotations    - EEG annotations in numpy array format (default = None)
        ch_to_use      - This is the channel that will be used (default = F4_C4)
        Fs             - This is the smapling frequency (default = 64)
        baby_id        - This is the baby id/ file id

    Outputs:
        data_dict      - Dictionary containing eeg data, annotations, samling frequency, channel extracted and baby id

    Example:

    """


    # If there are no annotations just make annotation array of all ones
    if not annotations:
        annotations = np.ones(data.shape[0])

    if Fs != 64:
        raise ValueError('Model was generated with data that had a sampling frequency of 64 Hz')

    data_dict = {'eeg_data': data[ch_to_use].to_numpy(), 'anno': annotations, 'Fs': Fs, 'channel': ch_to_use,
                 'baby_ID': baby_id[:-4]}

    return data_dict



def tfd_data_prep(data):
    """
    This is just some steps to get the data ready to be classified by the fully trained model

    Syntax: x, y, IDs = tfd_data_prep(data)

    Inputs:
        data    - Dictionary containing information about the file and channel along with TFD annotations and
                  TFD values

    Outputs:
        x       - Numpy array containing just the TFD values
        y       - Numpy array containing annotations
        IDs     - The baby/ file id

    Example:

    """

    # Convert to annos to binary keeping only bursts (1)
    data.loc[data['anno'] != 1, 'anno'] = 0

    # Split DataFrame into features abd labels
    x, y = np.array(data.iloc[:, 4:]), np.array(data.iloc[:, 3])
    IDs = data['baby_ID'].to_numpy()
    return x, y, IDs




def main(eeg_data=None, Fs=None, file_name=None, channel='F4_C4', path_to_TFD_package=None):
    """
    This is the main part of the file that is used to call all the sub-functions in the correct order.

    Syntax: eeg_data, tfd_data_df, y_preds_probs = main(eeg_data=None, Fs=None, file_name=None, path_to_TFD_package=None)

    Inputs:
        eeg_data                - Pandas DataFrame containing EEG data
        Fs                      - Sampling frequency of eeg_data
        channel                 - Bi-polar channel to compute TFD from (default = 'F4_C4')
        file_name               - This is the name of the file, it is also used as the baby id
        path_to_TFD_package     - This is the path to the MATLAB package used to generate the
                                  time-frequency distributions (TFDs)

    Outputs:
        eeg_data                - The eeg data
        tfd_data_df             - The TFD DataFrame
        y_preds_probs           - The burst class probability predictions

    Example:
            path_to_TFD_package will need to be defined by the user!!!!!!!

            Example 1 - using the demo_data
            eeg_ts_data, tfd_data_df, y_preds_probs = main(path_to_TFD_package='path_to_memeff_TFDs_package')

            Example 2 - loading a csv file (assuming it is bi-polar order, has been filtered and downsampled to 64 Hz)
            eeg_data = pd.read_csv('demo_data.csv')
            eeg_df, tfd_data_df, y_preds_probs = main(eeg_data=eeg_data, Fs=64, file_name='File_1', channel='F3–C3',
                                                      path_to_TFD_package='path_to_memeff_TFDs_package')



    """
    ###################################################################################
    #                           Initial checks
    ###################################################################################

    if not path_to_TFD_package:
        print('Need to add path to the MATLAB memeff_TFDs package')
        print('Available: https://github.com/otoolej/memeff_TFDs')
        raise ValueError('This program will not work without the MATLAB memeff_TFDs package')

    # If no eeg_data (DataFrame) is passed in it will load the demo data
    if not eeg_data:
        # Load sample edf file - created with https://github.com/BrianMur92/NEURAL_py_EEG_feature_set
        if not file_name:
            file_name = 'demo_data.edf'
        f = mne.io.read_raw_edf(file_name, preload=True)
        eeg_data = pd.DataFrame(data=np.transpose(f._data) * 10 ** 6, columns=f.ch_names)  # This the DataFrame EEG data
        Fs = int(f.info['sfreq'])

    ###################################################################################
    # Generate time-frequency distribution using https://github.com/otoolej/memeff_TFDs
    ###################################################################################

    tfd_params = {'epoch_seconds': 32, 'tfd_pad_seconds': 2, 'do_diff': 1}

    ta_data = prepare_EEG_data(eeg_data, ch_to_use=channel, Fs=Fs, baby_id=file_name)
    tfd_data = gen_TFD_EEG_epochs(d_win_len=31, l_win_len=171, Nfreq_in=256, data=ta_data,
                                  params=tfd_params, path_to_TFD_package=path_to_TFD_package)
    tfd_data_flat = prepare_tfd_data(tfd_data, params=tfd_params)
    tfd_data_df = make_tfd_dataframe(tfd_data_flat)



    # Split the tfd into feaures, annotations and patient IDs fo use with the model
    x, y, IDs = tfd_data_prep(tfd_data_df)

    ###################################################################################
    #                           Load the pre-trained model
    ###################################################################################
    model_name = 'tfd_feature_set_dwin_35_lwin_171.model'
    random_seed = 42
    params = {'n_jobs': 4, 'seed': random_seed, 'subsample': 0.8, 'colsample_bytree': 0.85, 'n_estimators': 35}
    trained_model = xgb.XGBClassifier(**params)
    trained_model.load_model(model_name)

    ###################################################################################
    #                           Classify TFD data
    ###################################################################################

    y_preds_probs = trained_model.predict_proba(x)[:, 1]  # Probability of epochs being bursts

    return eeg_data, tfd_data_df, y_preds_probs

