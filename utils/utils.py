from statsmodels.tsa.stattools import adfuller, kpss
from scipy.integrate import quad, IntegrationWarning
import numpy as np
from torch.utils.data import Dataset
import os
import h5py
import pickle
import math

root = r'/zwork/garrett/DT_CDA'
dataDir = os.path.join(root, 'EEG')
modelDir = os.path.join(root, 'EEG_Forecast/Models')
configDir = os.path.join(root, 'EEG_Forecast/Model_Training_Configs')

def load_experiment_data(sjNum):
    
    # combine data across conditions to maximize training data size
    cons = [0, 1]

    con_data = []
    for con in cons:
    
        data_file_wrapper = h5py.File(os.path.join(dataDir, f'sj{sjNum:02d}_cond{con:02d}_EEG.mat'))
    
        EEG = data_file_wrapper['EEG']
    
        # trials x time x chans
        data = np.asarray(EEG['data'])

        con_data.append(data)
        
    data = np.concatenate(con_data, axis=0)
    
    times = EEG['times']
    Fs = EEG['srate'][0].item()

    nTrials, nTimepoints, nChans = data.shape

    # load training configuration file
    with open(os.path.join(configDir, f'sj{sjNum}_config.pickle'), 'rb') as f:
        training_config = pickle.load(f)
    
    # split data into a training / validation / test set for forecasting generalization
    shuff_idx = training_config['shuff_idx']
    test_trials_idx = training_config['test_idx']
    train_val_trials_idx = training_config['train_val_trials_idx']
    val_size = training_config['val_size']
    con_labels = training_config['con_labels']

    X = data[shuff_idx, :]
    con_labels=con_labels[shuff_idx]

    return {'data': X, 'con_labels': con_labels, 'times': times, 'test_trials_idx': test_trials_idx, 'nTrials': nTrials, 'nTimepoints': nTimepoints, 'nChans': nChans, 'Fs': Fs}


def extract_ML_comparison_trials(sjNum, model, horizon):
    # load deep learning training config to get test comparison trials
    with open(f'/zwork/garrett/DT_CDA/EEG_Forecast/Model_Training_Configs/sj{sjNum}_config.pickle', 'rb') as f:
        DL_training_config = pickle.load(f)

    shuff_idx = DL_training_config['shuff_idx']
    test_trials_idx = DL_training_config['test_idx']
    con_labels = DL_training_config['con_labels']

    # Load ML model predictions
    all_con_residuals = []
            
    for iCon in range(2):
        sj_file_name = f'sj{sjNum:02d}_con{iCon:02d}_{model}_horizon_{horizon}_fit_stim_delay.pickle'
        with open(os.path.join(modelDir, model, sj_file_name), 'rb') as f:
            fit_results = pickle.load(f)
        ground_truth = fit_results['ground_truth']
        predictions = fit_results['X_hat']
        nTrials = len(predictions)
        con_residuals = [(ground_truth[i, -horizon:] - predictions[i][-horizon:])[np.newaxis,:] for i in range(nTrials)]
        con_residuals = np.concatenate(con_residuals, axis=0)
        all_con_residuals.append(con_residuals)
        
    all_con_residuals = np.vstack(all_con_residuals)

    if all_con_residuals.shape[-1] != horizon:
        all_con_residuals = all_con_residuals.swapaxes(1,-1)

    shuff_residuals = all_con_residuals[shuff_idx]
    con_labels = con_labels[shuff_idx]
    
    test_residuals = shuff_residuals[test_trials_idx]
    test_con_labels = con_labels[test_trials_idx]

    return test_residuals, test_con_labels



class OptimalSVHT:
    """
    A class to compute the optimal singular value hard thresholding (SVHT) coefficient
    for matrix denoising using singular values hard thresholding.
    """

    def __init__(self, sigma_known):
        """
        Initialize the class with matrix aspect ratio and noise level knowledge.

        Parameters:
            sigma_known (bool): True if noise level is known, False if unknown.
        """
        self.sigma_known = sigma_known

    def compute_optimal_SVHT_coef(self, beta):
        """
        Computes the optimal singular value hard thresholding coefficient.
        Parameters:
            beta (float or np.ndarray): Aspect ratio m/n of the matrix to be denoised, 0 < beta <= 1.
        Returns:
            np.ndarray: Optimal location of hard threshold.
        """
        if self.sigma_known:
            coef = self.optimal_SVHT_coef_sigma_known(beta)
        else:
            coef = self.optimal_SVHT_coef_sigma_unknown(beta)
        self.coef = coef
        return coef

    def optimal_SVHT_coef_sigma_known(self, beta):
        """
        Computes the optimal singular value hard thresholding coefficient 
        when noise level is known.

        Returns:
            np.ndarray: Threshold coefficient.
        """
        assert np.all((beta > 0) & (beta <= 1)), "beta must be in the range (0,1]"
        w = (8 * beta) / (beta + 1 + np.sqrt(beta**2 + 14 * beta + 1))
        return np.sqrt(2 * (beta + 1) + w)

    def optimal_SVHT_coef_sigma_unknown(self, beta):
        """
        Computes the optimal singular value hard thresholding coefficient 
        when noise level is unknown.

        Returns:
            np.ndarray: Threshold coefficient.
        """
        coef = self.optimal_SVHT_coef_sigma_known(beta)
        MPmedian = np.array([self.MedianMarcenkoPastur(b) for b in np.atleast_1d(beta)])
        return coef / np.sqrt(MPmedian)

    def MarcenkoPasturIntegral(self, x, beta):
        """
        Computes the integral of the Marčenko-Pastur distribution up to x.

        Parameters:
            x (float): Upper limit of integration.
            beta (float): Aspect ratio m/n.

        Returns:
            float: Integral value.
        """
        if not (0 < beta <= 1):
            raise ValueError("beta must be in the range (0,1]")

        lobnd = (1 - np.sqrt(beta))**2
        hibnd = (1 + np.sqrt(beta))**2
        if not (lobnd <= x <= hibnd):
            raise ValueError("x must be within the support of the Marčenko-Pastur distribution")

        def density(t):
            return np.sqrt((hibnd - t) * (t - lobnd)) / (2 * np.pi * beta * t)

        integral, _ = quad(density, lobnd, x)
        return integral

    def MedianMarcenkoPastur(self, beta):
        """
        Computes the median of the Marčenko-Pastur distribution.

        Parameters:
            beta (float): Aspect ratio m/n.

        Returns:
            float: Median of the distribution.
        """
        def MarPas(x):
            return 1 - self.incMarPas(x, beta, 0)

        lobnd = (1 - np.sqrt(beta))**2
        hibnd = (1 + np.sqrt(beta))**2

        while hibnd - lobnd > 0.001:
            x_vals = np.linspace(lobnd, hibnd, 5)
            y_vals = np.array([MarPas(x) for x in x_vals])

            if np.any(y_vals < 0.5):
                lobnd = np.max(x_vals[y_vals < 0.5])
            if np.any(y_vals > 0.5):
                hibnd = np.min(x_vals[y_vals > 0.5])

        return (hibnd + lobnd) / 2

    def incMarPas(self, x0, beta, gamma):
        """
        Computes the incomplete Marčenko-Pastur integral.

        Parameters:
            x0 (float): Lower bound of integration.
            beta (float): Aspect ratio m/n.
            gamma (float): Exponent applied to x in integration.

        Returns:
            float: Integral value.
        """
        if beta > 1:
            raise ValueError("beta must be in the range (0,1]")

        topSpec = (1 + np.sqrt(beta))**2
        botSpec = (1 - np.sqrt(beta))**2

        def MarPas(x):
            condition = (topSpec - x) * (x - botSpec) > 0
            return np.where(condition, np.sqrt((topSpec - x) * (x - botSpec)) / (beta * x * 2 * np.pi), 0)

        if gamma != 0:
            fun = lambda x: x**gamma * MarPas(x)
        else:
            fun = lambda x: MarPas(x)

        integral, _ = quad(fun, x0, topSpec)
        return integral


def multivariate_stationarity_check(X, run_adf=True, run_kpss=False):
    nChans = X.shape[1]
    metric_dim = 0
    if run_adf:
        metric_dim += 2
    if run_kpss:
        metric_dim += 2
    stationarity_checks = np.zeros((nChans, metric_dim))
    verbose = False
    threshold = 0.05
    for iChan in range(nChans):
        if run_adf:
            metric_idx = range(2)
            # Null: The series has unit root (non-stationary)
            # Alternative: The series has no unit root (stationary)
            adf_test = adfuller(X[:, iChan], autolag='AIC')
            
            adf_p = adf_test[1] # fail to reject null -> time series is non-stationary
            stationarity_checks[iChan, metric_idx] = [adf_p, adf_p >= threshold]

        if run_kpss:
            metric_idx = range(2,4) if run_adf else range(2)
            # Null: The process is trend stationary
            # Alternative: The series has unit root (non-stationary)
            kpss_test = kpss(x[:, iChan], regression='c', nlags='auto')
            kpss_p = kpss_test[1] # fail to reject null -> time series is TREND stationary
            stationarity_checks[iChan, metric_idx] = [kpss_p, kpss_p < threshold]
        
        if verbose:
            if run_adf:
                print(f'Variable {iChan}: ADF Test Statistic {adf_test[0]:0.3f} | p-value {adf_test[1]:0.3f} \
                    | #Lags Used {adf_test[2]}')

            if run_kpss:
                print(f'Variable {iChan}: KPSS Test Statistic {kpss_test[0]:0.3f} | p-value {kpss_test[1]:0.3f} \
                    | #Lags Used {kpss_test[2]}')
    
    metric_idx = [1,3] if run_adf and run_kpss else 1 
    overall_nonstationarity_check = stationarity_checks[:, metric_idx]
    if overall_nonstationarity_check.ndim == 1:
        overall_nonstationarity_check = overall_nonstationarity_check[:, np.newaxis]
    return any(overall_nonstationarity_check.sum(axis=1)), stationarity_checks


def nth_order_difference(X, n=1, axis=0):
    for _ in range(n):
        X = np.diff(X, axis=axis)
    return X

def invert_nth_order_difference(X_diff, initial_values, n=1, axis=0):
    """
    Invert the n-th order difference of a 2D matrix along the specified axis.

    Parameters:
        X_diff (np.ndarray): The n-th order differenced matrix.
        initial_values (np.ndarray): The first n elements of the original sequence.
        n (int): Order of the difference.
        axis (int): Axis along which to reconstruct the sequence (0 for rows, 1 for columns).
    
    Returns:
        np.ndarray: The reconstructed matrix.
    """
    X = initial_values.copy()
    for diff_order in range(n, 0, -1):
        shape = list(X_diff.shape)
        shape[axis] = shape[axis]+1
        new_X = np.zeros(shape, dtype=X_diff.dtype)

        if axis==0:
            new_X[:initial_values.shape[0], :] = initial_values
            for i in range(initial_values.shape[0], shape[0]):
                new_X[i, :] = new_X[i-1, :] + X_diff[i-1, :]
        else:
            new_X[:, :initial_values.shape[1]] = initial_values
            for j in range(initial_values.shape[1], shape[1]):
                new_X[:, j] = new_X[:, j-1] + X_diff[:, j-1]

        X = new_X  # Update for the next order

    return X

class customDataSet(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        X = self.data[idx, :]
        return X

def normalize_data(data, data_min, data_max):
    x = (data - data_min) / (data_max - data_min)
    return x


def sliding_window_extraction(data: np.ndarray, times: np.ndarray,
                               window_size: int, step_size: int = 1) -> np.ndarray:
    """
    Splits a NumPy array of shape (N, T, C) into overlapping windows of size window_size (L) along time axis.

    Args:
        data (np.ndarray): Input array of shape (N, T, C)
        window_size (int): Size of the window (L)
        step_size (int): Stride between windows

    Returns:
        np.ndarray: Output array of shape (N, num_windows, L, C)
    """
    N, T, C = data.shape
    L = window_size

    times = (times-times.min()) / (times.max() - times.min())

    if L > T:
        raise ValueError("window_size cannot be greater than T")
    if times.shape[0] != T:
        raise ValueError("Length of times must match T dimension of data")

    # Use sliding_window_view to extract windows
    data_windows = np.lib.stride_tricks.sliding_window_view(data, (L, C), axis=(1, 2)).squeeze(2)
    time_windows = np.lib.stride_tricks.sliding_window_view(times, L)

    # Apply step size
    data_windows = data_windows[:, ::step_size, :, :]  # shape: (N, num_windows, L, C)
    time_windows = time_windows[::step_size, :]        # shape: (num_windows, L)

    # Flatten over trials
    N_windows = data_windows.shape[1]
    windows = data_windows.reshape(N * N_windows, L, C)
    window_times = np.tile(time_windows, (N, 1))  # repeat time_windows for each trial

    return windows, window_times

def create_cmw_time(n_cycles, freq, t_start, srate):

    # higher n_cycles -> less temporal resolution, higher freq resolution
    # lower n_cycles -> high temporal resolution, low freq resolution
    
    fwhm_sec = n_cycles * np.sqrt(2*np.log(2))/(math.pi*freq)
    t = np.arange(-t_start, t_start+(1/srate), 1/srate)

    sinewave = np.exp(1j*2*math.pi*freq*t)
    gaus = np.exp(-4*np.log(2) * (t**2) / (fwhm_sec**2))
    cmw = sinewave * gaus

    #normalize to unit energy
    cmw /= np.sqrt(np.sum(np.abs(cmw)**2))
    return cmw, t

def create_cmw_freq(peak_freq, fwhm_hz, Fs, seq_length):
    freqs = np.linspace(0, Fs, seq_length) # frequencies from 0 - sampling rate at length of spectrum (e.g., length of signal/convolution)
    x = freqs - peak_freq
    s = (fwhm_hz*(2*math.pi - 1) )/(4*math.pi)
    gaussian = np.exp(-0.5*(x/s)**2)

    # normalize to unit energy (Parseval’s theorem)
    energy_freq = np.sum(np.abs(gaussian)**2) / seq_length
    gaussian_norm = gaussian / np.sqrt(energy_freq)

    return g, freqs