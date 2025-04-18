import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import convolve
import glob
from scipy.optimize import curve_fit
from scipy.fftpack import fft, ifft, fftfreq
from scipy.special import voigt_profile
import ast

def convert_linear_2_dB(y_data):
    return 10* np.log10(y_data)
def convert_dB_2_linear(y_data):
    return 10**(y_data/10)

def low_pass_band_filter_fft(cut_off_freq, x_data, y_data, plot_check=True): 
    unfilt_signal = y_data
    fft_signal = fft(unfilt_signal)
    fft_frequencies = fftfreq(len(x_data), d=(x_data[1] - x_data[0]))


    low_pass_mask = np.abs(fft_frequencies)<= cut_off_freq

    filtered_fft_signal = fft_signal * low_pass_mask

    filtered_signal = np.real(ifft(filtered_fft_signal))

    if plot_check ==True: 
        positive_freqs = fft_frequencies>0
        fig  = plt.figure(figsize=(6,4))
        plt.plot(fft_frequencies[positive_freqs], np.abs(fft_signal[positive_freqs]))
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude")
        plt.xlim([0,1000])
        plt.ylim([0,5000])
        plt.grid()
        plt.show()
        plt.tight_layout()
    return filtered_signal

def read_fridge_VNA_data(file_path): # MHZ
    # Read the data after the "BEGIN" line into a DataFrame
    data = pd.read_csv(file_path, skiprows=1, encoding='latin1', header=None)
    # data.columns.values[0] = 'Frequency (GHz)'  # Change 'Column_1' to your desired name
    # data.columns.values[1] = 'S21 (dB)'
    # data.columns.values[2] = 'Date'  # Change 'Column_1' to your desired name
    # data.columns.values[3] = 'Time'
    data.columns = ['Frequency (GHz)', 'S21 (dB)', 'Date', "Time"]
    data["Frequency (GHz)"] = pd.to_numeric(data["Frequency (GHz)"], errors='coerce')
    data["Frequency (GHz)"] = data["Frequency (GHz)"]/1000
    data['S21 (dB)'] = pd.to_numeric(data['S21 (dB)'], errors='coerce') 
    return data

def read_VNA_data(file_path): # 19-12 data wont work with this function - transition to this. 
    # Reads VNA data and Magnetic field data from PSU's and
    data = pd.read_csv(file_path, skiprows=1, encoding='latin1', header=None)
    data.columns = ['Frequency (GHz)', 'S21 (dB)', 'Date', "Time", "Magnetic Field (T)"]
    data["Frequency (GHz)"] = pd.to_numeric(data["Frequency (GHz)"], errors='coerce')
    data["Frequency (GHz)"] = data["Frequency (GHz)"]
    data['S21 (dB)'] = pd.to_numeric(data['S21 (dB)'], errors='coerce') 
    data["Magnetic Field (T)"] = pd.to_numeric(data["Magnetic Field (T)"], errors='coerce')
    return data

def read_VNA_data_magnets(file_path):
    # Read the file without header to determine the number of columns
    raw_data = pd.read_csv(file_path, skiprows=1, encoding='latin1', header=None)
    
    # Define default column names (extend as needed)
    default_columns = ['Frequency (GHz)', 'S21 (dB)', 'Date', 'Time', 'Magnetic Field (T)']
    
    # If there are more columns than defaults, use original headers for the extras
    num_columns = raw_data.shape[1]
    
    if num_columns > len(default_columns):
        extra_columns = [f'Extra Column {i+1}' for i in range(len(default_columns), num_columns)]
        column_names = default_columns + extra_columns
    else:
        column_names = default_columns[:num_columns]
    
    # Assign column names to DataFrame
    raw_data.columns = column_names
    
    # Convert numeric columns where applicable
    for col in column_names:
        raw_data[col] = pd.to_numeric(raw_data[col], errors='coerce')
    
    return raw_data

def read_VNA_data_magnets_revised(file_path):
    # Read CSV and ensure State column is treated as a single field
    raw_data = pd.read_csv(file_path, quotechar='"')  
    
    # Convert the 'State' column from string to actual Python list
    raw_data["State"] = raw_data["State"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x)
    
    # Extract individual values from the 'State' list
    raw_data["State Device"] = raw_data["State"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
    raw_data["Current Limit"] = raw_data["State"].apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else None)
    raw_data["Current Ramp Limit"] = raw_data["State"].apply(lambda x: x[2] if isinstance(x, list) and len(x) > 2 else None)
    raw_data["Coil Constant"] = raw_data["State"].apply(lambda x: x[3] if isinstance(x, list) and len(x) > 3 else None)
    raw_data["Current"] = raw_data["State"].apply(lambda x: x[4] if isinstance(x, list) and len(x) > 4 else None)
    raw_data["Calculated Supply Current"] = raw_data["State"].apply(lambda x: x[5] if isinstance(x, list) and len(x) > 5 else None)
    raw_data["Supply Voltage"] = raw_data["State"].apply(lambda x: x[6] if isinstance(x, list) and len(x) > 6 else None)
    raw_data["Calculated System Resistance"] = raw_data["State"].apply(lambda x: x[7] if isinstance(x, list) and len(x) > 7 else None)
    raw_data["Calculated Field"] = raw_data["State"].apply(lambda x: x[8] if isinstance(x, list) and len(x) > 8 else None)
    
    # Drop the original 'State' column (optional)
    raw_data.drop(columns=["State"], inplace=True)

    return raw_data

def calculate_S21_features(input_data_dB, freq_range, **kwargs):
    col_name_amplitude = kwargs.get("col_name_amplitude", "S21 (dB)")
    col_name_freq = kwargs.get("col_name_amplitude", "Frequency (GHz)")
    output_filename=kwargs.get("output_filename", "S21_features")
    print_bool=kwargs.get("to_print", "True")
    data = input_data_dB
    
    try:
        # Filter data within the specified frequency range
        data_in_range = data[(data[col_name_freq] >= freq_range[0]) & (data[col_name_freq] <= freq_range[1])]
        
        if data_in_range.empty:
            raise ValueError("No data in specified frequency range.")

        # Find the resonant frequency (f_r) as the frequency at max dB(S(2,1)) within the range
        f_r_index = data_in_range[col_name_amplitude].idxmax()
        f_r = data_in_range.loc[f_r_index, col_name_freq]
        S_21_mag_at_f_r = data_in_range.loc[f_r_index, col_name_amplitude]
        
        
        S_21_mag_3dB = S_21_mag_at_f_r - 3

        # Find the data point below the peak (f1) where S21 is closest to the 3 dB cutoff
        data_below_peak = data_in_range[data_in_range[col_name_freq] < f_r]
        f1 = data_below_peak.iloc[(data_below_peak[col_name_amplitude] - S_21_mag_3dB).abs().argsort()].iloc[0][col_name_freq]

        # Find the data point above the peak (f2) where S21 is closest to the 3 dB cutoff
        data_above_peak = data_in_range[data_in_range[col_name_freq] > f_r]
        f2 = data_above_peak.iloc[(data_above_peak[col_name_amplitude] - S_21_mag_3dB).abs().argsort()].iloc[0][col_name_freq]

        # Handle cases where f1 or f2 are not found
        if pd.isna(f1) or pd.isna(f2):
            raise ValueError("Unable to find both 3 dB points (f1 and/or f2).")

        # Retrieve S21 magnitude values at f1 and f2
        S21_at_f1 = data_in_range[data_in_range[col_name_freq] == f1][col_name_amplitude].values[0]
        S21_at_f2 = data_in_range[data_in_range[col_name_freq] == f2][col_name_amplitude].values[0]
        
        # Calculate the frequency difference (delta_f)
        delta_f = abs(f2 - f1)
        
        # Calculate loaded and unloaded Q-factors
        Q_loaded = f_r / delta_f
        Q_unloaded = abs(Q_loaded / (1 - np.sqrt(10 ** (-S_21_mag_at_f_r / 10))))

        # Calculate uncertainty based on percent variation from S_21_mag_3dB
        percent_variation_f1 = abs((S21_at_f1 - S_21_mag_3dB) / S_21_mag_3dB) * 100
        percent_variation_f2 = abs((S21_at_f2 - S_21_mag_3dB) / S_21_mag_3dB) * 100
        uncertainty = (percent_variation_f1 + percent_variation_f2) / 2
        Q_loaded_uncertainty = uncertainty
        Q_unloaded_uncertainty = uncertainty / abs(1 - np.sqrt(10 ** (-S_21_mag_at_f_r / 10)))
        
        if print_bool: 
            print("S_21 at resonant frequency (f_r) =", S_21_mag_at_f_r)
            print("Resonant Frequency (f_r) =", f_r)
            print("Q loaded =", Q_loaded)
            print("Q unloaded =", Q_unloaded)
            print("Delta F =", delta_f)
            print("f1 =",f1)
            print("f2 =",f2)
            print("S21 at f1", S21_at_f1)
            print("S21 at f2", S21_at_f2)
            print("Q Loaded Uncertainty", Q_loaded_uncertainty, "%")
            print("Q Unloaded Uncertainty", Q_unloaded_uncertainty, "%")
        
        # Write results to CSV
        results = {
            'f_r': [f_r],
            'S_21_mag_at_f_r': [S_21_mag_at_f_r],
            'f1': [f1],
            'S21_at_f1': [S21_at_f1],
            'f2': [f2],
            'S21_at_f2': [S21_at_f2],
            'delta_f': [delta_f],
            'Q_loaded': [Q_loaded],
            'Q_unloaded': [Q_unloaded],
            'Q_loaded_uncertainty (%)': [Q_loaded_uncertainty],
            'Q_unloaded_uncertainty (%)': [Q_unloaded_uncertainty]
        }

    except (ValueError, IndexError, KeyError) as e:
        print(f"Error: {str(e)}. Writing null data to CSV.")
        results = {
            'f_r': [None],
            'S_21_mag_at_f_r': [None],
            'f1': [None],
            'S21_at_f1': [None],
            'f2': [None],
            'S21_at_f2': [None],
            'delta_f': [None],
            'Q_loaded': [None],
            'Q_unloaded': [None],
            'Q_loaded_uncertainty (%)': [None],
            'Q_unloaded_uncertainty (%)': [None]
        }

    # Convert results to DataFrame and write to CSV
    df_results = pd.DataFrame(results)
    #df_results.to_csv(output_filename, index=False)
    #print("Results written to", output_filename)

    return results

def read_bluefores_logs(channel):
    pass

#### Functions ####
def s21_lorentzian_adjusted(x_p, Q_tot, Q_ext, alpha, x_0, S21_baseline):
    x = (2 * Q_tot * (x_p - x_0)) / x_0
    A = Q_tot / Q_ext
    B = -2 * Q_tot * alpha / x_0
    
    # Lorentzian form
    real_part = (A + B * x) / (1 + x**2)
    
    # Final function
    return np.abs(1-real_part + S21_baseline)

def s21_lorentzian_original(x_p, Q_tot, Q_ext, alpha, x_0, S21_baseline):
    """
    Computes |S21(ω_probe)| using the original lorentzian_function.
    
    Parameters:
    - x_p: Frequency array
    - Q_tot: Total quality factor
    - Q_ext: External quality factor
    - alpha: Loss parameter
    - x_0: Resonance frequency
    - S21_baseline: Baseline offset
    
    Returns:
    - Theoretical |S21| values.
    """
    A = Q_tot / Q_ext
    B = 2 * Q_tot * alpha / x_0
    x = (2 * Q_tot * (x_p - x_0)) / x_0
    
    # Complex denominator
    denominator = 1 + 1j * x
    
    # Compute the full S21 expression
    S21 = 1 - (A - 1j * B) / denominator + S21_baseline
    
    # Return magnitude
    return np.abs(np.real(S21))

# def lorentzian(x, a,x_0, b, c): 
#     #### Define Lorentzuab function ####
#     # arguments: 
#     # a = peak amplitude
#     # x_0 = center of peak
#     # b = HWHM 
#     # c = baseline offset
#     return a * b**2 / ((x - x_0)**2 + b**2) + c

def lorentzian_smoothing(y_data, x_kernal=np.linspace(-1, 1, 101), kernal_width= 0.1): 
    # Define Lorentzian kernel
    kernel_width = 0.1  # Adjust based on your resolution needs
    x_kernel = np.linspace(-1, 1, 101)  # Kernel range (adjust size as needed)
    lorentzian= kernel_width / (np.pi * (x_kernel**2 + kernel_width**2))

    # Normalize kernel (important for smoothing)
    lorentzian /= np.sum(lorentzian)

    # Convolve data with Lorentzian kernel
    smoothed_y = convolve(y_data, lorentzian, mode='same')
    return smoothed_y


def calculate_r_squared(y_data, y_pred):
    """
    Calculate the coefficient of determination (R^2).

    Parameters:
    - y_data (array-like): Observed data.
    - y_pred (array-like): Predicted data.

    Returns:
    - r2 (float): Coefficient of determination (R^2).
    """
    # Convert to numpy arrays for computation
    y_data = np.array(y_data)
    y_pred = np.array(y_pred)
    
    # Calculate the mean of observed values
    y_mean = np.mean(y_data)
    
    # Calculate SS_res (Residual Sum of Squares)
    ss_res = np.sum((y_data - y_pred) ** 2)
    
    # Calculate SS_tot (Total Sum of Squares)
    ss_tot = np.sum((y_data - y_mean) ** 2)
    
    # Calculate R^2
    r2 = 1 - (ss_res / ss_tot)
    
    return r2
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

def lorentzian(f, f0, gamma, A, B):
    """Lorentzian function for fitting."""
    return A / ((f - f0)**2 + (gamma / 2)**2) + B


def extract_s21_features(x_data, y_data, to_print=False, db_2_linear=True):
    ## convert to linear space
    if db_2_linear: 
        y_data = convert_dB_2_linear(y_data)
    """
    Extracts the key features of an S21 resonance curve.

    Parameters:
        x_data (numpy array): Frequency values.
        y_data (numpy array): Magnitude of S21 (linear scale, not dB).

    Returns:
        dict: Extracted features including:
            - 'Q_tot': 3dB quality factor
            - 'f_res': Center frequency of the resonance
            - 'FWHM': Full width at half maximum
            - 'baseline': Estimated baseline level
    """
    # Find peak in |S21|
    peak_idx, _ = find_peaks(y_data, height=np.max(y_data) * 0.8)
    
    if len(peak_idx) == 0:
        raise ValueError("No significant peak found in S21 data.")

    peak_idx = peak_idx[np.argmax(y_data.iloc[peak_idx])]  # Select the highest peak
    f_res = x_data.iloc[peak_idx]

    # Fit Lorentzian to get better estimate of resonance parameters
    p0 = [f_res, np.ptp(x_data) / 10, np.max(y_data), np.min(y_data)]
    try:
        popt, _ = curve_fit(lorentzian, x_data, y_data, p0=p0)
        f_res, gamma, A, baseline = popt
    except RuntimeError:
        gamma = np.ptp(x_data) / 10  # Fallback if fit fails
        baseline = np.min(y_data)

    # Find FWHM
    peak_value = np.max(y_data)
    half_max = (peak_value + baseline) / 2
    above_half_max = np.where(y_data >= half_max)[0]

    if len(above_half_max) < 2:
        raise ValueError("Could not determine FWHM.")

    fwhm = x_data.iloc[above_half_max[-1]] - x_data.iloc[above_half_max[0]]

    # Compute 3dB Q factor
    Q_tot = f_res / fwhm if fwhm > 0 else np.inf

    s21_peak = y_data.iloc[peak_idx]
    
    s21_peak_dB = convert_linear_2_dB(s21_peak)
    baseline_dB = convert_linear_2_dB(baseline)
   
    if to_print: 
        print("\n")
        print("############S21 Features##############")
        print(f"Q Total: {Q_tot:.9f}")
        print(f"Estimated Resonant Frequency (GHz): {f_res:.9f}")
        print(f"FWHM (GHz): {fwhm:.9f}")
        print(f"Baseline (linear): {baseline:.9f}")
        print(f"S21 Peak (linear): {s21_peak:.9f}")
        print(f"Baseline (dB): {baseline_dB:.9f}")
        print(f"S21 Peak (dB): {s21_peak_dB:.9f}")
        print("######################################")
        print("\n")

    return {
        "Q_tot": Q_tot,
        "f_res": f_res,
        "FWHM": fwhm,
        "baseline": baseline,
        "S21_peak": s21_peak}

def extract_quality_factors(lorentzian_function, x_data, y_data, to_print=False):
    
    """
    Extracts quality factors from S21 data by fitting a Lorentzian function.
    
    Parameters:
        x_data (array-like): Frequency data in GHz.
        y_data (array-like): S21 data in dB.
        
    Returns:
        tuple: (parameters, fitted_y) where parameters contain the extracted quality factors
               and fitted_y contains the fitted S21 values in dB.
    """

    s21_features = extract_s21_features(x_data, y_data, to_print=False)

    # Convert dB to linear scale
    y_data = convert_dB_2_linear(y_data)
    
    # Initial guess for the Lorentzian fit

    Q_tot = s21_features["Q_tot"]
    Q_ext = 1000000
    alpha = 0.01
    x_0 =  s21_features["f_res"]
    S21_baseline = 0
    initial_guess = [Q_tot, Q_ext, alpha, x_0, S21_baseline]
    
    # Fit the Lorentzian function to the data
    params, _ = curve_fit(lorentzian_function, x_data, y_data, p0=initial_guess)
    
    # Extract fitted parameters
    Q_tot_fit, Q_ext_fit, alpha_fit, x_0_fit, S21_baseline_fit = params
    Q_int_fit = 1 / (1 / Q_tot_fit - 1 / Q_ext_fit)
    
    # Compute fitted curve
    fitted_y = lorentzian_function(x_data, *params)
    
    # Convert back to dB scale
    fitted_y = convert_linear_2_dB(fitted_y)

    if to_print: 
        print("\n")
        print("############Quality Factors############")
        print(f"Fitted Q_tot: {Q_tot_fit:.9f}")
        print(f"Fitted Q_ext: {Q_ext_fit:.9f}")
        print(f"Estimated Q_int: {Q_int_fit:.9f}")
        print(f"Fitted alpha: {alpha_fit:.9f}")
        print(f"Fitted x_0: {x_0_fit:.9e} GHz")
        print(f"Fitted S21_baseline: {S21_baseline_fit:.9f}")
        print(f"R-Squared: {calculate_r_squared(convert_linear_2_dB(y_data),fitted_y)}")
        print("#######################################")
        print("\n")
        
    
    # Return the parameters and fitted values
    return {
        "Q_tot": Q_tot_fit,
        "Q_ext": Q_ext_fit,
        "Q_int": Q_int_fit,
        "alpha": alpha_fit,
        "x_0": x_0_fit,
        "S21_baseline": S21_baseline_fit
    }, fitted_y


def extract_absorption_features(x_data, y_data, to_print=False, db_2_linear=False):
    """
    Extracts key features from an absorption dip (min in signal).

    Parameters:
        x_data (numpy array): Frequency values.
        y_data (numpy array): Absorption signal (linear scale or dB).
        to_print (bool): Print extracted features.
        db_2_linear (bool): Convert y_data from dB to linear before analysis.

    Returns:
        dict: Features including:
            - 'f_dip': Frequency of minimum absorption
            - 'FWHM': Full width at half minimum
            - 'baseline': Estimated max (i.e., non-absorbed level)
            - 'depth': Absorption dip depth
    """
    
    if db_2_linear:
        y_data = convert_dB_2_linear(y_data)

    # Invert data to use find_peaks on dips
    inverted = -y_data
    dip_idx, _ = find_peaks(inverted, height=np.max(inverted))

    if len(dip_idx) == 0:
        raise ValueError("No significant absorption dip found.")

    # Select the deepest dip
    dip_idx = dip_idx[np.argmin(y_data[dip_idx])]
    f_dip = x_data[dip_idx]

    # Estimate baseline and depth
    baseline = np.max(y_data)
    dip_value = y_data[dip_idx]
    depth = baseline - dip_value

    # Calculate half-minimum level (i.e., halfway from baseline to dip)
    half_min = dip_value + depth / 2

    # Get FWHM: width where signal is below half-minimum
    below_half = np.where(y_data <= half_min)[0]

    if len(below_half) < 2:
        raise ValueError("Could not determine FWHM.")

    fwhm = x_data[below_half[-1]] - x_data[below_half[0]]
    print(x_data[below_half[-1]])
    print(x_data[below_half[0]])

    if to_print:
        print("\n######## Absorption Dip Features ########")
        print(f"Min Frequency (GHz): {f_dip:.6f}")
        print(f"FWHM (GHz): {fwhm:.6f}")
        print(f"Baseline (max signal): {baseline:.6f}")
        print(f"Dip Value (min signal): {dip_value:.6f}")
        print(f"Absorption Depth: {depth:.6f}")
        print("#########################################\n")

    return {
        "f_dip": f_dip,
        "FWHM": fwhm,
        "baseline": baseline,
        "dip_value": dip_value,
        "depth": depth
    }

def extract_all_absorption_dips(freq, signal, min_prominence=0.05, to_print=True):
    """
    Extracts the FWHM and center frequency of all dips in an absorption spectrum.

    Parameters:
        freq (np.array): Frequency axis (e.g., GHz)
        signal (np.array): Absorption signal (lower = more absorbed)
        min_prominence (float): Minimum prominence to consider a valid dip (adjust for noise)

    Returns:
        List of dicts with keys:
            - 'f_dip': Frequency of dip
            - 'FWHM': Full width at half minimum
            - 'depth': Dip depth
    """
    inverted = -signal  # Flip so dips become peaks
    dip_indices, _ = find_peaks(inverted, prominence=min_prominence)

    dip_data = []

    for idx in dip_indices:
        f_dip = freq[idx]
        dip_val = signal[idx]
        baseline = np.max(signal)  # or local baseline if available
        depth = baseline - dip_val
        half_min = dip_val + depth / 2

        # Find FWHM: points around the dip where signal crosses half-min
        # Walk left
        i_left = idx
        while i_left > 0 and signal[i_left] < half_min:
            i_left -= 1

        # Walk right
        i_right = idx
        while i_right < len(signal) - 1 and signal[i_right] < half_min:
            i_right += 1

        # Linear interpolation to improve FWHM estimate
        if i_left == 0 or i_right == len(signal) - 1:
            continue  # dip not well contained

        f_left = np.interp(half_min, [signal[i_left], signal[i_left+1]],
                                     [freq[i_left], freq[i_left+1]])
        f_right = np.interp(half_min, [signal[i_right-1], signal[i_right]],
                                      [freq[i_right-1], freq[i_right]])

        fwhm = f_right - f_left

        dip_data.append({
            "f_dip": f_dip,
            "FWHM": fwhm,
            "depth": depth,
            "index": idx
        })
        
        if to_print:
            print("\n######## Absorption Dip Features ########")
            print(f"Min Frequency (GHz): {f_dip:.6f}")
            print(f"FWHM (GHz): {fwhm:.6f}")
            print(f"Baseline (max signal): {baseline:.6f}")
            print(f"Dip Value (min signal): {dip_val:.6f}")
            print(f"Absorption Depth: {depth:.6f}")
            print("#########################################\n")

    return dip_data

def lorentzian_dip(f, f0, gamma, A, baseline):
    return baseline - A / (1 + ((f - f0)/(gamma/2))**2)

def fit_lorentzian_dip(freq, signal, window_width=0.02, min_prominence=0.01, to_print=True):
    """
    Detects and fits Lorentzian dips in an absorption signal.

    Parameters:
        freq (np.array): Frequency axis (GHz)
        signal (np.array): Absorption data (lower = more absorbed)
        window_width (float): Width of fit window (GHz) around each dip
        min_prominence (float): Minimum prominence to detect dips

    Returns:
        List of dicts: Each dict contains fitted f0, FWHM, depth, baseline, and fit params
    """
    inverted = -signal
    dip_indices, _ = find_peaks(inverted, prominence=min_prominence)
    fits = []

    for idx in dip_indices:
        f0_guess = freq[idx]
        A_guess = np.max(signal) - signal[idx]
        baseline_guess = np.max(signal)
        gamma_guess = window_width / 2

        # Define fit window around dip
        mask = (freq >= f0_guess - window_width/2) & (freq <= f0_guess + window_width/2)
        f_fit = freq[mask]
        s_fit = signal[mask]

        if len(f_fit) < 5:
            continue  # not enough data to fit

        p0 = [f0_guess, gamma_guess, A_guess, baseline_guess]

        try:
            popt, _ = curve_fit(lorentzian_dip, f_fit, s_fit, p0=p0)
            f0_fit, gamma_fit, A_fit, baseline_fit = popt

            fits.append({
                "f_dip": f0_fit,
                "FWHM": gamma_fit,
                "depth": A_fit,
                "baseline": baseline_fit,
                "fit_curve": lorentzian_dip(f_fit, *popt),
                "f_fit": f_fit,
                "s_fit": s_fit
            })

        except RuntimeError:
            continue  # fit failed
        if to_print: 
            for i, f in enumerate(fits):
                if len(fits)> 5: 
                    print("WARNING TOO MANY FITS...IM NOT PRINTING")
                    break
                    #raise ValueError("WARNING TOO MANY FITS")
                print("\n######## Absorption Lorentzian Fit Features ########")
                print(f"Dip {i+1}:")
                print(f"  Center: {f['f_dip']:.6f} GHz")
                print(f"  FWHM:   {f['FWHM']*1e3:.2f} MHz")
                print(f"  Depth:  {f['depth']:.4f}")
                print("#########################################\n")
    return fits

def multi_lorentzian(x, *params):
    baseline = params[-1]
    result = np.zeros_like(x) + baseline
    for i in range(0, len(params) - 1, 3):
        f0, gamma, A = params[i], params[i+1], params[i+2]
        result -= A * gamma**2 / ((x - f0)**2 + gamma**2)
    return result


def fit_multi_lorentzian_dip(freq, signal, window_width=0.8, min_prominence=0.05, max_lorentzians=5, to_print=True):
    inverted = -signal
    dip_indices, _ = find_peaks(inverted, prominence=min_prominence)

    # Limit the number of Lorentzians to fit
    if len(dip_indices) > max_lorentzians:
        dip_indices = dip_indices[:max_lorentzians]

    # Initial guesses
    p0 = []
    for idx in dip_indices:
        f0_guess = freq[idx]
        A_guess = np.max(signal) - signal[idx]
        gamma_guess = window_width / 2
        p0.extend([f0_guess, gamma_guess, A_guess])
    p0.append(np.max(signal))  # baseline guess

    try:
        popt, _ = curve_fit(multi_lorentzian, freq, signal, p0=p0)
        fits = []
        for i in range(0, len(popt) - 1, 3):
            fits.append({
                "f_dip": popt[i],
                "FWHM": popt[i+1],
                "depth": popt[i+2],
            })
        baseline_fit = popt[-1]

        if to_print:
            print("\n######## Multi-Lorentzian Fit Features ########")
            for i, f in enumerate(fits):
                print(f"Dip {i+1}:")
                print(f"  Center: {f['f_dip']:.6f} GHz")
                print(f"  FWHM:   {f['FWHM']*1e3:.2f} MHz")
                print(f"  Depth:  {f['depth']:.4f}")
                print("#########################################\n")

        return {
            "fits": fits,
            "baseline": baseline_fit,
            "fit_curve": multi_lorentzian(freq, *popt)
        }
    except RuntimeError:
        print("Fit failed")
        return None
    
# # Tim's Handy Plotting Helpers
# def cm2inch(*tupl):
#     inch = 2.54
#     if isinstance(tupl[0], tuple):
#         return tuple(i/inch for i in tupl[0])
#     else:
#         return tuple(i/inch for i in tupl)

# # Note - this function currently only works for single-axis plots
# def setfontsize(size, ax):
#     for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
#              ax.get_xticklabels() + ax.get_yticklabels()):
#         item.set_fontsize(size)
#     return

# # Alice's small fig style for thesis - 1/2 the page
# def small_fig_style(fig, ax):
#     setfontsize(8, ax)
#     fig.set_dpi(1200)
#     fig.set_tight_layout(True)
#     ax.set_box_aspect(0.625)
#     fig.set_size_inches(cm2inch((8,6)))
#     return

# # Alice's full-width fig style for thesis - full page width
# def large_fig_style(fig, ax):
#     setfontsize(11, ax)
#     fig.set_dpi(1200)
#     fig.set_tight_layout(True)
#     fig.set_size_inches(cm2inch((16.4,10)))
#     return

# Convert cm to inches
def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)

# Dynamically scale font size based on figure width
def scale_fontsize(fig, base_width_cm=16.4, base_fontsize=11):
    fig_width_cm = fig.get_size_inches()[0] * 2.54
    scale = fig_width_cm / base_width_cm
    return base_fontsize * scale

# Set font size for axis text
def set_ax_fontsize(size, ax):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(size)

# Set font size for colorbar tick labels and axis label
def set_colorbar_fontsize(cbar, size):
    cbar.ax.tick_params(labelsize=size)
    if cbar.ax.yaxis.label:
        cbar.ax.yaxis.label.set_size(size)

def resize_colorbar_vertically(cbar, ax, fraction=0.03, pad=0.02):
    """Resize a vertical colorbar to match the height of the axes."""
    fig = ax.figure
    bbox = ax.get_position()
    cbar_ax = cbar.ax
    new_cbar_position = [
        bbox.x1 + pad,      # x0
        bbox.y0,            # y0
        fraction,           # width
        bbox.height         # height
    ]
    cbar_ax.set_position(new_cbar_position)

def resize_colorbar_horizontally(cbar, ax, fraction=0.03, pad=0.05):
    """Resize a horizontal colorbar to match the width of the axes."""
    fig = ax.figure
    bbox = ax.get_position()
    cbar_ax = cbar.ax
    new_cbar_position = [
        bbox.x0,            # x0
        bbox.y0 - pad,      # y0
        bbox.width,         # width
        fraction            # height
    ]
    cbar_ax.set_position(new_cbar_position)


def apply_fig_style(fig, ax, style='small', custom_size_cm=None,
                    line_size=1.0, marker_size=4, freeze_marker_size=True,
                    dpi=1200, colorbar=None, colorbar_orientation='vertical',
                    freeze_lines=True, freeze_ticks=True):
    styles = {
        'small': {'size_cm': (8, 6), 'base_fontsize': 10},
        'large': {'size_cm': (16.4, 10), 'base_fontsize': 10},
    }

    if style not in styles and custom_size_cm is None:
        raise ValueError(f"Style '{style}' not recognized. Use one of: {list(styles.keys())}, or provide 'custom_size_cm'.")

    # Use custom size if provided
    size_cm = custom_size_cm if custom_size_cm is not None else styles[style]['size_cm']
    base_fontsize = styles.get(style, {}).get('base_fontsize', 10)

    # Resize figure
    fig.set_dpi(dpi)
    fig.set_size_inches(cm2inch(size_cm))
    fig.tight_layout()

    # Aspect ratio for 'small'
    if style == 'small' and custom_size_cm is None:
        ax.set_box_aspect(0.625)

    # Scale font size
    fontsize = scale_fontsize(fig, base_width_cm=16.4, base_fontsize=base_fontsize)
    set_ax_fontsize(fontsize, ax)

    # Adjust colorbar (if any)
    if colorbar is not None:
        set_colorbar_fontsize(colorbar, fontsize)
        if colorbar_orientation == 'vertical':
            resize_colorbar_vertically(colorbar, ax)
        elif colorbar_orientation == 'horizontal':
            resize_colorbar_horizontally(colorbar, ax)

    # Freeze line and marker sizes
    if freeze_lines or freeze_marker_size:
        for line in ax.lines:
            if freeze_lines:
                line.set_linewidth(line_size)
            if freeze_marker_size:
                line.set_markersize(marker_size)

    # Freeze tick width and length
    if freeze_ticks:
        ax.tick_params(width=0.8, length=4)




def save_figure(fig, filename="figure", folder="figures", file_format="png", dpi=1200):
    """
    Save a matplotlib figure with specified settings.

    Parameters:
        fig (matplotlib.figure.Figure): The figure object to save.
        filename (str): The base name for the saved file.
        folder (str): The directory to save the file in.
        file_format (str): The file format (e.g., 'png', 'pdf', 'svg').
        dpi (int): The resolution of the saved figure.

    Returns:
        str: Full path of the saved file.
    """
    import os

    # Ensure the directory exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Construct the full file path
    full_path = os.path.join(folder, f"{filename}.{file_format}")

    # Save the figure
    fig.savefig(full_path, format=file_format, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved as {full_path}")

    return full_path


# fit TIME
def special_lorentzian(x, params = (1, 0, 1)):
    Gamma, x0, scale = params
    return scale*(1/np.pi)*(Gamma/2) / ( (x - x0)**2 + (Gamma/2)**2  )

def lorentzian(x, Gamma, x0, scale, bkgd):
    return scale*(1/np.pi)*(Gamma/2) / ( (x - x0)**2 + (Gamma/2)**2  ) + bkgd

def many_lorentzians(xaxis, *params):
    param_mtx = np.reshape(np.array(params), [-1, 3])
    return sum(special_lorentzian(xaxis, param_mtx[i]) for i in range(len(param_mtx)))

def exponential(x, a, b, c):
    return a*np.exp(-x/b) + c      # don't need to do bx+c in the exponent as this is covered for by the a scaling factor

def normalise_0_1(data_arrray):
    data_arrray = np.asarray(data_arrray)
    return (data_arrray - np.min(data_arrray)) / (np.max(data_arrray) - np.min(data_arrray))



def normalise_traces(traces, floors = None, flip = False):
    '''
    Normalisation function for traces. Normalised between a provided floor for *each* trace
    (or zero) and the maximum of each array OR between zero and the maximum, if no floors provided.

    If flip is true, then subtracts 1 at the end and flips, so it's an upwards peak (for peak fitting convenience).

    Inputs:
        traces: enumerable of individual np.arrays. Typically a tuple. 
        floors = None: enumerable of detector floors/blanks the gain setting used to take data. This value will be subtracted before normalisation. 
        flip = False: if true, make the background 0 and flip the peaks so that they go upwards (rather than down from 1). 
    
    Returns:
        cal_traces: tuple of individual np.arrays.
    '''
    
    # If no floors provided, make them all zero
    if not floors:
        floors = np.zeros(len(traces))
    # Do the thing
    if flip: # if flip true then flip it
        cal_traces = tuple( 
            -1*((trace - floors[i]) / (np.max(trace) - floors[i]) - 1) for i, trace in enumerate(traces)
        )
    else: # otherwise don't flip it
        cal_traces = tuple( 
            (trace - floors[i]) / (np.max(trace) - floors[i]) for i, trace in enumerate(traces)
        )
    
    return cal_traces