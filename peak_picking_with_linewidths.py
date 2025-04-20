import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
from datetime import datetime
import my_functions as mf

data_path = "data_optics/data_fergus/2025-03-25_IH_line_magnet_ramp/"
save_path = "data_optics/peak_picking_data_2"
selected_points = []

def read_B_field(file_name):
    match = re.search(r'fld_([0-9.]+)', file_name)
    if match: 
        b_field = float(match.group(1).strip("."))
        return b_field
    else: 
        print("Error: field value not found in filename.")

def get_peak_parameters(clicks, freq_axis, voltage_trace):
    """
    Extracts [x0, y0, Γ, scale] from 3 clicks: center, left, right at FWHM
    """
    if len(clicks) % 3 != 0:
        print("Warning: Number of clicks is not divisible by 3. Ignoring incomplete peak definitions.")
        clicks = clicks[:(len(clicks) // 3) * 3]

    peaks = []
    for i in range(0, len(clicks), 3):
        x0, _ = clicks[i]       # center
        xL, _ = clicks[i + 1]   # left FWHM
        xR, _ = clicks[i + 2]   # right FWHM

        Γ = abs(xR - xL)        # linewidth estimate
        y0 = np.interp(x0, freq_axis, voltage_trace)  # peak height
        scale = y0 * np.pi * Γ / 2                    # estimated area
        peaks.append((x0, y0, Γ, scale))
    return peaks

def save_selected_points(description='peak'):
    peak_name = input("Enter the peak name: ")
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    filename = f"{timestamp}_{description}_{peak_name}.csv"
    df = pd.DataFrame(selected_points, columns=[
        'Frequency (GHz)', 'Signal', 'Linewidth (GHz)', 'Scale', 'B Field (mT)'
    ])
    file_path = os.path.join(save_path, filename)
    df.to_csv(file_path, index=False)
    print(f'Saved selected points to {filename}')

def plot_individual_spectra(data_path, repeat=1):
    global current_voltage, current_field, freq_axis_GHz
    print("Reading Data...")
    data_files = sorted(glob.glob(data_path + '*'))
    voltages = []
    fields = []

    for file in data_files:
        data = pd.read_csv(file, header=None)
        time_axis = data.iloc[0].values.astype(float)
        voltage = data.iloc[1].values.astype(float)
        freq_axis = data.iloc[2].values.astype(float)

        freq_axis_GHz = freq_axis / 1e9
        voltage = mf.low_pass_band_filter_fft(20, freq_axis_GHz, voltage, plot_check=False)
        
        B_field = read_B_field(file_name=file)
        fields.append(B_field)
        voltages.append(voltage)

    voltage_traces = tuple(voltages)
    floors = tuple(-0.0035351922089636857 * np.ones(len(voltage_traces)))
    normalized_traces = mf.normalise_traces(voltage_traces, floors=floors, flip=True)
    signals = np.array(normalized_traces)

    fields = np.array(fields)
    sort_idx = np.argsort(fields)
    fields = fields[sort_idx]
    signals = signals[sort_idx]

    try:
        field_step = int(input("Enter the size of field steps to take (e.g., 5mT): "))
        if field_step <= 0:
            raise ValueError("Field step must be a positive integer.")
    except ValueError:
        print("Invalid input. Using default field step of 5.")
        field_step = 5
    for  peak_index in range(1,repeat+1):
        print(f"\n--- Repeat round {peak_index + 1} of {repeat} ---")
        selected_points =[]
        for i in range(len(fields) - 1, -1, -field_step):
            current_voltage = signals[i]
            current_field = fields[i]
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(freq_axis_GHz, current_voltage, label=f'B Field: {current_field * 1000:.1f} mT')
            ax.set_xlabel('Frequency (GHz)')
            ax.set_ylabel('Signal')
            #ax.set_xlim([-5, 5])
            # ax.set_xticks(np.arange(-5, 5.2, 0.5))
            ax.set_yticks(np.arange(0,0.5,0.02))
            ax.set_title(f'Spectrum at {current_field * 1000:.1f} mT')
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            plt.show(block=False)

            print("Click THREE points per peak: center, left FWHM, right FWHM. Press ENTER when done.")
            clicks = plt.ginput(-1, timeout=0)
            plt.close()

            # Compute peak parameters from user clicks
            peak_params = get_peak_parameters(clicks, freq_axis_GHz, current_voltage)
            for x0, y0, Gamma, scale in peak_params:
                selected_points.append((x0, y0, Gamma, scale, current_field * 1000))

        save_selected_points(description="absorption_peaks_2025-03-25_site1")

plot_individual_spectra(data_path, repeat=6)

