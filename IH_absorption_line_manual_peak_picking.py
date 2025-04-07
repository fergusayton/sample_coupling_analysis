import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
from datetime import datetime
import my_functions as mf

data_path = "data_optics/data_fergus/2025-03-25_IH_line_magnet_ramp/"
save_path = "data_optics/peak_picking_data"
selected_points = []

def read_B_field(file_name):
    match = re.search(r'fld_([0-9.]+)', file_name)
    if match: 
        b_field = float(match.group(1).strip("."))
        return b_field
    else: 
        print("Error: field value not found in filename.")

def onclick(event):
    if event.xdata and event.ydata:
        # Find the nearest voltage value to the clicked x-coordinate (frequency)
        freq_clicked = event.xdata
        voltage_clicked = np.interp(freq_clicked, freq_axis_GHz, current_voltage)
        selected_points.append((freq_clicked, voltage_clicked, current_field * 1000))
        plt.scatter(freq_clicked, voltage_clicked, color='red')
        plt.draw()

def save_selected_points(description='peak'):
    peak_name = input("Enter the peak name: ")
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    filename = f"{timestamp}_{description}_{peak_name}.csv"
    df = pd.DataFrame(selected_points, columns=['Frequency (GHz)','Voltage (V)',  'B Field (mT)', ])
    file_path = os.path.join(save_path, filename)
    df.to_csv(file_path, index=False)
    print(f'Saved selected points to {filename}')

def plot_individual_spectra(data_path):
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

    fields = np.array(fields)
    voltages = np.array(voltages)

    sort_idx = np.argsort(fields)
    fields = fields[sort_idx]
    voltages = voltages[sort_idx]

    # Prompt user for number of field steps
    try:
        field_step = int(input("Enter the size of field steps to take (e.g., 5mT): "))
        if field_step <= 0:
            raise ValueError("Field step must be a positive integer.")
    except ValueError:
        print("Invalid input. Using default field step of 5.")
        field_step = 5

    # Plot individual spectra based on user-defined field steps
    for i in range(len(fields) - 1, -1, -field_step):
        current_voltage = voltages[i]
        current_field = fields[i]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(freq_axis_GHz, current_voltage, label=f'B Field: {current_field * 1000:.1f} mT')
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Voltage')
        ax.set_title(f'Spectrum at {current_field * 1000:.1f} mT')
        ax.grid(True)
        ax.legend()
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show(block=False)
        print("Click on the graph to select peaks. Press any key to move to the next plot.")
        plt.waitforbuttonpress()
        plt.close()

    save_selected_points(description="absorption_peaks_2025-03-25")

plot_individual_spectra(data_path)
