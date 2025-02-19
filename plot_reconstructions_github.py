import csv
import statistics
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter

# Function to extract statistics for hetero_norm and homo_norm by mode
def extract_norm(input_filenames):
    hetero_data = {}
    homo_data = {}

    # Loop through each file and compile data
    for input_filename in input_filenames:
        with open(input_filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                mode = int(row['device_modes'])

                # May also replace instances of hetero_norm and homo norm with
                # hetero_fidelity and homo_fidelity within this for loop to plot
                # fidelity instead of norms.
                hetero_norm = float(row['hetero_norm'])
                homo_norm = float(row['homo_norm'])

                if mode not in hetero_data:
                    hetero_data[mode] = []

                # Record scaled norms. If plotting fidelity, remove mode dependence
                hetero_data[mode].append(hetero_norm/mode)

                if mode not in homo_data:
                    homo_data[mode] = []
                homo_data[mode].append(homo_norm/mode)

    hetero_statistics_results = {mode: {'mean': statistics.mean(hetero_data[mode]),
                                         'stddev': statistics.stdev(hetero_data[mode])}
                                 for mode in hetero_data}

    homo_statistics_results = {mode: {'mean': statistics.mean(homo_data[mode]),
                                       'stddev': statistics.stdev(homo_data[mode])}
                               for mode in homo_data}

    num_trials = len(next(iter(hetero_data.values())))
    return hetero_statistics_results, homo_statistics_results, num_trials

# Function to plot the results
def plot_norm(file_groups, loss_levels):
    markers = {
        (False, 50): {'marker': 's', 'label': 'Homodyne 50% Loss', 'markersize': 10, 'color': 'red'},
        (False, 0): {'marker': 'o', 'label': 'Homodyne 0% Loss', 'markersize': 10, 'color': 'orange'},
        (True, 50): {'marker': 'D', 'label': 'Heterodyne 50% Loss', 'markersize': 10, 'color': 'green'},
        (True, 0): {'marker': '^', 'label': 'Heterodyne 0% Loss', 'markersize': 10, 'color': 'blue'}
    }

    plt.figure(figsize=(10, 8))

    for files, loss in zip(file_groups, loss_levels):
        hetero_statistics, homo_statistics, num_trials = extract_norm(files)
        device_modes = sorted(hetero_statistics.keys())

        plt.errorbar(device_modes, 
                    [hetero_statistics[mode]['mean'] for mode in device_modes],
                    yerr=[hetero_statistics[mode]['stddev'] / np.sqrt(num_trials) for mode in device_modes],
                    fmt=markers[(True, loss)]['marker'], 
                    capsize=5, 
                    label=markers[(True, loss)]['label'], 
                    color=markers[(True, loss)]['color'])

        plt.errorbar(device_modes, 
                    [homo_statistics[mode]['mean'] for mode in device_modes],
                    yerr=[homo_statistics[mode]['stddev'] / np.sqrt(num_trials) for mode in device_modes],
                    fmt=markers[(False, loss)]['marker'], 
                    capsize=5, 
                    label=markers[(False, loss)]['label'], 
                    color=markers[(False, loss)]['color'])


        plt.xlabel('Number of Modes', fontsize=24)
        plt.ylabel('Average Frobenius Norm', fontsize=24)
        plt.gca().tick_params(axis='both', which='major', labelsize=20)
        plt.gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.gca().set_ylim(0, 0.17)
        handles, labels = plt.gca().get_legend_handles_labels()
        desired_order = ['Homodyne 50% Loss', 'Homodyne 0% Loss', 'Heterodyne 50% Loss', 'Heterodyne 0% Loss']
        sorted_handles_labels = sorted(zip(handles, labels), key=lambda hl: desired_order.index(hl[1]) if hl[1] in desired_order else float('inf'))
        sorted_handles, sorted_labels = zip(*sorted_handles_labels)
        plt.legend(sorted_handles, sorted_labels, loc="upper left", prop={'size': 15})
        plt.title("Average Frobenius Norm vs Number of Modes", fontsize=24)
    plt.show()

# File paths
lossless_files = ['test_gaussian_file_lossless.csv']

lossy_files = ['test_gaussian_file_lossy.csv']
# Replace file paths with unitary file names if desired

file_groups = [lossy_files, lossless_files]

loss_levels = [50, 0]

plot_norm(file_groups, loss_levels)

