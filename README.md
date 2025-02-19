README for GitHub

---

Gaussian Quantum Device Simulation

This repository contains Python scripts for simulating Gaussian and linear quantum devices using the StrawberryFields library. More information on implementation and syntax is available at https://strawberryfields.ai/. These simulations model quantum systems with loss, apply random transformations, and perform homodyne and heterodyne measurements to reconstruct key parameters of the quantum devices.

1. Gaussian Device Simulation (`gaussian_homodyne_heterodyne.py`)

Key Features
- Homodyne and Heterodyne Detection: Performs quantum measurements using homodyne (`experiment_homodyne`) and heterodyne (`experiment_heterodyne`) detection methods.
- Random Symplectic Transformations: Uses `thewalrus.random.random_symplectic` to generate random symplectic matrices.
- Loss Modeling: Simulates photon loss using beamsplitters with adjustable transmittance.
- Matrix Reconstruction: Compares reconstructed symplectic matrices against the true transformations using the Frobenius norm.
- Batch Simulations: Runs multiple trials across different device modes (2-20) and varying levels of loss (0% and 50%).
- CSV Logging: Stores results in CSV files.

2. Linear Device Simulation (`unitary_homodyne_heterodyne.py`)

Key Features
- Homodyne and Heterodyne Detection: Implements `experiment_homodyne` and `experiment_heterodyne` to measure quadratures in linear optical devices.
- Random Unitary Transformations: Uses `thewalrus.random.random_interferometer` to generate random unitary matrices.
- Loss Modeling: Simulates loss using beamsplitters with user-defined transmissivity.
- Unitary Matrix Reconstruction: Compares the reconstructed unitary matrix fidelity against the true transformation.
- Batch Simulations: Runs multiple trials for device modes (1-20) and varying loss levels (0% and 50%).
- CSV Logging: Stores results in CSV files.

3. Figure Generation (`plot_reconstructions`)

Key Features
- Reconstruction Statistics Extraction: Computes mean and standard deviation of homodyne and heterodyne reconstruction methods across different device mode sizes.
- Error Bar Plots: Visualizes reconstruction trends across different device mode sizes for varying loss levels.
- Batch Processing: Supports automatic extraction and visualization from multiple CSV files.

Note: Portions of the plotting script were generated with assistance from ChatGPT.


Usage: 
Save file to a local directory and run the script directly to simulate linear devices with predefined parameters. For example:
`python unitary_homodyne_heterodyne.py`
By default, the script:
- Simulates devices with 1 to 20 modes.
- Uses a coherent state amplitude of 1000.
- Runs 50 trials per device mode.
- Logs results to CSV files.

Simulation results will be recorded to csv files in the same directory that you have the .py file saved to. In order to graph results, the `plot_reconstructions.py` file must be saved in the same directory and run.

Generating Plots
- To generate plots, modify plot_reconstructions.py to include the correct CSV file paths, then execute:
`python plot_reconstructions.py`
This script extracts reconstruction statistics from specified CSV files and generates plots comparing trends over different mode sizes under different loss conditions. By default, it plots scaled Frobenius norms.

Dependencies
- Python 3
- `numpy`
- `strawberryfields`
- `thewalrus`
- `matplotlib`
- `csv`
- `statistics`

These scripts allow for efficient benchmarking of homodyne and heterodyne measurement strategies in both Gaussian and linear quantum devices. The results can be used to analyze Frobenius norm trends and the impact of loss on quantum transformations.
