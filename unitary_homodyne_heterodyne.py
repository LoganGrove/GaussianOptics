# Derived from Gaussian_homodyne_heterodyne.ipynb
import numpy as np
import strawberryfields as sf
from strawberryfields.ops import *
from thewalrus.random import random_interferometer
import matplotlib.pyplot as plt
import csv
import matplotlib.pyplot as plt


np.set_printoptions(suppress = True, precision = 6) # Display all matrix entries with 3 decimals
sf.hbar = 1  # Default is hbar = 2, Set hbar to 1 to align with the definition of quadrature operators

def fidelity(U_exp, U):
    # Computes the normalized fidelity between two unitary matrices U_exp (experimentally reconstructed) and U (generated unitary).
    U_dagger = np.conjugate(np.transpose(U))
    trace_value = np.trace(np.dot(U_dagger, U_exp))
    N = U_exp.shape[0]
    U_exp_dagger = np.conjugate(np.transpose(U_exp))
    denominator = np.sqrt(N * np.trace(np.dot(U_exp_dagger, U_exp)))
    fidelity = np.abs(trace_value / denominator) ** 2
    return fidelity


def experiment_homodyne(mode, phase_in, phase_out, amplitude, device_modes, loss, U_rand, eng):
    """
    experiment_homodyne is a function to make an input state, model loss, apply a unitary transformation, and perform homodyne detection over all modes.
    Uses strawberry fields 'Interferometer' function to decompose a unitary operation.
    It returns the result of measuring the specified quadrature (<x> or <p>) over all modes
    
    Parameters:
    - mode: The mode into which a coherent state is input (all other modes receive vacuum states)
    - phase_in: Phase of the input coherent state |alpha * e^(i*phase_in)>
    - phase_out: Determines quadrature measurement (0 for <X>, pi/2 for <P>)
    - amplitude: Amplitude of the input coherent state (set to 0 when measuring displacement vector d)
    - device_modes: Number of input modes (N), each paired with a loss mode
    - loss: Fractional loss (0 to 1), is the loss (0,1) used to calculate the transmissivity of each beamsplitter
    - U_rand: Random unitary transformation applied to the input state
    - eng: StrawberryFields engine for running the simulation
    
    Returns:
    - Homodyne measurement results for all modes
    """
    experiment = sf.Program(2*device_modes) #This creates a experiment of 2N modes. The first N of these are the modes of the device, and the next N are the loss modes.
    with experiment.context as q:
        for input in range(2*device_modes): #This assigns an input in each mode. Note that the variable 'mode' will only take values < N, so inputs to loss modes are always vacuum
            if input == mode:
                Coherent(amplitude, phase_in) | q[input] # There is exactly one mode with coherent state input each time.
            else:
                Vacuum() | q[input]

        for i in range(device_modes):
            BSgate(np.arccos(np.sqrt(1-loss)), 0) | (q[i], q[i + device_modes]) # Loss is modeled by beamsplitters

        Interferometer(U_rand) | tuple([q[i] for i in range(device_modes)]) # Device that implements the random unitary transformation

        for j in range(device_modes):
            MeasureHomodyne(phase_out) | q[j] # phase_out determines if <x> or <p> is measured. Either <x> or <p> is measured in all modes

        results = eng.run(experiment)
        return results.samples[0] # results.samples[0] contains the quadrature values measured


def experiment_heterodyne(mode, phase_in, amplitude, device_modes, loss, U_rand, eng): 
    """
    Simulates heterodyne detection on a quantum optical system.
    
    Parameters are the same as experiment_homodyne, except phase_out is not needed since heterodyne measurement inherently measures both quadratures.
    
    Returns:
    - Heterodyne measurement results for all modes
    """
    experiment = sf.Program(2*device_modes)
    with experiment.context as q:
        for input in range(2*device_modes):
            if input == mode:
                Coherent(amplitude, phase_in) | q[input]
            else:
                Vacuum() | q[input]

        for i in range(device_modes):
            BSgate(np.arccos(np.sqrt(1-loss)) , 0) | (q[i], q[i + device_modes])

        Interferometer(U_rand) | tuple([q[i] for i in range(device_modes)])

        for j in range(device_modes):
            MeasureHeterodyne() | q[j]

        results = eng.run(experiment)
        return results.samples[0]


def run_simulation(amplitude, loss, num_trials, mode_list, filename):
    """
    Runs a simulation comparing homodyne and heterodyne detection for reconstructing a unitary transformation.
    Computes fidelity to measure reconstruction accuracy. Saves the number of modes of the device, the trial number, 
    and the homodyne and heterodyne fidelities and norms in a row of a csv file. Although the script is set up to plot norms,
    fidelity may also be used to compare reconstructions for unitaries. 
    """
    fieldnames = ['device_modes', 'trial', 'hetero_fidelity', 'homo_fidelity', 'homo_norm', 'hetero_norm']

    # Open file once at the beginning
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()  

    for device_modes in mode_list:
        # print("Mode: "+ str(device_modes))    # This is to let you know which mode we are in. Can be helpful to know for large simulations

        for trial in range(num_trials):
            eng = sf.Engine("gaussian") # Set up SF environment
            U_recon_hetero = np.full((device_modes, device_modes), 0, dtype=complex) # Empty matrix to fill with heterodyne reconstruction results 
            U_recon_homo = np.full((device_modes, device_modes), 0, dtype=complex) # Empty matrix to fill with homodyne reconstruction results 
            U_rand = random_interferometer(device_modes) # Generate a random unitary which is to be reconstructed

            # Reconstruction of the unitary matrix using heterodyne measurements
            for mode in range(device_modes): # Step over all modes
                quadratures1 = experiment_heterodyne(mode, 0, amplitude, device_modes, loss, U_rand, eng) # Measure all quads with a real coherent state
                quadratures2 = experiment_heterodyne(mode, 0, amplitude, device_modes, loss, U_rand, eng) # Measure a second time to ensure equal resource counting with homodyne reconstruction
                quadratures = (quadratures1 + quadratures2)/2 # Average heterodyne results for resource consistency between dyne approaches
                U_recon_hetero[:,mode] = (quadratures/amplitude).T # Lossy reconstruction

            transmittance_hetero = abs((np.linalg.det(U_recon_hetero)))**(1/(device_modes)) # Estimate loss paramater
            U_recon_hetero /= transmittance_hetero # Account for loss in reconstruction. The result here is the fully reconstructed heterodyne unitary.
            fidelity_hetero = fidelity(U_recon_hetero, U_rand) # Save this to file if needed
            fro_norm_hetero = np.linalg.norm(U_recon_hetero - U_rand, 'fro') #Save this to file if needed

            # Reconstruction of the unitary matrix using homodyne measurements
            for mode in range(device_modes): # Step over all modes
                X_quads = experiment_homodyne(mode, 0, 0, amplitude, device_modes, loss, U_rand, eng) # Measure all quads with a real coherent state
                P_quads = experiment_homodyne(mode, 0, np.pi/2, amplitude, device_modes, loss, U_rand, eng) # Measure a second time to ensure equal resource counting with homodyne reconstruction
                U_recon_homo[:,mode] += ((X_quads +1j* P_quads)/(np.sqrt(2) * amplitude)).T # Lossy reconstruction

            transmittance_homo = abs((np.linalg.det(U_recon_homo)))**(1/device_modes) # Estimate loss paramater
            U_recon_homo /= transmittance_homo # Account for loss in reconstruction. The result here is the fully reconstructed homodyne unitary.
            fidelity_homo = fidelity(U_recon_homo,U_rand) # Save this to file if needed
            fro_norm_homo = np.linalg.norm(U_recon_homo - U_rand, 'fro') #Save this to file if needed

            # Write results to file
            with open(filename, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'device_modes': device_modes, 'trial': trial, 'hetero_fidelity': fidelity_hetero, 'homo_fidelity': fidelity_homo, 'homo_norm': fro_norm_homo, 'hetero_norm': fro_norm_hetero})

# Define parameters for the simulation
amplitude = 1000 # amplidude of input coherent state
loss = 0.5 # Percentge of loss to be modeled
num_trials = 50 # Number of random unitaries to be reconstruced
mode_list = range(1,21) # Will reconstruct 'num_trials' of each device mode size included in the range
filename = 'test_file_name.csv'
run_simulation(amplitude, loss, num_trials, mode_list, filename)


