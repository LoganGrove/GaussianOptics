import numpy as np
import strawberryfields as sf
from strawberryfields.ops import *
from thewalrus.random import random_symplectic
import matplotlib.pyplot as plt
import csv
import matplotlib.pyplot as plt


np.set_printoptions(suppress = True, precision = 6) # Display all matrix entries with 3 decimals
sf.hbar = 1  # Default is hbar = 2, Set hbar to 1 to align with the definition of quadrature operators

def experiment_homodyne(mode, phase_in, phase_out, amplitude, device_modes, loss, S_rand, eng):
    """
    experiment_homodyne is a function to make an input state, model loss, apply symplectic transformation, and perform homodyne detection over all modes.
    Uses strawberry fields 'GaussianTransform' to decompose a symplectic matrix,
    It returns the result of measuring the specified quadrature (<x> or <p>) over all modes
    
    Parameters:
    - mode: The mode into which a coherent state is input (all other modes receive vacuum states)
    - phase_in: Phase of the input coherent state |alpha * e^(i*phase_in)>
    - phase_out: Determines quadrature measurement (0 for <X>, pi/2 for <P>)
    - amplitude: Amplitude of the input coherent state (set to 0 when measuring displacement vector d)
    - device_modes: Number of input modes (N), each paired with a loss mode
    - loss: Fractional loss (0 to 1), is the loss (0,1) used to calculate the transmissivity of each beamsplitter
    - S_rand: Random symplectic transformation applied to the input state
    - eng: StrawberryFields engine for running the simulation
    
    Returns:
    - Homodyne measurement results for all modes
    """
    experiment = sf.Program(2*device_modes) #This creates a experiment of 2N modes. The first N of these are the modes of the device, and the next N are the loss modes.
    with experiment.context as q:
        for input in range(2*device_modes): #This assigns an input in each mode. Note that the variable 'mode' will only take values < N, so inputs to loss modes are always vacuum
            if input == mode:
                Coherent(amplitude, phase_in) | q[input]  # There is exactly one mode with coherent state input each time.
            else:
                Vacuum() | q[input]

        for i in range(device_modes):
            BSgate(np.arccos(np.sqrt(1-loss)), 0) | (q[i], q[i + device_modes]) # Loss is modeled by beamsplitters

        GaussianTransform(S_rand) | tuple([q[i] for i in range(device_modes)]) # Device that implements the random symplectic transformation

        for j in range(device_modes):
            MeasureHomodyne(phase_out) | q[j] # phase_out determines if <x> or <p> is measured. Either <x> or <p> is measured in all modes

        results = eng.run(experiment)
        return results.samples[0] # results.samples[0] contains the quadrature values measured


def experiment_heterodyne(mode, phase_in, amplitude, device_modes, loss, S_rand, eng): # No phase_out needed as we perform heterodyne measurement at the end.
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

        GaussianTransform(S_rand) | tuple([q[i] for i in range(device_modes)])

        for j in range(device_modes):
            MeasureHeterodyne() | q[j]

        results = eng.run(experiment)
        return results.samples[0]


def run_simulation(amplitude, loss, num_trials, mode_list, filename):
    """
    Runs a simulation comparing homodyne and heterodyne detection for reconstructing a symplectic transformation.
    Computes the frobenius norm to quantify reconstruction accuracy. Saves the number of modes of the device, the trial number, 
    and the homodyne and heterodyne norms in a row of a csv file.
    """
    fieldnames = ['device_modes', 'trial', 'hetero_norm', 'homo_norm']

    # Open file once at the beginning
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()  

    for device_modes in mode_list:
        # print("Mode: "+ str(device_modes))    # This is to let you know which mode we are in. Can be helpful to know for large simulations

        for trial in range(num_trials):
            eng = sf.Engine("gaussian") # Set up SF environment
            S_recon_hetero = np.full((2*device_modes, 2*device_modes), 0, dtype=float) # Empty matrix to fill with heterodyne reconstruction results 
            S_recon_homo = np.full((2*device_modes, 2*device_modes), 0, dtype=float) # Empty matrix to fill with homoodyne reconstruction results 
            S_rand = random_symplectic(device_modes) # Generate a random symplectic matrix which is to be reconstructed


            # Reconstruction of the symplectic matrix using heterodyne measurements
            for mode in range(device_modes): # Step over all modes
                quadratures1 = experiment_heterodyne(mode, 0, amplitude, device_modes, loss, S_rand, eng) # Measure all quads with a real coherent state
                quadratures2 = experiment_heterodyne(mode, 0, amplitude, device_modes, loss, S_rand, eng) # Mesure a second time as well for equal resource with homodyne reconstruction

                X_quads = np.real((quadratures1 + quadratures2)/2) # Average the X quadrature measurments
                P_quads = np.imag((quadratures1 + quadratures2)/2) # Average the P quadrature measurments

                S_recon_hetero[:,mode] += np.concatenate((X_quads/amplitude, P_quads/amplitude), axis=0).T # Lossy heterodyne reconstruction of left side of S (real alpha input)

                quadratures1 = experiment_heterodyne(mode, np.pi/2, amplitude, device_modes, loss, S_rand, eng) # Measure all quads with a imaginary coherent state
                quadratures2 = experiment_heterodyne(mode, np.pi/2, amplitude, device_modes, loss, S_rand, eng) # Mesure a second time as well for equal resource with homodyne reconstruction

                X_quads = np.real((quadratures1 + quadratures2)/2) # Average the X quadrature measurments
                P_quads = np.imag((quadratures1 + quadratures2)/2) # Average the P quadrature measurments

                S_recon_hetero[:,device_modes + mode] += np.concatenate((X_quads/amplitude, P_quads/amplitude), axis=0).T # Lossy heterodyne reconstruction of right side of S (imaginary alpha input)

            transmittance_hetero = abs((np.linalg.det(S_recon_hetero)))**(1/(device_modes)) # Estimate loss parameter
            S_recon_hetero /= np.sqrt(transmittance_hetero) # Account for loss in reconstruction. The result here is the fully reconstructed heterodyne symplectic matrix.
            fro_norm_hetero = np.linalg.norm(S_recon_hetero - S_rand, 'fro') # Save this to file if needed

            # Reconstruction of the symplectic matrix using homodyne measurements
            for mode in range(device_modes): # Step over all modes
                X_quads = experiment_homodyne(mode, 0, 0, amplitude, device_modes, loss, S_rand, eng) # Measure X quadratures 
                P_quads = experiment_homodyne(mode, 0, np.pi/2, amplitude, device_modes, loss, S_rand, eng) # Measure P quadratures
                S_recon_homo[:,mode] += np.concatenate((X_quads, P_quads)/(np.sqrt(2) * amplitude), axis=0).T # Lossy heterodyne reconstruction of left side of S (real alpha input)

                X_quads = experiment_homodyne(mode, np.pi/2, 0, amplitude, device_modes, loss, S_rand, eng) # Measure X quadratures
                P_quads = experiment_homodyne(mode, np.pi/2, np.pi/2, amplitude, device_modes, loss, S_rand, eng) # Measure P quadratures
                S_recon_homo[:,device_modes + mode] += np.concatenate((X_quads, P_quads)/(np.sqrt(2) * amplitude), axis=0).T # Lossy homodyne reconstruction of left side of S (imaginary alpha input)

            transmittance_homo = abs((np.linalg.det(S_recon_homo)))**(1/device_modes) # Estimate loss parameter
            S_recon_homo /= np.sqrt(transmittance_homo) # Account for loss in reconstruction. The result here is the fully reconstructed homodyne symplectic matrix.
            fro_norm_homo = np.linalg.norm(S_recon_homo - S_rand, 'fro') # Save this to file if needed

            # Write results to file
            with open(filename, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({'device_modes': device_modes, 'trial': trial, 'hetero_norm': fro_norm_hetero, 'homo_norm': fro_norm_homo})



# Define parameters for the simulation
amplitude = 1000 # amplidude of input coherent state
loss = 0.5 # Percentge of loss to be modeled
num_trials = 500 # Number of random unitaries to be reconstruced
mode_list = range(1,21) # Will reconstruct 'num_trials' of each device mode size included in the range
filename = 'test_file_name.csv'
run_simulation(amplitude, loss, num_trials, mode_list, filename)
