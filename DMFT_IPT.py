import numpy as np
import matplotlib.pyplot as plt

########################################################################
# INPUT ROUTINES
########################################################################

def read_input(input):

    '''
    Reads the content of the file input. It should be a txt file with the variables:

        D0: half bandwith of the Bethe lattice. If explicit DOS is given, this value is irrelevant, but some should be included.
        u0: Hubbard interaction, in units of the half bandwidth.
        nloop: number of DMFT loops.
        xmix: mix between old and new self energy.
        n_omega: number of Matsubara frequencies.
        beta: inverse temperature.
        tol: convergence threshold for the DMFT loop.
        seed: initial trial self energy. 0 for metallic, 1 for insulating, 2 for external file.
    '''

    try:

        with open(input, 'r') as file:
            data = file.readlines()
        return float(data[0]), float(data[1]), int(data[2]), float(data[3]), int(data[4]), float(data[5]), float(data[6]), int(data[7])
    
    except:

        print('INCORRECT INPUT FORMATTING. The input file should contain:')
        print('  D0: half bandwith of the Bethe lattice. If explicit DOS is given, this value is irrelevant, but some should be included.')
        print('  u0: Hubbard interaction, in units of the half bandwidth.')
        print('  nloop: number of DMFT loops.')
        print('  xmix: mix between old and new self energy.') 
        print('  n_omega: number of Matsubara frequencies.') 
        print('  beta: inverse temperature.')
        print('  tol: convergence threshold for the DMFT loop.')
        print('  seed: initial trial self energy. 0 for metallic, 1 for insulating, 2 for external file.')



def read_DOS(DOS = None, D0 = None):
    
    if DOS is None:
        
        energies = np.linspace(-D0, D0, 1000)
        DOS = 2*np.sqrt(1-(energies/D0)**2)/(np.pi * D0)

        return energies, DOS
    
    elif isinstance(DOS, str):

        data1 = np.loadtxt(DOS)
        energies = data1[:,0]
        DOS   = data1[:,1]

        return energies, DOS
    
    else:
        raise ValueError("Invalid input: DOS must be a string or None")
    


def initialize_Sigma(n_omega, beta, seed, filename = None):

    Momega = np.pi/beta*( np.array(-2*n_omega*np.fft.fftfreq(n_omega),dtype=float) +1.)
    tau    = np.linspace(0, beta*(1. - 1./n_omega), n_omega)

    if seed == 0:
        S_imp = np.zeros((len(Momega)), dtype=complex)

    elif seed == 1:
        S_imp = 1/(1j * Momega)

    else:
        data = np.loadtxt(filename)
        S_imp = data[:,1] + 1j * data[:,2]

    return Momega, tau, S_imp
    

########################################################################
# DMFT ROUTINES
########################################################################

# Computation of the local Green's function

def Local_Green (S_imp, energies, dos, Momega):

    """
    From the self-energy in Matsubara axis, computes the local Green's
    function of the system, as an integral over energies.

    Input:
        S_imp: The current impurity self-energy in Matsubara axis
        Energies: The energy grid used to obtain the local Green's function
        dos: The input DOS of the system, evaluated in Energies
        Momega: Matsubara frequencies grid


    Output:
        G_loc: Local Green's function in Matsubara axis

    """
    
    from scipy.integrate import simpson
    denergies_bethe = energies[2] - energies[1]

    return simpson( dos/(1j*Momega[:,np.newaxis]-energies-S_imp[:,np.newaxis]), dx=denergies_bethe)



# IPT solver; computes the new self-energy

def Self_Energy_from_G (g0, U, tau, Momega):

    """
    IPT solver to obtain the new self-energy in Matsubara axis from the
    Weiss mean field in imaginary time axis.

    Input:
        g0: Weiss mean field in imaginary time axis
        U: Hubbard interaction parameter
        tau: Imaginary time grid
        Momega: Matsubara frequencies grid


    Output:
        new_S_imp: Updated self-energy in the Matsubara axis

    """

    dtau = tau[1] - tau[0]
    esp = np.exp(1.0j*np.tensordot(tau, Momega, 0))
    g0_minus = -np.roll(g0[::-1],1)
    g0_minus[0] = -g0_minus[0]
    integrand = g0*g0*g0_minus
    new_S_imp = -U*U*dtau*np.matmul(integrand,esp)
    
    return new_S_imp



# Fast inverse Fourier transform

def matsubara_ifft(G_omega, beta, Momega):

    """
    Fast inverse Fourier transform. Computes the function in imaginary 
    time axis. It's assumed that both time and frequency axis have the
    same length.

    Input:
        G_omega: Function in Matsubara axis
        beta: Inverse temperature of the system
        Momega: Matsubara frequencies grid


    Output:
        G_loc: Local Green's function in Matsubara axis

    """

    
    N = len(Momega)
#    freq = np.pi/beta*( np.array(-2*N*np.fft.fftfreq(N),dtype=float) +1.)
    k = np.arange(N,dtype='float')
    ifft = -1./2 + N/beta * np.exp(-1.0j*np.pi*k/N) * np.fft.ifft(G_omega - 1./(1.0j*Momega))
    
    return ifft



# DMFT first iteration and subsequent iterations

def DMFT_first_iteration(u0, S_imp, Energies, dos, Momega, tau, beta):

    """
    First iteration of the DMFT loop. Returns the updated self-energy,  
    as well as the non-interacting Green's function and the impurity 
    Green's function.

    Input:
        u0: The Hubbard interaction parameter
        S_imp: The current impurity self-energy in Matsubara axis
        Energies: The energy grid used to obtain the local Green's function
        dos: The input DOS of the system, evaluated in Energies
        Momega: Matsubara frequencies grid
        tau: Imaginary time grid
        beta: Inverse temperature of the system


    Output:
        G_loc: Local Green's function in Matsubara axis
        g_0: Weiss mean field in Matsubara axis
        g_0_tau: Weiss mean field in imaginary time axis
        new_S_imp: Updated self-energy in Matsubara axis, from IPT solver 

    """
    
    G_loc = Local_Green(S_imp, Energies, dos, Momega)    # Local GF in Matsubaras
  
    g_0 = G_loc/(1+G_loc*S_imp)                          # Weiss MF from Dyson equation
    g_0_tau = matsubara_ifft(g_0, beta, Momega)          # Weiss MF in imaginary time axis

    # Sanity check of the Weiss MF in the first iteration
    fig, ax = plt.subplots(1, 1, figsize = (8,6))
    ax.plot(tau, g_0_tau.real, 'k-')
    ax.set_title('First iteration', fontsize = 16)
    ax.set_xlabel(r'$\tau$', fontsize = 14)
    ax.set_ylabel(r'$\text{Re} \; G_0 (\tau)$', fontsize = 14)
    plt.show()


    new_S_imp = Self_Energy_from_G(g_0_tau, u0, tau, Momega) # Updated self-energy 

    return G_loc, g_0, g_0_tau, new_S_imp


def DMFT_iteration(u0,S_imp,Energies,dos,Momega,tau,beta):

    """
    Iteration of the DMFT loop. Returns the updated self-energy,  
    as well as the non-interacting Green's function and the impurity 
    Green's function.

    Input:
        u0: The Hubbard interaction parameter
        S_imp: The current impurity self-energy in Matsubara axis
        Energies: The energy grid used to obtain the local Green's function
        dos: The input DOS of the system, evaluated in Energies
        Momega: Matsubara frequencies grid
        tau: Imaginary time grid
        beta: Inverse temperature of the system


    Output:
        G_loc: Local Green's function in Matsubara axis
        g_0: Weiss mean field in Matsubara axis
        g_0_tau: Weiss mean field in imaginary time axis
        new_S_imp: Updated self-energy in Matsubara axis, from IPT solver 

    """
    
    G_loc = Local_Green(S_imp, Energies, dos, Momega)    # Local GF in Matsubaras
    g_0 = G_loc/(1+G_loc*S_imp)                          # Weiss MF from Dyson equation
    g_0_tau = matsubara_ifft(g_0, beta, Momega)  # Weiss MF in imaginary time axis

    new_S_imp = Self_Energy_from_G(g_0_tau,u0,tau,Momega) # Updated self-energy  

    return G_loc, g_0, g_0_tau, new_S_imp



# Dyson equation

def solve_dyson(G_0, sigma):
    
    """
    Solves the Dyson equation given the impurity self-energy and a non-interacting Green's function. 
    
    Yields the impurity Green's function

    Input:
        G_0:   Non-interacting Green's function in Matsubara axis
             
        Sigma: The impurity self-energy
        
    Output:
        G_imp: Impurity Green's function in Matsubara axis 
    """
    
    inverse_G0 = 1/G_0
    G_imp = 1/(inverse_G0 - sigma)
    
    return G_imp



# Self consistent test

def convergence_test(G_imp, G_loc, tol):
    
    """
    Given the newly computed impurity Green's function and the old local Green's
    function, performs the convergence test.
    
    Returns True if convergence has been reached with precision delta. Returns 
    False otherwise.
    
    Input:
        G_imp: New impurity Green's function in Matsubara axis
        G_loc: Old local Green's function in Matsubara axis
        delta: Precision level of the DMFT loop
        
    Output:
        bool: True if convergence has been achieved; false otherwise
    """

    difference = np.absolute(G_imp - G_loc)**2
    D = np.sqrt(np.sum(difference))
    
    return (bool(np.where(D < tol, True, False)), D)



# Spectral function

def pade_analytic_cont(G_loc, Momega, n_freqs):
    
    '''
    Given the local Green's function in the Matsubara axis, computes the analytical
    continuation to the real axis with the Maximum Entropy method.
    
    Input:
        G_loc:  Local Green's function in Matsubara axis
        Momega: Matsubara frequency axis grid
        n_freqs: number of Matsubara frequencies to be used in the Padé approximant
        
    Output:
        Romega: Real frequency axis grid
        spectral: Spectral function in the real axis
    '''
    path = '.\\g_loc'
    with open(path, 'w') as writer:

        for i in range(n_freqs):
            writer.write(str(Momega[Momega>0][i]) + ' ' + str(np.real(G_loc[Momega>0][i])) + ' ' + str(np.imag(G_loc[Momega>0][i])) +'\n')

    import os
    os.system('.\\Pade\\pade.exe -if=g_loc -inener=imag -emin=-4.0 -emax=4.0 -eta=0.01 -npoints=80')

    data = np.loadtxt('.\\g_loc__output')
    Romega = data[:,0]
    spectral = -data[:,3]/np.pi

    os.system('del "g_loc"')
    os.system('del "g_loc__output"')

    return Romega, spectral



# Main DMFT loop

def DMFT_loop(input, DOS_file = None, Sigma_file = None, first_ite = True):

    '''
    Main DMFT loop. Reads an input file and, potentially, a DOS from an
    external file.

    Input:
        input: name of the input file.
        DOS_file: name of the file containing the DOS.
        Sigma_file: name of the file cotaining the inital self-energy.
        first_ite: if True, plots the Green's function in imaginary time after first iteration.

    Output:
        Romega: grid of real frequencies.
        Momega: grid of Matsubara frequencies.
        tau: grid of imaginary time.
        G_loc: local Green's function in Matsubara axis.
        S_imp: self-energy in Matsubara axis.
        G_tau: Green's function in imaginary time axis.
        spectral: Spectral function in real frequency axis.

    '''
    
    D0, u0, nloop, xmix, n_omega, beta, tol, seed = read_input(input)
    energies, DOS = read_DOS(DOS_file, D0)
    Momega, tau, S_imp = initialize_Sigma(n_omega, beta, seed, filename = Sigma_file)

    correction_to_loop = 0

    if first_ite == True:

        correction_to_loop += 1
        G_loc, g_0, g_0_tau, new_S_imp = DMFT_first_iteration(u0, S_imp, energies, DOS, Momega, tau, beta)
        G_imp = solve_dyson(g_0, new_S_imp)
        S_imp = new_S_imp

    converged_flag = False

    for k_loop in range(nloop - correction_to_loop):

        G_loc, g_0, g_0_tau, new_S_imp = DMFT_iteration(u0, S_imp, energies, DOS, Momega, tau, beta)
        G_imp = solve_dyson(g_0, new_S_imp)
        
        if convergence_test(G_imp, G_loc, tol)[0]:
            print(' ')
            print('----------------------------------------')   
            print('Convergence has been reached, iteration '+str(k_loop + correction_to_loop))
            print('----------------------------------------')
            converged_flag = True
            break

        else:
            S_imp = xmix * new_S_imp + (1-xmix) * S_imp
            
            if (k_loop + correction_to_loop)%5 == 0:
                print(f'ITERATION {k_loop + correction_to_loop}')
                print(f'Convergence test: {convergence_test(G_imp, G_loc, tol)[1]}')
                print(' ')

    if not converged_flag:
        print('----------------------------------------')
        print('WARNING: Completed the required number of iterations without reaching convergence')
        print('----------------------------------------')
    
    G_tau  = matsubara_ifft(G_loc, beta, Momega)
    Momega = np.fft.fftshift(Momega)[::-1]
    G_loc  = np.fft.fftshift(G_loc)[::-1]
    S_imp  = np.fft.fftshift(S_imp)[::-1]
    Romega, spectral = pade_analytic_cont(G_loc, Momega, 200)
    
    return Romega, Momega, tau, G_loc, S_imp, G_tau, spectral