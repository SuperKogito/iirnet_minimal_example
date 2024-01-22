import time 
import torch
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from scipy.signal import resample
from iirnet.designer import Designer
from scipy.interpolate import interp1d


fs = 48000
fmin_in_hz = 24
fmax_in_hz = 16000

target_eq_dB = np.loadtxt("target_eq_dB.out", delimiter=',')
eq_IIR = np.loadtxt("eq_IIR.out", delimiter=',')
f_axis = np.loadtxt("f_axis.out", delimiter=',')

iir_fc = np.array([ 135.09213, 139.0246, 505.97818, 755.47754, 1194.6179, 1738.2924 ])
iir_Q = np.array([3.1858082, 1.77384 , 1.4948567, 1.5824703, 2.9184678, 2.9487677])
iir_G = np.array([-1.4194046 , 0.66702837,-2.898815  , 2.5335517 ,-0.56966037,-0.11178557])
iir_coeffs_a = np.array([[ 1.        , 1.99368346,-0.99399513],
                         [ 1.        , 1.98984671,-0.99017632],
                         [ 1.        , 1.94474542,-0.94901884],
                         [ 1.        , 1.93797731,-0.94749224],
                         [ 1.        , 1.92258906,-0.94633752],
                         [ 1.        , 1.87621808,-0.92585939]])

iir_coeffs_b = np.array([[ 0.9995474 ,-1.9936835 , 0.9944478 ],
                         [ 1.0003921 ,-1.9898467 , 0.9897842 ],
                         [ 0.99276674,-1.9447454 , 0.956252  ],
                         [ 1.0088918 ,-1.9379773 , 0.9386006 ],
                         [ 0.99829686,-1.9225891 , 0.9480408 ],
                         [ 0.9995259 ,-1.8762181 , 0.9263334 ]])

def normalize(x):
    x_min = np.min(x)
    x_max = np.max(x)
    normalized_x = (x - x_min) / (x_max - x_min)
    return normalized_x


# first load IIRNet with pre-trained weights
designer = Designer()
nls = 5


# Define the subplots
fig, axs = plt.subplots(len([4, 8, 16, 32, 64]), 1, figsize=(10, 5*len([4, 8, 16, 32, 64])))

# Use IIRNet to compute IIR filter
for i, n in enumerate([4, 8, 16, 32, 64]):
    t0 = time.time()
    m = target_eq_dB  # Magnitude response specification
    mode = "linear"  # interpolation mode for specification
    output = "sos"   # Output type ("sos", or "ba")

    # now call the designer with parameters
    sos = designer(n, m, mode=mode, output=output)

    # measure and plot the response
    w, h = scipy.signal.sosfreqz(sos.numpy(), fs=float(fs))

    # interpolate the target for plotting
    m_int = torch.tensor(m).view(1, 1, -1).float()
    m_int = torch.nn.functional.interpolate(m_int, target_eq_dB.shape[0], mode=mode)

    # Upsample the array
    # upsample_factor = target_eq_dB.shape[0] / len(h)
    # upsampled_array = resample(h, int(len(h) * upsample_factor))

    # Target x-coordinate values after extrapolation
    x = np.arange(len(h))
    target_x = np.linspace(0, len(h) - 1, target_eq_dB.shape[0])

    # Linear interpolation
    interpolator = interp1d(x, h, kind='linear')
    extrapolated_array = interpolator(target_x)

    axs[i].semilogx(normalize(f_axis[0, :]), target_eq_dB)
    axs[i].semilogx(normalize(f_axis[0, :]), eq_IIR)
    axs[i].semilogx(normalize(f_axis[0, :]), 20 * np.log10(np.abs(extrapolated_array)))
    axs[i].legend(("Target", "IIR-optimizer", "IIR-IIRNet"))
    axs[i].set_xlabel('$f$ in Hz')
    axs[i].set_ylabel(r'$|H(f)|$')
    axs[i].grid(True)
    axs[i].set_xlim([0, 1])
    axs[i].set_ylim([-5., 5.])
    axs[i].set_title("Idx: " + str(nls) + "| IIR order: " + str(iir_fc.shape[0]) + "| IIRnet order: " + str(n))

    print("IIRNET Runtime: " + str((time.time() - t0)*1000))
    print("+"*30)

plt.tight_layout()
plt.show()
