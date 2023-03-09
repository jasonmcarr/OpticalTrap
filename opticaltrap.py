import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

therm = pd.read_csv('placeholder-for-actual-csv')
therm = therm.loc[:, 'x':'y']
therm = therm.values

cal_x = 0.08620  # microns/pixel
cal_y = 0.0819
sigma_calx = 0.05/10000*cal_x
sigma_caly = 0.05/10000*cal_y

therm_x = therm[:, 0]*cal_x
therm_y = therm[:, 1]*cal_y
therm_x_avg = therm_x - np.mean(therm_x)
therm_y_avg = therm_y - np.mean(therm_y)

# take ffts
ftherm_x = np.fft.rfft(therm_x_avg)
# are we sure the framerate was 500 Hz?
nutherm_x = np.fft.rfftfreq(len(therm_x_avg), 1./5000.)
ftherm_y = np.fft.rfft(therm_y_avg)
nutherm_y = np.fft.rfftfreq(len(therm_y_avg), 1./5000.)

# plot trajectory
plt.figure(figsize=(8, 6))
plt.plot(therm_x, therm_y)
plt.xlabel("x ($\mathrm{\mu m}$)")
plt.ylabel("y ($\mathrm{\mu m}$)")
plt.title("Brownian motion of particle in optical trap")
plt.show()

# plot ffts in x and y
fig0, ax0 = plt.subplots(1, 2, figsize=(16, 6))
fig0.suptitle("Fourier domain Brownian motion", fontsize=16, fontweight='bold')
ax0[0].plot(nutherm_x, np.absolute(ftherm_x)**2)
ax0[0].set_xlabel("Frequency (Hz)")
ax0[0].set_ylabel("Power spectrum in x")
ax0[0].set_title("Motion in x")
ax0[0].set_yscale('log')
ax0[0].set_ylim(1e0)
ax0[0].set_xlim(0, 120)

ax0[1].plot(nutherm_y, np.absolute(ftherm_y)**2)
ax0[1].set_xlabel("Frequency (Hz)")
ax0[1].set_ylabel("Power spectrum in y")
ax0[1].set_title("Motion in y")
ax0[1].set_yscale('log')
ax0[1].set_ylim(1e0)
ax0[1].set_xlim(0, 120)
plt.show()

# plot histograms
fig1, ax1 = plt.subplots(1, 2, figsize=(16, 6))
fig1.suptitle("Histograms for Brownian motion", fontsize=16, fontweight='bold')
ax1[0].hist(therm_x_avg**2, bins=30)
ax1[0].set_title("Motion in x")
ax1[0].set_yscale("log")
ax1[0].set_xlabel("$|x-x_0|^2$ $\mathrm{\mu m}^2$")
ax1[0].set_ylabel("Occurences")

ax1[1].hist(therm_y_avg**2, bins=30)
ax1[1].set_title("Motion in y")
ax1[1].set_yscale("log")
ax1[1].set_xlabel("$|y-y_0|^2$ $\mathrm{\mu m}^2$")
ax1[1].set_ylabel("Occurences")
plt.show()

# measure spring constant
kB = 1.38e-23  # m^2 kg s^-2 K^-1
T = 298  # K


def function_to_fit(x, a, b):
    return (a*np.exp(-b*x))


therm_x_avg = therm_x_avg[np.where(np.abs(therm_x_avg)**2 < 0.18)]
hist_therm_x = np.histogram((np.abs(therm_x_avg))**2, bins=30)
# print(hist_therm_x[0][0:30])
popt_therm_x, pcov_therm_x = curve_fit(function_to_fit, hist_therm_x[1][0:30],
                                       hist_therm_x[0][0:30],
                                       sigma=np.sqrt(hist_therm_x[0][0:30]))
print(popt_therm_x)
k_x = popt_therm_x[1]*2*kB*T
print("k_x =", k_x, "\n")

therm_y_avg = therm_y_avg[np.where(np.abs(therm_y_avg)**2 < 0.12)]
hist_therm_y = np.histogram((np.abs(therm_y_avg))**2, bins=30)
# print(hist_therm_y[0][0:30])
popt_therm_y, pcov_therm_y = curve_fit(function_to_fit, hist_therm_y[1][0:30],
                                       hist_therm_y[0][0:30],
                                       sigma=np.sqrt(hist_therm_y[0][0:30]))

print(popt_therm_y)
k_y = popt_therm_y[1]*2*kB*T
print("k_y =", k_y)
