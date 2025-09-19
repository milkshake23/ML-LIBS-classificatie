import h5py
from datetime import datetime
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

CHANNEL_LABELS = {
    0: "UV",
    1: "VIS"
}

def main():
    # Open the HDF5 file and read the required datasets
    with h5py.File(r"C:\Users\joost\Documents\GitHub\ML-LIBS-classificatie\data\2024-11-05T11-26-41_nr-001_B1_tread_aided_10Hz_280A.h5") as h5:
        # A 2-D array (time x wavelength) with measured intensities
        intensity: NDArray = h5['intensity'][:] 

        # A 1-D array with the wavelengths corresponding to each pixel
        wavelength: NDArray = h5['wavelength'][:] 

        # A 1-D array containing the channel index belonging to each pixel (0 for UV, 1 for VIS)
        order: NDArray = h5['order'][:]

        # A 2-D array (channel x wavelength) with timestamps for each spectrum
        # UNIX timestamps (microseconds since 1970-01-01 00:00:00Z)
        timestamps: NDArray = h5['timeReceived'][:]

    # Calculate the average spectrum of this file
    mean_intensity: NDArray = np.mean(intensity, axis=0)

    fig, axes = plt.subplots(figsize=(18,9), ncols=2)
    ax_avg, ax_ts = axes

    # Plot the mean spectrum
    for channel_idx, channel_label in CHANNEL_LABELS.items():
        # Create a mask with True for pixels belonging to the current spectrometer channel
        channel_mask: NDArray = order == channel_idx

        # Plot the mean intensity of the current channel
        ax_avg.plot(wavelength[channel_mask], mean_intensity[channel_mask], label=channel_label)

        # Convert the UNIX timestamps to datetime objects
        datetimes: list[datetime] = [
            datetime.fromtimestamp(t/1e6, tz=None) for t in timestamps[channel_idx,:]
        ]

        # Calculate the sum of intensities for each spectrum
        intensity_sums: NDArray = intensity[:,channel_mask].sum(axis=1)

        ax_ts.scatter(datetimes, intensity_sums, label=channel_label)

    ax_avg.set_xlabel("Wavelength (nm)")
    ax_avg.set_ylabel("Intensity (counts)")
    ax_avg.set_title("Mean LIBS spectrum")

    ax_ts.set_xlabel("Time (UTC)")
    ax_ts.set_ylabel("Intensity sum (counts)")
    ax_ts.set_title("Intensity sums over time")

    for ax in axes:
        ax.legend()

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
