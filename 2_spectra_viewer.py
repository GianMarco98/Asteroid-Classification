###############################################################################################

# Plot the spectra of the asteroids to see the proprieties of different classifications

###############################################################################################



# Import standard libraries
import os
import pathlib

# Import installed libraries
from matplotlib import pyplot as plt
import pandas as pd



# Read the level 1 dataframe
current_path = pathlib.Path().absolute()
asteroids_df = pd.read_pickle(os.path.join(current_path, "data/lvl1/", "asteroids.pkl"))

# Set the dark mode and the font size and style
plt.style.use('dark_background')
plt.rc('font', family='serif', size=10)

# Plot a figure with four plots that will display the spectra for each classification label
fig, axs = plt.subplots(2, 2)

# Set dimentions
fig.set_size_inches(18.5, 10.5)

# Iterate trough all the classification labels and plot in the corresponding axes
for sub_class,axes in zip(asteroids_df.loc[:,"Main_Group"].unique(),axs.flat):

    # Set title
    axes.set_title(sub_class+" spectra")

    # Set labels
    axes.set(xlabel="Wavelength [μm]", ylabel="Reflectance normalized at 0.55 μm")

    # Set a fixed y limit range
    axes.set_ylim(0.5, 1.5)

    # Set a grid
    axes.grid(linestyle="dashed", alpha=0.3)

    # Take only the selected asteroids
    asteroids_filtered_df = asteroids_df.loc[asteroids_df["Main_Group"]==sub_class]
    
    # Iterate trough the spectra and plot them
    for _, row in asteroids_filtered_df.iterrows():

        # Set x limit
        axes.set_xlim(min(row["SpectrumDF"]["Wavelength_in_micron"]),
                max(row["SpectrumDF"]["Wavelength_in_micron"]))

        # Plot the spectra
        axes.plot(row["SpectrumDF"]["Wavelength_in_micron"],
                    row["SpectrumDF"]["Reflectance_norm550nm"],
                    alpha=0.1,
                    color='#ccebc4')

# Save as a pdf
pathlib.Path(current_path / "plots").mkdir(parents=True, exist_ok=True)
plt.savefig(str(current_path) + "/plots/spectra_plot.pdf")
print("plots/spectra_plot.pdf has been created")