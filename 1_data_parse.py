###############################################################################################

# Downloading the data from http://smass.mit.edu/smass.html
# We download two files, the first one: "Bus.Taxonomy.txt", contains the list of all asteroid
# names with the proper classification label. 
# The second file: "smass2data.tar.gz" contains the reflectance spectra of each asteroid.
# We need to create a dataset in which each asteroid is associated with its proper spectra and
# classification label. In this way we will be able to use this dataset to train a classifier.
# The dataset will be stored in folders named after the "level" of feature engeneering that
# will be done to the dataset.

###############################################################################################



import hashlib
import pathlib
import os
import tarfile
import urllib.request
import glob
import re

# Import installed libraries
import pandas as pd



# Define function to compute the sha256 value of the downloaded files, that we
# will use to see if the download of the file was successful 
def comp_sha256(file_name):
    """
    Compute the SHA256 hash of a file.
    Parameters
    ----------
    file_name : str
        Absolute or relative pathname of the file that shall be parsed.
    Returns
    -------
    sha256_res : str
    """
    # Set the SHA256 hashing
    hash_sha256 = hashlib.sha256()

    # Open the file in binary mode (read-only) and parse it in 65,536 byte chunks (in case of
    # large files, the loading will not exceed the usable RAM)
    with pathlib.Path(file_name).open(mode="rb") as f_temp:
        for _seq in iter(lambda: f_temp.read(65536), b""):
            hash_sha256.update(_seq)

    # Digest the SHA256 result
    sha256_res = hash_sha256.hexdigest()

    return sha256_res


# Create a directory for the unprocessed data that is the level 0 data.
# The level indicates the number of feature engeneering that will be done to the data
current_path = pathlib.Path().absolute()
path_to_create = 'data/lvl0'
path = current_path / path_to_create
pathlib.Path(path).mkdir(parents=True, exist_ok=True)
print("Directory",path_to_create,"created")



# Dictionary of paths and hash of the files that we will download
files_to_dl = \
    {'file1': {'url': 'http://smass.mit.edu/data/smass/Bus.Taxonomy.txt',
               'sha256': '0ce970a6972dd7c49d512848b9736d00b621c9d6395a035bd1b4f3780d4b56c6'},
     'file2': {'url': 'http://smass.mit.edu/data/smass/smass2data.tar.gz',
               'sha256': 'dacf575eb1403c08bdfbffcd5dbfe12503a588e09b04ed19cc4572584a57fa97'}}

# Iterate through the dictionary and download the files
for dl_key in files_to_dl:

    # Get the URL, split it at the last "/" to get the name of the file and create the filename
    # as the download path "data/lvl0/" + name of the file 
    # In this way the data will be downloaded in the right place, and well separated
    url = urllib.parse.urlsplit(files_to_dl[dl_key]["url"])
    filename = pathlib.Path(os.path.join(current_path, "data/lvl0/", url.path.split("/")[-1]))

    # Download file if it is not available
    if not filename.is_file():

        print(f"Downloading now: {files_to_dl[dl_key]['url']}")

        # Download file and retrieve the created filepath
        downl_file_path, _ = urllib.request.urlretrieve(url=files_to_dl[dl_key]["url"],
                                                        filename=filename)

        # Compute and compare the hash value
        tax_hash = comp_sha256(downl_file_path)
        assert tax_hash == files_to_dl[dl_key]["sha256"]

# Untar the spectra data
tar = tarfile.open(os.path.join(current_path, "data/lvl0/", "smass2data.tar.gz"), "r:gz")
tar.extractall(os.path.join(current_path, "data/lvl0/"))
tar.close()

# Get a list of the path of all spectra files (consider only the spfit files, whose
# present spline fit values for the spectrum)
spectra_filepaths = sorted(glob.glob(os.path.join(current_path, "data/lvl0/", "smass2/*spfit*")))

# Not all asteroids have been assinged with a proper designation numbers. We need to separate them.
# Designated asteroids are named a* (eg. a000001.spfit.[2]), while non designated are named au*
# (eg. au1995BM2.spfit.[2])
number_of_nondes_ast = 0
for filepath in spectra_filepaths:
    asteroid_name = filepath.split("/")[-1]
    if "au" in asteroid_name:
        number_of_nondes_ast += 1

# Split designated and non designated asteroids
des_file_paths = spectra_filepaths[:-number_of_nondes_ast]
non_file_paths = spectra_filepaths[-number_of_nondes_ast:]

# Convert the path arrays to Pandas dataframes
des_file_paths_df = pd.DataFrame(des_file_paths, columns=["FilePath"])
non_file_paths_df = pd.DataFrame(non_file_paths, columns=["FilePath"])

# Add now the designation / "non-designation" number as a new column labelled "DesNr"
# The designation / "non-designation" number is taken by the name of the files
des_file_paths_df.loc[:, "DesNr"] = \
    des_file_paths_df["FilePath"].apply(lambda x: int(re.search(r'smass2/a(.*).spfit',x).group(1)))
non_file_paths_df.loc[:, "DesNr"] = \
    non_file_paths_df["FilePath"].apply(lambda x: re.search(r'smass2/au(.*).spfit',x).group(1))

# Read the classification file "Bus.Taxonomy.txt". Since Pandas recognizes more than 3 columns,
# we read and than delete them afterwards
asteroid_class_df = pd.read_csv(os.path.join(current_path, "data/lvl0/", "Bus.Taxonomy.txt"),
                                skiprows=21,
                                sep="\t",
                                names=["Name","Tholen_Class","Bus_Class","unknown1","unknown2"])

# Remove white spaces
asteroid_class_df.loc[:, "Name"] = asteroid_class_df["Name"].apply(lambda x: x.strip()).copy()

# Separate between designated and non-designated asteroid classes. Unfortunately the index number
# 1403 must be retrieved by reading the documentation and analizing by eye the "Bus.Taxonomy.txt"
# file
des_ast_class_df = asteroid_class_df[:1403].copy()
non_ast_class_df = asteroid_class_df[1403:].copy()

# Now split the designated names and get the designation number (The number before the name)
# eg. designation name: "1 Ceres", designation number: "1"
# Than add the designation number as an additional column
des_ast_class_df.loc[:, "DesNr"] = des_ast_class_df["Name"].apply(lambda x: int(x.split(" ")[0]))

# Merge with the spectra file paths
des_ast_class_join_df = des_ast_class_df.merge(des_file_paths_df, on="DesNr")

# For the non designated names we do the same thing, but using as designation number the whole
# name, removing the whitespace
non_ast_class_df.loc[:, "DesNr"] = non_ast_class_df["Name"].apply(lambda x: x.replace(" ", ""))

# Merge with the spectra file paths
non_ast_class_join_df = non_ast_class_df.merge(non_file_paths_df, on="DesNr")

# Merge now both datasets
asteroids_df = pd.concat([des_ast_class_join_df, non_ast_class_join_df], axis=0)

# Reset the index
asteroids_df.reset_index(drop=True, inplace=True)

# Remove the tholen class and both unknown columns. We will use only the SMASSII (or also called Bus)
# classification because it is the more complete
asteroids_df.drop(columns=["Tholen_Class", "unknown1", "unknown2"], inplace=True)

# Drop now all rows that do not contains a Bus classification (we want to use only the ones 
# that are classified)
asteroids_df.dropna(subset=["Bus_Class"], inplace=True)

# Read the spectra of each asteroid and store it in the data frame.
# The spectrum is composed by two columns, the first one contains the wavelenght in microns,
# while the second one contains the reflectance associated with that specific wavelenght.
# The reflectance is normalized at 550nm
asteroids_df.loc[:, "SpectrumDF"] = \
    asteroids_df["FilePath"].apply(lambda x: pd.read_csv(x, sep="\t",names=\
        ["Wavelength_in_micron","Reflectance_norm550nm"]))

# Convert the Designation Number to string
asteroids_df.loc[:, "DesNr"] = asteroids_df["DesNr"].astype(str)

# Create a dictionary that maps the Bus Classification with the 4 categories of the
#  Main Group: C,S,X and Other
bus_to_main_dict = {
                    'A': 'Other',
                    'B': 'C',
                    'C': 'C',
                    'Cb': 'C',
                    'Cg': 'C',
                    'Cgh': 'C',
                    'Ch': 'C',
                    'D': 'Other',
                    'K': 'Other',
                    'L': 'Other',
                    'Ld': 'Other',
                    'O': 'Other',
                    'R': 'Other',
                    'S': 'S',
                    'Sa': 'S',
                    'Sk': 'S',
                    'Sl': 'S',
                    'Sq': 'S',
                    'Sr': 'S',
                    'T': 'Other',
                    'V': 'Other',
                    'X': 'X',
                    'Xc': 'X',
                    'Xe': 'X',
                    'Xk': 'X'
                   }

# Add a new column for the main group classification using the dictionary
asteroids_df.insert(1,"Main_Group",asteroids_df["Bus_Class"].apply(lambda x: bus_to_main_dict.get(x, "None")))

# Remove the file path and Designation Number (we won't use them anymore)
asteroids_df.drop(columns=["DesNr", "FilePath"], inplace=True)

# Create (if applicable) the level 1 directory
pathlib.Path(os.path.join(current_path, "data/lvl1")).mkdir(parents=True, exist_ok=True)
print("Directory data/lvl1 created")

# Save the dataframe as a pickle file
asteroids_df.to_pickle(os.path.join(current_path, "data/lvl1/", "asteroids.pkl"), protocol=4)

print(asteroids_df)