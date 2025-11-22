import argparse
import numpy as np
import pandas as pd
import os

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--nevents",
        default=100000,
        help="Number of total events to generate",
        type=int,
    )
    parser.add_argument(
        "--output_directory",
        default = "./",
        help = "Directory where the output Pandas dataframes will be stored",
        type=str,
    )
    flags = parser.parse_args()

    return flags

# Function used to randomly generate theta-phi data
def generate_theta_phi(num_events):

    if num_events <= 0:
        raise ValueError("Number of events must be > 0!")
    
    theta_phi_df = pd.DataFrame(np.random.rand(num_events, 2), columns=['theta', 'phi'])

    return theta_phi_df

# Generates a 3D array of size resolution x resolution x resolution, and randomly places a number in it
def generate_random_object(resolution=64):
    return np.random.rand(resolution, resolution, resolution)

def main():
    flags = parse_arguments()

    # Making output directory if it doesn't already exist
    if not os.path.exists(flags.output_directory):
        os.makedirs(flags.output_directory)
    
    # Generating data with and without the object and saving it to .csv files
    data_without_object = generate_theta_phi(
        num_events = flags.nevents,
    )
    data_without_object.to_csv(os.path.join(flags.output_directory, "theta_phi_without_object.csv"), index=False)

    data_with_object = generate_theta_phi(
        num_events = flags.nevents,
    )
    data_with_object.to_csv(os.path.join(flags.output_directory, "theta_phi_with_object.csv"), index=False)

    random_object = generate_random_object()
    np.savez(os.path.join(flags.output_directory, "object_densities.npz"), random_object)


if __name__ == "__main__":
    main()