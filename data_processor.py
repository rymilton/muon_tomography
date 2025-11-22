# data_processor.py
import numpy as np
import pandas as pd
import argparse
import os


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_directory",
        default="./",
        type=str,
        help="Directory containing CSV files",
    )
    parser.add_argument(
        "--output_directory",
        default="./",
        type=str,
        help="Directory to save histogram npz",
    )
    parser.add_argument(
        "--nbins_theta", default=64, type=int, help="Number of bins for theta"
    )
    parser.add_argument(
        "--nbins_phi", default=64, type=int, help="Number of bins for phi"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Ensure output directory exists
    os.makedirs(args.output_directory, exist_ok=True)

    # Load theta-phi CSV files
    without_object = pd.read_csv(
        os.path.join(args.input_directory, "theta_phi_without_object.csv")
    )
    with_object = pd.read_csv(
        os.path.join(args.input_directory, "theta_phi_with_object.csv")
    )

    # Define bin edges
    theta_edges = np.linspace(0, 1, args.nbins_theta + 1)  # theta normalized 0-1
    phi_edges = np.linspace(0, 1, args.nbins_phi + 1)  # phi normalized 0-1

    # Compute 2D histograms
    hist_without, _, _ = np.histogram2d(
        without_object["theta"], without_object["phi"], bins=[theta_edges, phi_edges]
    )

    hist_with, _, _ = np.histogram2d(
        with_object["theta"], with_object["phi"], bins=[theta_edges, phi_edges]
    )

    # Compute transmission fraction
    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        transmission = np.true_divide(hist_with, hist_without)
        transmission[~np.isfinite(transmission)] = 0  # set NaN and inf to 0

    # Save the transmission histogram
    output_path = os.path.join(args.output_directory, "theta_phi_transmission.npz")
    np.savez(
        output_path,
        transmission=transmission,
        theta_edges=theta_edges,
        phi_edges=phi_edges,
    )

    print(f"Saved transmission histogram to {output_path}")


if __name__ == "__main__":
    main()
