import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from gui import label_cross_section, compute_distance_to_river_center, update_and_save_to_csv

# Load the existing output CSV
name = "BLACK"
df = pd.read_csv(f"data/{name}_output.csv")

# Function to plot a cross section for review
def plot_cross_section(df, idx):
    # Extract the cross section data
    cross_section = df.iloc[idx]
    
    # Plot the cross section
    plt.figure(figsize=(15, 8))
    plt.plot(cross_section['dist_along'], cross_section['elevation'], '-o', markersize=2, color='blue')
    plt.xlabel('Along Track Distance')
    plt.ylabel('Elevation')
    
    # Plot the labeled points
    labels = ['channel', 'ridge1', 'floodplain1', 'ridge2', 'floodplain2']
    colors = {'channel': 'b', 'ridge1': 'g', 'ridge2': 'g', 'floodplain1': 'r', 'floodplain2': 'r'}
    for label in labels:
        if not np.isnan(cross_section[f"{label}_dist_along"]):
            plt.plot(cross_section[f"{label}_dist_along"], cross_section[f"{label}_elevation"], 'X', markersize=10, color=colors[label])

    plt.show()

# Function to handle user input for accepting, rejecting, or repicking a cross section
def handle_user_input(df, idx):
    # Plot the cross section
    plot_cross_section(df, idx)
    
    # Get user input
    action = input("Enter 'a' to accept, 'r' to reject, or 'p' to repick: ")
    
    if action == 'a':
        # Accept the cross section
        print(f"Cross section {idx} accepted.")
    elif action == 'r':
        # Reject the cross section
        print(f"Cross section {idx} rejected.")
        df.drop(idx, inplace=True)
    elif action == 'p':
        # Repick the cross section
        print(f"Cross section {idx} repicked.")
        df, points = label_cross_section(df)
        df = compute_distance_to_river_center(df, points)
        update_and_save_to_csv(df, points, filename = f"data/{name}_output.csv")

    return df

# Loop through the cross sections
for idx in range(len(df)):
    df = handle_user_input(df, idx)

# Save the modified DataFrame
df.to_csv(f"data/{name}_output_edited.csv", index=False)