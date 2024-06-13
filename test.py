import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt

# Define the path to the text file containing the CSV file paths and column counts
txt_file_path = 'miss_ecg.txt'
download_folder = 'download'  # Define the download folder

# Create the download folder if it does not exist
os.makedirs(download_folder, exist_ok=True)

# Read the file paths and column counts from the text file
files_columns = []
with open(txt_file_path, 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        file_path = parts[0]
        column_count = int(parts[1])
        files_columns.append((file_path, column_count))

        # Copy file to the download folder
        shutil.copy(file_path, download_folder)

# Check the copied files
copied_files = os.listdir(download_folder)
print("Files copied to 'download' folder:")
print(copied_files)