import os
import shutil

# Define the source and destination directories
source_dir = r"E:/EALPR/x/Characters"
destination_dir = r"C:/Users/MOSTAFA/Desktop/data/feh"

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Iterate through all files in the source directory
for file_name in os.listdir(source_dir):
    print("Processing file:", file_name)
    if file_name.endswith(('.jpg', '.png', '.jpeg', '.bmp')):  # Add other extensions if needed
        # Split the file name by '.'
        name_parts = file_name.split('.')
        if len(name_parts) > 1 and name_parts[0][-3]=='ف':
            # Move the file to the 'non' folder
            print("Moving file to destination directory")
            shutil.move(os.path.join(source_dir, file_name), destination_dir)

print("Files moved successfully!")
