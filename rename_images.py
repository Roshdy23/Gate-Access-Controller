import os

# Define the directory containing the images
directory = r'C:\Users\MOSTAFA\Desktop\data\yeh'

# Specify the desired file extension for renamed images
desired_extension = '.png'  # Change to '.jpg', '.jpeg', etc., if needed

# Ensure the directory exists
if not os.path.exists(directory):
    print(f"Directory '{directory}' does not exist.")
    exit()

# Get all files in the directory
files = os.listdir(directory)

# Filter only image files (based on extensions)
image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
images = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]

# Rename the images
for i, image in enumerate(images, start=1):
    old_path = os.path.join(directory, image)
    new_name = f"image_{i}{desired_extension}"
    new_path = os.path.join(directory, new_name)
    try:
        os.rename(old_path, new_path)
        print(f"Renamed: '{image}' -> '{new_name}'")
    except Exception as e:
        print(f"Failed to rename '{image}': {e}")

print("Renaming completed.")
