import os
from PIL import Image

# Define the dictionary for mapping
label_map = {"Lamp": 0, "Table": 1, "Chair": 2, "Couch": 3, "Bed": 4}

def get_image_dimensions(image_path):
    """Get the dimensions of the image."""
    with Image.open(image_path) as img:
        return img.width, img.height

# Function to normalize coordinates
def normalize_coordinates(x1, y1, x2, y2, img_width, img_height):
    x1_norm = x1 / img_width
    y1_norm = y1 / img_height
    x2_norm = x2 / img_width
    y2_norm = y2 / img_height
    return x1_norm, y1_norm, x2_norm, y2_norm

# Function to transform a single line of data
def transform_line(line, img_width, img_height):
    parts = line.strip().split()
    
    if len(parts) != 5:
        # Line does not have the correct number of parts, skip it
        return None
    
    label = parts[0]
    if label not in label_map:
        # Label is not in the dictionary, skip it
        return None
    
    try:
        x1 = float(parts[1])
        y1 = float(parts[2])
        x2 = float(parts[3])
        y2 = float(parts[4])
    except ValueError:
        # One of the parts is not a number, skip it
        return None
    
    # Normalize coordinates
    x1_norm, y1_norm, x2_norm, y2_norm = normalize_coordinates(x1, y1, x2, y2, img_width, img_height)
    
    # Calculate center_point_x, center_point_y, width, and height
    center_point_x = (x1_norm + x2_norm) / 2
    center_point_y = (y1_norm + y2_norm) / 2
    width = abs(x2_norm - x1_norm)
    height = abs(y2_norm - y1_norm)
    
    return f"{label_map[label]} {center_point_x} {center_point_y} {width} {height}"

# Function to process all files in a directory
def process_files(txt_directory, img_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for filename in os.listdir(txt_directory):
        if filename.endswith(".txt"):
            txt_file_path = os.path.join(txt_directory, filename)
            img_file_path = os.path.join(img_directory, os.path.splitext(filename)[0] + '.jpg')
            
            if not os.path.exists(img_file_path):
                print(f"Image file {img_file_path} does not exist, skipping.")
                continue

            img_width, img_height = get_image_dimensions(img_file_path)
            
            output_file_path = os.path.join(output_directory, filename)
            
            with open(txt_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
                for line in infile:
                    transformed_line = transform_line(line, img_width, img_height)
                    if transformed_line:
                        outfile.write(transformed_line + '\n')

# Define the input and output directories
txt_directory = '/Users/tirthpatel/Desktop/Python/data/labels/val'
img_directory = '/Users/tirthpatel/Desktop/Python/data/images/val'
output_directory = '/Users/tirthpatel/Desktop/Python/data/labels/val_output'

# Process the files
# process_files(txt_directory, img_directory, output_directory)
