import torch
import torcwa
import os
import pandas as pd
from PIL import Image
import numpy as np

# Parameters for the pattern generation
X_Period = 500  # Period in x direction (nm)
Y_Period = 500  # Period in y direction (nm)
X_pixel_number = 200  # Number of pixels in x direction
Y_pixel_number = 200  # Number of pixels in y direction

pixel_size = X_Period / X_pixel_number  # Pixel size in nm (assuming square pixels)
radii_nm = np.linspace(50, 150, 10)    # List of radii in nm for the circles

# Directory setup for saving patterns and parameters
shape_dir = r"dataset\circle"
pattern_dir = os.path.join(shape_dir, 'pattern')
os.makedirs(pattern_dir, exist_ok=True)

records = []  # List to store parameters for each pattern
for idx, radius_nm in enumerate(radii_nm, 1):
    # Set the center of the circle
    circle_center_x = X_Period / 2
    circle_center_y = Y_Period / 2
    circle_radius = radius_nm

    # Create geometry object for the pattern
    geometry = torcwa.geometry(
        Lx = X_Period,
        Ly = Y_Period,
        nx = X_pixel_number,
        ny = Y_pixel_number,
        edge_sharpness = 1000.,
        device = torch.device('cpu'),
        dtype = torch.float32
    )
    geometry.grid()
    # Generate a circle pattern (0 inside, 1 outside or vice versa)
    layer_0 = geometry.circle(Cx = circle_center_x, Cy = circle_center_y, R = circle_radius)
    # Convert to 8-bit grayscale image for saving as PNG
    img = Image.fromarray((layer_0.cpu().numpy()*255).astype(np.uint8))
    img = img.convert('L')
    filename = f"{idx:05d}.png"
    save_path = os.path.join(pattern_dir, filename)
    img.save(save_path)
    print(f"Saved: {save_path}")

    # Record parameters for Excel
    records.append({
        "filename": filename,
        "radius (nm)": radius_nm,
        "nx": X_pixel_number,
        "ny": Y_pixel_number,
        "pixel size (nm)": pixel_size
    })

# Save all pattern parameters to an Excel file
excel_path = os.path.join(shape_dir, "pattern_parameters.xlsx")
df = pd.DataFrame(records)
df.to_excel(excel_path, index=False)
print(f"Excel saved: {excel_path}")