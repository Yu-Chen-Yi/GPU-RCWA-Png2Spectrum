import numpy as np
from PIL import Image
import os
import pandas as pd

# Function to generate a circle image
# nx, ny: image width and height in pixels
# pixel_size: size of one pixel in nm
# radius_nm: radius of the circle in nm
# save_path: where to save the png image

def generate_circle_image(nx, ny, pixel_size, radius_nm, save_path):
    # Create a black image
    img = np.zeros((ny, nx), dtype=np.uint8)
    # Calculate the center of the image
    cx, cy = nx // 2, ny // 2
    # Convert radius from nm to pixels
    radius_px = radius_nm / pixel_size
    # Create a grid of coordinates
    y, x = np.ogrid[:ny, :nx]
    # Calculate distance from center
    dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    # Set pixels inside the circle to white (255)
    img[dist_from_center <= radius_px] = 255
    # Save the image as PNG
    Image.fromarray(img).save(save_path)

# Example usage: loop over different radii
if __name__ == "__main__":
    nx = 200  # image width in pixels
    ny = 200  # image height in pixels
    pixel_size = 2  # pixel size in nm
    radii_nm = np.linspace(50, 150, 10)  # list of radii in nm
    shape_dir = r"dataset\circle"
    pattern_dir = os.path.join(shape_dir, 'pattern')
    spectrum_dir = os.path.join(shape_dir, 'spectrum')
    os.makedirs(shape_dir, exist_ok=True)
    os.makedirs(pattern_dir, exist_ok=True)
    os.makedirs(spectrum_dir, exist_ok=True)
    records = []
    for idx, radius_nm in enumerate(radii_nm, 1):
        filename = f"{idx:05d}.png"
        save_path = os.path.join(pattern_dir, filename)
        radius_nm = radius_nm//pixel_size * pixel_size
        generate_circle_image(nx, ny, pixel_size, radius_nm, save_path)
        print(f"Saved: {save_path}")
        # Record parameters for Excel
        records.append({
            "filename": filename,
            "radius (nm)": radius_nm,
            "nx": nx,
            "ny": ny,
            "pixel size (nm)": pixel_size
        })

    # Save parameters to Excel
    df = pd.DataFrame(records)
    excel_path = os.path.join(shape_dir, "pattern_parameters.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"Excel saved: {excel_path}")
