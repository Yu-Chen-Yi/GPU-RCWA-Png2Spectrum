# GPU Accelerated RCWA (Rigorous Coupled Wave Analysis)

This project implements a GPU-accelerated Rigorous Coupled Wave Analysis (RCWA) simulation for optical nanostructures. It provides tools for generating pattern images and computing optical spectra with support for various polarization states.

## Features

- **GPU Acceleration**: Leverages CUDA for faster computation when available
- **Pattern Generation**: Automated generation of nanostructure patterns from parameters
- **Batch Processing**: Process multiple patterns simultaneously
- **Multi-polarization Support**: Computes spectra for various polarization states:
  - X-polarized input → X/Y polarized output
  - Y-polarized input → X/Y polarized output  
  - RCP (Right Circular Polarized) input → RCP/LCP output
  - LCP (Left Circular Polarized) input → RCP/LCP output
- **Material Database**: Extensible material properties system
- **Flexible Layer Structure**: Support for multi-layer structures

## Project Structure

```
Gpu acceleration RCWA/
├── RCWA.py                    # Main RCWA simulation engine
├── Materials.py               # Material properties handling
├── Pattern_img_generator.py   # Pattern image generation
├── Example_circle_pattern.py  # Example: circular pattern simulation
├── Materials_data/            # Material property files
│   ├── air.txt
│   ├── aSiH.txt
│   ├── Fused_silica.txt
│   └── Si (Silicon) - Palik.txt
└── dataset/                   # Generated patterns and spectra
    └── circle/
        ├── pattern/           # Generated PNG patterns
        └── spectrum/          # Computed spectra (CSV files)
```

## Installation

### Prerequisites

- Python 3.7+
- PyTorch (with CUDA support recommended)
- NumPy
- OpenCV (cv2)
- Pandas
- SciPy
- tqdm
- torcwa (RCWA library)

### Setup

1. Clone the repository
2. Install required packages:
```bash
pip install torch numpy opencv-python pandas scipy tqdm
```
3. Install torcwa library (follow their installation instructions)
4. Place material data files in `Materials_data/` directory

## Usage

### 1. Pattern Generation

The `Pattern_img_generator.py` generates nanostructure patterns from parameters. Users can:

- Adjust image resolution and pixel size
- Modify structure parameters (e.g., circle radius)
- Add custom Boolean conditions for different structures
- Generate PNG files for RCWA simulation

### 2. RCWA Simulation

The `Example_circle_pattern.py` demonstrates batch processing:

```python
from RCWA import RCWA
import torch
import glob
import os
import tqdm

INPUT_FolderPath = r'dataset\circle\pattern'
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
WAVELENGTH_RANGE = [400., 1000.]
WAVELENGTH_STEP = 2.
HARMONIC_ORDER = 7
PERIOD = 400.
PATTERN_THK = 100.
INPUT_MATERIAL = 'Fused_silica.txt'
PATTERN_A_MATERIAL = 'aSiH.txt'
PATTERN_B_MATERIAL = 'air.txt'
OUTPUT_MATERIAL = 'air.txt'

if __name__ == "__main__":
    pattern_paths = glob.glob(os.path.join(INPUT_FolderPath, '*.png'))
    for pattern_path in tqdm.tqdm(pattern_paths):
        rcwa_sim = RCWA(
            device=DEVICE,
            geo_dtype=torch.float32,
            wavelength_range=WAVELENGTH_RANGE,
            wavelength_step=WAVELENGTH_STEP,
            pattern_path=pattern_path,
            harmonic_order=HARMONIC_ORDER,
            inc_ang=0.,
            azi_ang=0.,
            period=PERIOD,
            pattern_thk = PATTERN_THK,
            input_material=INPUT_MATERIAL,
            pattern_A_material=PATTERN_A_MATERIAL,
            pattern_B_material=PATTERN_B_MATERIAL,
            output_material=OUTPUT_MATERIAL,
        )
        rcwa_sim.get_Spectrum()
```

### 3. Output

Spectra are automatically saved to `dataset/circle/spectrum/` as CSV files containing:
- Wavelength data
- Transmission coefficients (Trans_xx, Trans_xy, Trans_yx, Trans_yy, Trans_RL, Trans_RR, Trans_LR, Trans_LL)
- Reflection coefficients (Refl_xx, Refl_xy, Refl_yx, Refl_yy, Refl_RL, Refl_RR, Refl_LR, Refl_LL)

## Configuration

### RCWA Parameters

- `device`: Computation device ('cuda' or 'cpu')
- `wavelength_range`: [min_wavelength, max_wavelength] in nm
- `wavelength_step`: Wavelength resolution in nm
- `harmonic_order`: Number of diffraction orders
- `period`: Structure period in nm
- `pattern_thk`: Pattern layer thickness in nm
- `inc_ang`: Incident angle in degrees
- `azi_ang`: Azimuthal angle in degrees

### Materials

Materials are defined in text files with format:
```
Wavelength(nm)  Refractive index  Extinction coefficient
573	            3.355274	      0.046603
574	            3.352986          0.046115
575	            3.350715	      0.045636
576	            3.348461	      0.045163
577	            3.346225	      0.044699
578	            3.344005	      0.044241
579	            3.341802	      0.043791
580	            3.339616	      0.043347
581	            3.337446	      0.042911
582	            3.335292	      0.042481
583	            3.333154	      0.042057
```

To add new materials:
1. Create a new `.txt` file in `Materials_data/`
2. Follow the format above
3. Reference the material name in RCWA initialization

### Multi-layer Structures

To add additional layers in `RCWA.py`:

```python
# Add custom layer
sim.add_layer(thickness=layer_thickness, eps=layer_permittivity)
```

Where:
- `thickness`: Layer thickness in nm
- `eps`: Layer permittivity distribution (tensor)

## Example: Circular Pattern

The included example demonstrates circular nanostructure simulation:

1. Generate circular patterns with `Pattern_img_generator.py`
2. Process patterns with `Example_circle_pattern.py`
3. Results saved in `dataset/circle/spectrum/`

## Performance

- **GPU Mode**: Significantly faster computation using CUDA
- **CPU Mode**: Compatible fallback for systems without GPU
- **Batch Processing**: Efficient handling of multiple patterns
- **Memory Optimization**: Configurable data types (float32/float64)

## Troubleshooting

### Common Issues

1. **CUDA not available**: Falls back to CPU automatically
2. **Material file not found**: Ensure material files are in `Materials_data/`
3. **Pattern size mismatch**: Ensure nx = ny for square patterns
4. **Memory issues**: Reduce harmonic_order or use float32 dtype

### Error Messages

- "CUDA is not available": Using CPU computation
- "nx and ny must be equal": Pattern must be square
- "Material file not found": Check Materials_data directory

## Contributing

To extend the project:

1. **New Structures**: Modify `Pattern_img_generator.py`
2. **New Materials**: Add material files to `Materials_data/`
3. **New Layers**: Modify `RCWA.py` simulation setup
4. **New Features**: Extend RCWA class methods