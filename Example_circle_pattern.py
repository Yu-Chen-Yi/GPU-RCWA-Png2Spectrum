from RCWA import RCWA
import glob
import os
import tqdm
from sim_setting import *

INPUT_FolderPath = r'dataset\circle\pattern'
DEVICE = DEVICE
WAVELENGTH_RANGE = [WAVELENGTH_MIN, WAVELENGTH_MAX]
WAVELENGTH_STEP = WAVELENGTH_STEP
HARMONIC_ORDER = HARMONIC_ORDER
PERIOD = PERIOD
PATTERN_THK = PATTERN_THK
INPUT_MATERIAL = INPUT_MATERIAL
PATTERN_A_MATERIAL = PATTERN_A_MATERIAL
PATTERN_B_MATERIAL = PATTERN_B_MATERIAL
OUTPUT_MATERIAL = OUTPUT_MATERIAL

if __name__ == "__main__":
    pattern_paths = glob.glob(os.path.join(INPUT_FolderPath, '*.png'))
    for pattern_path in tqdm.tqdm(pattern_paths):
        rcwa_sim = RCWA(
            device=DEVICE,
            geo_dtype=GEO_DTYPE,
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