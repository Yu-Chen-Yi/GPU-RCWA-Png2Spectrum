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