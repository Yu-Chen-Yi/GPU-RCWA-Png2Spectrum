import torch

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
GEO_DTYPE = torch.float32
WAVELENGTH_MIN = 405.
WAVELENGTH_MAX = 900.
WAVELENGTH_STEP = 5.
HARMONIC_ORDER = 7
PERIOD = 400.
PATTERN_THK = 100.
INPUT_MATERIAL = 'Fused_silica.txt'
PATTERN_A_MATERIAL = 'aSiH.txt'
PATTERN_B_MATERIAL = 'air.txt'
OUTPUT_MATERIAL = 'air.txt'