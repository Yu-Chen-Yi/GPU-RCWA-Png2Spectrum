import torch
import numpy as np  
import torcwa
import Materials
import cv2
import os
import pandas as pd

class RCWA:
    def __init__(
        self,
        device= torch.device('cpu'),
        geo_dtype = torch.float32,
        wavelength_range = [400., 1000.],
        wavelength_step = 5.,
        pattern_path='png_path',
        harmonic_order=7,
        wavelength=940.,
        inc_ang=0.,
        azi_ang=0.,
        period=400.,
        pattern_thk = 100.,
        input_material='silica.txt',
        pattern_A_material='aSiH.txt',
        pattern_B_material='air.txt',
        output_material='air.txt',
    ):
        if device == torch.device('cuda'):
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available")
            else:
                print("CUDA is available")
        else:
            print("Using CPU for simulation")
        self.device = device
        if geo_dtype == torch.float32:
            self.geo_dtype = torch.float32
            self.sim_dtype = torch.complex64
        elif geo_dtype == torch.float64:
            self.geo_dtype = torch.float64
            self.sim_dtype = torch.complex128
        else:
            raise ValueError("geo_dtype must be torch.float32 or torch.float64")
        self.pattern_path = pattern_path
        self.harmonic_order = harmonic_order
        self.wavelength_range = wavelength_range
        self.wavelength_step = wavelength_step
        self.period = period
        self.inc_ang = inc_ang
        self.azi_ang = azi_ang
        self.pattern = torch.tensor(cv2.imread(pattern_path, cv2.IMREAD_GRAYSCALE)/255., dtype=self.geo_dtype,device=self.device)
        self.pattern_thk = pattern_thk
        self.nx, self.ny = self.pattern.shape
        if self.nx != self.ny:
            raise ValueError("nx and ny must be equal")
        self.pixel_size = period/self.nx
        self.input_material = input_material
        self.output_material = output_material
        self.pattern_A_material = pattern_A_material
        self.pattern_B_material = pattern_B_material
        
    def get_Spectrum(self):
        # Simulation environment
        inc_ang = self.inc_ang*(np.pi/180)                    # radian
        azi_ang = self.azi_ang*(np.pi/180)                    # radian
        wavelength_range = np.arange(self.wavelength_range[0], self.wavelength_range[1], self.wavelength_step)
        txx = torch.zeros(len(wavelength_range), dtype=self.sim_dtype,device=self.device)
        txy = torch.zeros(len(wavelength_range), dtype=self.sim_dtype,device=self.device)
        tyx = torch.zeros(len(wavelength_range), dtype=self.sim_dtype,device=self.device)
        tyy = torch.zeros(len(wavelength_range), dtype=self.sim_dtype,device=self.device)
        tRL = torch.zeros(len(wavelength_range), dtype=self.sim_dtype,device=self.device)
        tRR = torch.zeros(len(wavelength_range), dtype=self.sim_dtype,device=self.device)
        tLR = torch.zeros(len(wavelength_range), dtype=self.sim_dtype,device=self.device)
        tLL = torch.zeros(len(wavelength_range), dtype=self.sim_dtype,device=self.device)
        rxx = torch.zeros(len(wavelength_range), dtype=self.sim_dtype,device=self.device)
        rxy = torch.zeros(len(wavelength_range), dtype=self.sim_dtype,device=self.device)
        ryx = torch.zeros(len(wavelength_range), dtype=self.sim_dtype,device=self.device)
        ryy = torch.zeros(len(wavelength_range), dtype=self.sim_dtype,device=self.device)
        rRL = torch.zeros(len(wavelength_range), dtype=self.sim_dtype,device=self.device)
        rRR = torch.zeros(len(wavelength_range), dtype=self.sim_dtype,device=self.device)
        rLR = torch.zeros(len(wavelength_range), dtype=self.sim_dtype,device=self.device)
        rLL = torch.zeros(len(wavelength_range), dtype=self.sim_dtype,device=self.device)
        for i, lamb0 in enumerate(wavelength_range):
            lamb0 = torch.tensor(lamb0, dtype=self.geo_dtype,device=self.device)
            input_eps = Materials.Material.forward(wavelength=lamb0, name=self.input_material)**2
            output_eps = Materials.Material.forward(wavelength=lamb0, name=self.output_material)**2
            pattern_A_eps = Materials.Material.forward(wavelength=lamb0, name=self.pattern_A_material)**2
            pattern_B_eps = Materials.Material.forward(wavelength=lamb0, name=self.pattern_B_material)**2
            layer0_eps = self.pattern*pattern_A_eps +  pattern_B_eps*(1. - self.pattern)
            # geometry
            torcwa.rcwa_geo.dtype = self.geo_dtype
            torcwa.rcwa_geo.device = self.device
            torcwa.rcwa_geo.Lx = self.period
            torcwa.rcwa_geo.Ly = self.period
            torcwa.rcwa_geo.nx = self.nx
            torcwa.rcwa_geo.ny = self.ny
            torcwa.rcwa_geo.grid()

            # Generate and perform simulation
            order = [self.harmonic_order, self.harmonic_order]
            sim = torcwa.rcwa(
                freq=1/lamb0,order=order,
                L=[self.period,self.period],
                dtype=self.sim_dtype,
                device=self.device
                )
            sim.add_input_layer(eps=input_eps)
            sim.add_output_layer(eps=output_eps)
            sim.set_incident_angle(inc_ang=inc_ang,azi_ang=azi_ang)
            sim.add_layer(thickness=self.pattern_thk, eps=layer0_eps)
            sim.solve_global_smatrix()
            txx[i] = sim.S_parameters(orders=[0,0],direction='forward',port='transmission',polarization='pp',ref_order=[0,0])
            txy[i] = sim.S_parameters(orders=[0,0],direction='forward',port='transmission',polarization='ps',ref_order=[0,0])
            tyx[i] = sim.S_parameters(orders=[0,0],direction='forward',port='transmission',polarization='sp',ref_order=[0,0])
            tyy[i] = sim.S_parameters(orders=[0,0],direction='forward',port='transmission',polarization='ss',ref_order=[0,0])
            rxx[i] = sim.S_parameters(orders=[0,0],direction='forward',port='reflection',polarization='pp',ref_order=[0,0])
            rxy[i] = sim.S_parameters(orders=[0,0],direction='forward',port='reflection',polarization='ps',ref_order=[0,0])
            ryx[i] = sim.S_parameters(orders=[0,0],direction='forward',port='reflection',polarization='sp',ref_order=[0,0])
            ryy[i] = sim.S_parameters(orders=[0,0],direction='forward',port='reflection',polarization='ss',ref_order=[0,0])
        tRL = 0.5*((txx - tyy) - 1j * (txy + tyx))
        tRR = 0.5*((txx + tyy) + 1j * (txy - tyx))
        tLR = 0.5*((txx - tyy) + 1j * (txy + tyx))
        tLL = 0.5*((txx + tyy) - 1j * (txy - tyx))
        rRL = 0.5*((rxx - ryy) - 1j * (rxy + ryx))
        rRR = 0.5*((rxx + ryy) + 1j * (rxy - ryx))
        rLR = 0.5*((rxx - ryy) + 1j * (rxy + ryx))
        rLL = 0.5*((rxx + tyy) - 1j * (rxy - ryx))
        Trans_xx = torch.abs(txx)**2
        Trans_xy = torch.abs(txy)**2
        Trans_yx = torch.abs(tyx)**2
        Trans_yy = torch.abs(tyy)**2
        Trans_RL = torch.abs(tRL)**2
        Trans_RR = torch.abs(tRR)**2
        Trans_LR = torch.abs(tLR)**2
        Trans_LL = torch.abs(tLL)**2
        Refl_xx = torch.abs(rxx)**2
        Refl_xy = torch.abs(rxy)**2
        Refl_yx = torch.abs(ryx)**2
        Refl_yy = torch.abs(ryy)**2
        Refl_RL = torch.abs(rRL)**2
        Refl_RR = torch.abs(rRR)**2
        Refl_LR = torch.abs(rLR)**2
        Refl_LL = torch.abs(rLL)**2
        df = pd.DataFrame({
            'wavelength': wavelength_range,
            'Trans_xx': Trans_xx,
            'Trans_xy': Trans_xy,
            'Trans_yx': Trans_yx,
            'Trans_yy': Trans_yy,
            'Trans_RL': Trans_RL,
            'Trans_RR': Trans_RR,
            'Trans_LR': Trans_LR,
            'Trans_LL': Trans_LL,
            'Refl_xx': Refl_xx,
            'Refl_xy': Refl_xy,
            'Refl_yx': Refl_yx,
            'Refl_yy': Refl_yy,
            'Refl_RL': Refl_RL,
            'Refl_RR': Refl_RR,
            'Refl_LR': Refl_LR,
            'Refl_LL': Refl_LL
        })
        #是self.pattern_path的前面的資料夾
        shape_dir = os.path.dirname(self.pattern_path).split('\\')[:-1]
        shape_dir = '\\'.join(shape_dir)
        save_folder = os.path.join(shape_dir, 'spectrum')
        os.makedirs(save_folder, exist_ok=True)
        df.to_csv(os.path.join(save_folder, os.path.basename(self.pattern_path).replace('.png', '.csv')), index=False)
        print(f"Spectrum saved: {os.path.join(save_folder, os.path.basename(self.pattern_path).replace('.png', '.csv'))}")
