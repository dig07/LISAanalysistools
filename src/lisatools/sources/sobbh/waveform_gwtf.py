"""

Waveform/likelihood/inner-product generator using the gwtf package (https://github.com/cchapmanbird/gwtf).

The waveform used here is TaylorT3, same as what is used in waveform.py. 
The fresnel approximation is applied in a box-car window scenario to model the time-frequency waveform. 
Using a leading order time-frequency local response function. 

"""
import cupy as cp 
import numpy as np 
import pygwtf
from pygwtf.models import TaylorT3Spin
from pygwtf.generator import AnalyticTimeFrequencyWaveform
import lisaconstants as lc

class GWTF_generator: 

    def __init__(self, 
                 datas, 
                 psds, 
                 time_grid, 
                 frequency_grid, 
                 spacecraft_positions, 
                 spacecraft_LTTs, 
                 T_obs,
                 fresnel_kernel_width = 10,
                 use_GPU = True, 
                 ):
        """
        Assumptions: 
        - PSD and data are already pre-processed and in the correct shape corresponding to the tf grid provided. 
        - PSD is clipped around the zeros as expected. 
        """

        # Time and frequency grid
        self.t_grid = time_grid # time-grid boundaries (nT+1)
        self.f_grid = frequency_grid # Cental frequency grid (nF)
        self.dF = self.f_grid[1] - self.f_grid[0] # Frequency bin width
        self.dT = self.t_grid[1] - self.t_grid[0] # Time bin width

        self.nT = len(self.t_grid) - 1
        self.nF = len(self.f_grid)

        self.T_obs = T_obs # Observation time in seconds.

        # NOTE: Spacecraft positions are computed in equitorial frame, so the response function needs (ra,dec) instead of (lon,lat) for the sky location.

        # Spacecraft positions and light-travel times
        self.spacecraft_positions = spacecraft_positions #(nT, 3 (spacecrafts),3 (x,y,z)) for GWTF

        # TODO: Allow for the LTTs to not be symmetric, i.e. 12 =/= 21, should be simple inside GWTF response
        self.spacecraft_LTTs = spacecraft_LTTs #  (12, 23, 31) order #(nT, 3 (12,23,31)) for GWTF

        # Setup config that waveform generator needs
        config = {'nT':self.nT,
                  'nF':self.nF,
                  'dT':self.dT,
                  'dF':self.dF,
                  'kernel_width':fresnel_kernel_width}    
       
        if use_GPU:
            backend = 'gpu'
            self.xp = cp
        else:
            backend = 'cpu'
            self.xp = np
        self.config = config
        self.backend = backend

        # Read in the quantities defined per walker, GWTF expected shape is (#nWalkers, #nT, #nF, #nChannels)
        self.data = self.xp.asarray(datas)
        self.psd = self.xp.asarray(psds)

        # Waveform generator object which computes raw <h|h> and <d|d> statistics per waveform generator. 
        self.inner_product_statistics_object = AnalyticTimeFrequencyWaveform(model_class=TaylorT3Spin, 
                                                                config=config,
                                                                tdi_type=2,
                                                                backend=backend,
                                                                channels=self.data,
                                                                psds=self.psd,
                                                                spacecraft_orbits=self.spacecraft_positions,
                                                                spacecraft_ltts=self.spacecraft_LTTs,
                                                                block_vectorised_gpu = False, # Block vectorised mode, better for small batches -> global fit 
                                                                gf_mode = True, # Use the GF mode, allowing for one PSD and DATA array per walker.
                                                                )


        # Data is held constant during the PE process. 
        
        # shape (#nWalkers) Compute d_d for each walker. 
        # data , PSD: (#nWalkers, #nT, #nF, #nChannels)
        self.d_d = 4*self.xp.abs(self.xp.sum(self.data.conjugate() * self.data / self.psd * self.dF,axis=(1,2,3))).real
        
        # Waveform-only generator (non-gf_mode), built lazily in compute_time_frequency_waveform.
        # Only used for the residuals at the last step.
        self.waveform_gen_object = None

    def compute_time_frequency_waveform(self, params):
        """
        Compute the time-frequency waveform for each source. 

        params: (#nSources, 11) array of parameters, where the columns are:
        0: Mc, 1: eta, 2: cosinc, 3: D (Mpc), 4: f0, 5: s1, 6: s2, 7: phi_coal, 8: psi, 9: sky_lon (RA), 10: sky_lat (DEC)

        Returns waveform_array of shape (#nSources, #nT, #nF, #nChannels)
        """
        params = self.xp.asarray(params)

        Mc = params[:,0]
        eta = params[:,1]
        cosinc = params[:,2]
        D = params[:,3]*1.e+6 # Convert distance from Mpc to pc
        f0 = params[:,4]
        s1 = params[:,5]
        s2 = params[:,6]
        phi_coal = params[:,7]
        psi = params[:,8]
        sky_lon = params[:,9] # RA. in radians
        sky_lat = params[:,10] # DEC. in radians

        M = Mc * (eta)**(-3/5)

        wf_params = self.xp.column_stack((M, eta, cosinc, D, f0, s1, s2, phi_coal))
        resp_params = self.xp.column_stack((cosinc, psi, sky_lon, sky_lat))

        # gf_mode only supports the inner-product kernels, so waveforms come from a separate
        # non-gf_mode generator on the same grid/orbits. Built on first use, no data/PSD needed.
        if self.waveform_gen_object is None:
            self.waveform_gen_object = AnalyticTimeFrequencyWaveform(model_class=TaylorT3Spin,
                                                                    config=self.config,
                                                                    tdi_type=2,
                                                                    backend=self.backend,
                                                                    spacecraft_orbits=self.spacecraft_positions,
                                                                    spacecraft_ltts=self.spacecraft_LTTs,
                                                                    )

        waveform_array = self.waveform_gen_object(parameters=wf_params,
                                                  parameters_response=resp_params,
                                                  out=None,
                                                  compute_statistic=False)

        return waveform_array

    def compute_inner_products_per_segment(self, params, data_indices):
        """
        Per time-segment inner products.

        params: (#nSources, 11) array of parameters, where the columns are:
        0: Mc, 1: eta, 2: cosinc, 3: D (Mpc), 4: f0, 5: s1, 6: s2, 7: phi_coal, 8: psi, 9: sky_lon (RA), 10: sky_lat (DEC)

        data_indices: (#nSources,) array of indices corresponding to walker for the data and psd arrays for each source.

        Returns d_h, h_h each of shape (#nSources, #nT).
        """
        params = self.xp.asarray(params)

        Mc = params[:,0]
        eta = params[:,1]
        cosinc = params[:,2]
        D = params[:,3]*1.e+6 # Convert distance from Mpc to pc
        f0 = params[:,4]
        s1 = params[:,5]
        s2 = params[:,6]
        phi_coal = params[:,7]
        psi = params[:,8]
        sky_lon = params[:,9] # RA. in radians
        sky_lat = params[:,10] # DEC. in radians

        M = Mc * (eta)**(-3/5)

        wf_params = self.xp.column_stack((M, eta, cosinc, D, f0, s1, s2, phi_coal))
        resp_params = self.xp.column_stack((cosinc, psi, sky_lon, sky_lat))

        # NOTE:  nWalkers <= nSources, as we have nTemps. Where nWalkers x nTemps = nSources. 

        # Contains the inner product statistics for each source, shape (nSources, nT, 2) -> (d_h, h_h) per segment.
        statistic_array = self.inner_product_statistics_object(parameters=wf_params,#(#nSources)
                                        channels=self.data, #(#nWalkers, #nT, #nF, #nChannels)
                                        psds=self.psd,#(#nWalkers, #nT, #nF, #nChannels)
                                        parameters_response=resp_params,
                                        out = None,
                                        compute_statistic=True,
                                        data_indices = data_indices) # (nSources)

        return statistic_array[:,:,0], statistic_array[:,:,1]


    def compute_inner_products(self, params, data_indices):
        """
        Log-likelihood per source, same inputs as compute_inner_products_per_segment.
        """
        data_indices = self.xp.asarray(data_indices)

        d_h, h_h = self.compute_inner_products_per_segment(params, data_indices)

        d_h_per_source = self.xp.sum(d_h, axis=1)
        h_h_per_source = self.xp.sum(h_h, axis=1)
        log_likelihoods = -0.5 * (self.d_d[data_indices] + h_h_per_source.real - 2*d_h_per_source.real)

        return log_likelihoods


    def compute_waveform(self, ):
        pass
