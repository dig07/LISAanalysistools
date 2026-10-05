"""

Waveform/likelihood/inner-product generator using the gwtf package (https://github.com/cchapmanbird/gwtf).

The waveform used here is TaylorT3, same as what is used in waveform.py.
The fresnel approximation is applied in a box-car window scenario to model the time-frequency waveform.
Using a leading order time-frequency local response function.

"""
import cupy as cp
import numpy as np
from pygwtf.models import TaylorT3Spin
from pygwtf.generator import AnalyticTimeFrequencyWaveform

from ...utils.constants import C_SI
from ...detector import Orbits
from ...domains import STFTSettings

class GWTF_generator:

    def __init__(self,
                 settings: STFTSettings,
                 orbits: Orbits,
                 T_obs,
                 fresnel_kernel_width = 10,
                 use_GPU = True,
                 ):
        """
        Assumptions:
        - PSD and data are already pre-processed and in the correct shape corresponding to the tf grid provided.
          Use to_gwtf_layout to map lisatools STFT arrays onto the gwtf grid.
        - PSD is clipped around the zeros as expected.

        Frequency grid: gwtf bin k is centred on (k+1)*dF, i.e. the grid always starts at dF (no DC bin).
        The gwtf grid therefore covers lisatools STFT bins 1..settings.ind_max, and bins below settings.ind_min
        (outside the active band) are zero-filled in the data and given infinite PSD by to_gwtf_layout.
        """

        self.settings = settings

        self.dF = settings.df # Frequency bin width
        self.dT = settings.dt # Time bin width

        self.nT = settings.NT # Number of time segments
        # nF here is not the same as settings.NF_active, as the gwtf grid always starts at dF (no DC bin) and goes up to settings.ind_max.
        # LISAtools uses a more general (and better) setup where the STFT grid starts at settings.ind_min and goes up to settings.ind_max, and the active band is settings.ind_min..settings.ind_max.
        self.nF = settings.NF_active # Number of gwtf frequency bins, STFT bins ind_min..ind_max (must match the data/PSD width: gwtf does not check this at call time)

        # gwtf t=0 is the start of the first segment, i.e. settings.t0 (= t_init + 850.5 + pad_trim)
        self.t_grid = settings.t0 + np.arange(self.nT + 1) * self.dT # (nT+1,) segment edges for GWTF
        self.f_grid = settings.f_arr # (nF,) central frequency grid for GWTF

        self.T_obs = T_obs # Observation time in seconds.

        # midpoints of the time segments, used for the spacecraft positions and LTTs.
        t_midpoints = orbits.xp.asarray(self.t_grid[:-1] + self.dT/2) # (nT,) on the orbits backend

        # Compute orbital quantities for response from the orbits object.
        # NOTE: Spacecraft positions are computed in equitorial frame, so the response function needs (ra,dec) instead of (lon,lat) for the sky location.
        spacecraft_positions = orbits.xp.stack([orbits.get_pos(t_midpoints, sc) for sc in (1, 2, 3)], axis=1) # (nT, 3 (spacecrafts), 3 (x,y,z)) for GWTF

        # 12, 23, 31 only needed for gwtf for now
        spacecraft_LTTs = orbits.xp.stack([orbits.get_light_travel_times(t_midpoints, link) for link in (12, 23, 31)], axis=1) * C_SI # (nT, 3), seconds -> metres, as gwtf expects

        # Setup config that waveform generator needs
        config = {'nT':self.nT,
                  'nF':self.nF,
                  'dT':self.dT,
                  'dF':self.dF,
                  'fmin': float(settings.f_arr[0]),        # centre of active bin 0 = ind_min*dF
                  'kernel_width':fresnel_kernel_width}

        if use_GPU:
            backend = 'gpu'
            self.xp = cp
        else:
            backend = 'cpu'
            self.xp = np
        self.config = config
        self.backend = backend

        self.spacecraft_positions = self.xp.asarray(spacecraft_positions)
        self.spacecraft_LTTs = self.xp.asarray(spacecraft_LTTs)

        # Waveform generator object which computes raw <h|h> and <d|d> statistics per waveform generator.
        self.inner_product_statistics_object = AnalyticTimeFrequencyWaveform(model_class=TaylorT3Spin,
                                                                config=config,
                                                                tdi_type=2,
                                                                backend=backend,
                                                                spacecraft_orbits=self.spacecraft_positions,
                                                                spacecraft_ltts=self.spacecraft_LTTs,
                                                                block_vectorised_gpu = False, # Block vectorised mode, better for small batches -> global fit
                                                                gf_mode = True, # Use the GF mode, allowing for one PSD and DATA array per walker.
                                                                )

        # Waveform-only generator (non-gf_mode), built lazily in compute_time_frequency_waveform.
        # Only used for the residuals at the last step.
        self.waveform_gen_object = None

    def to_gwtf_layout(self, arr):
        """
        Map a lisatools STFT array onto the gwtf grid. 
        Used for both the data and the PSD.
        
        - Reshape from ([#nWalkers,] #nChannels, #nT, #NF_active) to ([#nWalkers,] #nT, #nF, #nChannels)

        Parameters:
            arr: ([#nWalkers,] #nChannels, #nT, #NF_active) array on the active band settings.ind_min..settings.ind_max.
                 The walker axis is optional.
        Returns:
            ([#nWalkers,] #nT, #nF, #nChannels) array on the gwtf grid (STFT bins 1..settings.ind_max).
        """
        out = self.xp.asarray(arr)          # ([nW,] 3, nT, NF_active)

        # (#nChannels, #nT, #nF) -> (#nT, #nF, #nChannels)
        if out.ndim == 3:
            return out.transpose(1, 2, 0)
        # (#nWalkers, #nChannels, #nT, #nF) -> (#nWalkers, #nT, #nF, #nChannels)
        return out.transpose(0, 2, 3, 1)

    def _split_parameters(self, params):
        """
        params: (#nSources, 11) array of parameters, where the columns are:
        0: Mc, 1: eta, 2: cosinc, 3: D (Mpc), 4: f0, 5: s1, 6: s2, 7: phi_coal, 8: psi, 9: sky_lon (RA), 10: sky_lat (DEC)

        Returns the gwtf waveform and response parameter arrays.
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

        return wf_params, resp_params

    def compute_time_frequency_waveform(self, params):
        """
        Compute the time-frequency waveform for each source.

        params: (#nSources, 11) array of parameters, see _split_parameters.

        Returns waveform_array of shape (#nSources, #nT, #nF, #nChannels)
        """
        wf_params, resp_params = self._split_parameters(params)

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

    def compute_inner_products_per_segment(self, params, data_indices, data, psd):
        """
        Per time-segment inner products.

        params: (#nSources, 11) array of parameters, see _split_parameters.

        data_indices: (#nSources,) array of indices corresponding to walker for the data and psd arrays for each source.

        data, psd: (#nWalkers, #nT, #nF, #nChannels) on the gwtf grid.

        Returns d_h, h_h each of shape (#nSources, #nT).
        """
        wf_params, resp_params = self._split_parameters(params)

        # NOTE:  nWalkers <= nSources, as we have nTemps. Where nWalkers x nTemps = nSources.

        # Contains the inner product statistics for each source, shape (nSources, nT, 2) -> (d_h, h_h) per segment.
        statistic_array = self.inner_product_statistics_object(parameters=wf_params,#(#nSources)
                                        channels=self.xp.asarray(data), #(#nWalkers, #nT, #nF, #nChannels)
                                        psds=self.xp.asarray(psd),#(#nWalkers, #nT, #nF, #nChannels)
                                        parameters_response=resp_params,
                                        out = None,
                                        compute_statistic=True,
                                        data_indices = data_indices) # (nSources)

        return statistic_array[:,:,0], statistic_array[:,:,1]


    def compute_inner_products(self, params, data_indices, data, psd):
        """
        Log-likelihood per source, same inputs as compute_inner_products_per_segment.
        """
        data_indices = self.xp.asarray(data_indices)

        d_h, h_h = self.compute_inner_products_per_segment(params, data_indices, data, psd)

        d_h_per_source = self.xp.sum(d_h, axis=1)
        h_h_per_source = self.xp.sum(h_h, axis=1)
        log_likelihoods = -0.5 * (h_h_per_source.real - 2*d_h_per_source.real)

        return log_likelihoods
