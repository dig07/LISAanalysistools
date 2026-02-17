from __future__ import annotations
from __future__ import annotations
from lisatools.detector import EqualArmlengthOrbits
import numpy as np
import cupy as cp 
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from typing import Optional, Any
from copy import deepcopy

# imports
from ..waveformbase import AETTDIWaveform
# time domain response function
from fastlisaresponse import ResponseWrapper
from ...detector import EqualArmlengthOrbits

# sobbh waveforms import 
import TaylorF2ecc
import TaylorT2

# LISA imports (BBHx)
# from bbhx.response.fastfdresponse import LISATDIResponse
# from bbhx.utils.interpolate import CubicSplineInterpolant
# from bbhx.waveformbuild import TemplateInterpFD


from lisaconstants import c
YRSID_SI = 31558149.763545603
# Armlength of LISA in seconds
Armlength = 2.5e+9/c

# Custom LISA response (JAX implementation)
import LISA_response

td_default_response_kwargs = dict(
    t0=30000.0,
    order=25,
    tdi="2nd generation",
    tdi_chan="AET",
    orbits=EqualArmlengthOrbits(),
)

from lisatools.detector import Orbits

# Crappy LISA-response code
import LISA_response


class T2wrapper:
    """ Wrapper for TaylorT2 waveform generator to be used with ResponseWrapper
     
    Args:
        times: Array of times at which to generate the waveform.
        downsampling_factor: Factor by which to downsample the times for waveform generation.
        T3_tf_correspondence: Whether to use T3 time-frequency correspondence.
    """
    def __init__(self,
                times: np.ndarray,
                downsampling_factor: Optional[int] = 100, 
                T3_tf_correspondence: Optional[bool] = True):

        self.times = times
        
        self.sparse_times = times[::downsampling_factor]

        self.T3_tf_correspondence = T3_tf_correspondence



    def __call__(self,
                 m1,
                 m2,
                 e0, 
                 D, 
                 inc, 
                 f0,
                 psi,
                 **kwargs) -> Any:
        
        # TODO: COALLESENCE PHASE PARAMETER NOT IN HERE
        """ Generate the waveform using TaylorT2 waveform generator.
        
        Args:
            m1: Mass of the primary black hole in solar masses.
            m2: Mass of the secondary black hole in solar masses.
            e0: Initial eccentricity.
            D: Distance to the source in pc.
            inc: Inclination angle in radians.
            f0: Initial frequency in Hz.
            psi: Polarization angle in radians.

        """

        # Containers to carry the waveform
        h_plus = np.zeros_like(self.times,dtype=complex)
        h_cross = np.zeros_like(self.times,dtype=complex)

        # Note waveform is generated from t0 = 0 all the way to t_obs, the clipping/throwing away of waveform is done by the responsewrapper 
        h_plus_t,h_cross_t,times_t = TaylorT2.CPU_old.waveform_construct_interp(
                m1,
                m2,
                e0,
                D,
                self.sparse_times,
                self.times,
                inc,
                f0,
                psi,
                T3_tf_correspondence= self.T3_tf_correspondence)

        # WF generation only occurs for t<tc, filling in the appropriate arrays. 
        h_plus[self.times<=times_t[-1]] = h_plus_t
        h_cross[self.times<=times_t[-1]] = h_cross_t
        
        return(h_plus-1j*h_cross)


class F2wrapper:

    """ Wrapper for TaylorF2ecc waveform generator to be used with ResponseWrapper
     
    Args:
        freqs: Array of frequencies at which to generate the waveform.
        downsampling_factor: Factor by which to downsample the frequencies for waveform generation.
        Tobs: Observation time in years.
        TDI: Type of TDI channels to use ('AET' or 'XYZ').
        force_backend: Backend to use for computation ('cuda12x' or 'cpu', see https://github.com/mikekatz04/BBHx/blob/dev/src/bbhx/response/fastfdresponse.py#L52-L53).
    """
    def __init__(self,
                freqs: np.ndarray,
                downsampling_factor: Optional[int] = 1000,
                Tobs : Optional[float] = 1.0,
                TDI: Optional[str] = 'AET',
                force_backend: Optional[str] = 'cuda12x',
                TDIversion: Optional[int] = 2,
                orbit_class: Optional[Orbits] = EqualArmlengthOrbits()):

        self.freqs = freqs
        self.downsampling_factor = downsampling_factor
        self.Tobs = Tobs*YRSID_SI
        self.force_backend = force_backend

        if TDI == "AET":
            TDITag = 'AET'
        elif TDI == 'XYZ':
            TDITag = 'XYZ'    
        else:
            raise ValueError("TDIType must be 'AET', 'XYZ' for BBHx reponse ")
        print('Using TDI channels: ',TDITag)
        # Setup BBHx response
        # self.response = LISATDIResponse(TDItag=TDITag,
        #                                 force_backend=self.force_backend,
        #                                 tdi2 = True if TDIversion==2 else False,orbits=orbit_class)

        # # Sets up FD interpolator, directly interpolating both the waveform and the response transfer functions
        # self.interp_response = TemplateInterpFD(force_backend=self.force_backend)    

    def __call__(self,
                 m1,
                 m2,
                 e0, 
                 D, 
                 inc, 
                 f0,
                 s1,
                 s2,
                 psi,
                 lam,
                 beta,
                 coallesence_phase=0.0,
                 **kwargs) -> Any:
    
        
        """ Generate the waveform using TaylorF2ecc waveform generator, then pass it through BBHx for response. 
        
        TODO: Batched waveform calls with BBHx response for speedup. Maybe can do this because we are only evaluating waveforms on a sparse frequency grid.

        Args:
            m1: Mass of the primary black hole in solar masses.
            m2: Mass of the secondary black hole in solar masses.
            e0: Initial eccentricity.
            D: Distance to the source in pc.
            inc: Inclination angle in radians.
            f0: Initial frequency in Hz.
            s1: Dimensionless spin of the primary black hole. (Spin always aligned with orbital angular momentum).
            s2: Dimensionless spin of the secondary black hole. (Spin always aligned with orbital angular momentum).
            psi: Polarization angle in radians.
            lam: Ecliptic longitude in radians.
            beta: Ecliptic latitude in radians.
            coallesence_phase: Phase at coalescence (default: 0.0).
            **kwargs: Additional keyword arguments (e.g., T, dt from ResponseWrapper).
            
        Returns:
            TDIchannels: Time-delay interferometry channels after passing the waveform through BBHx response.

        """
        import time
        t_start = time.perf_counter()
        
        print(f"[BENCH F2] Starting F2wrapper for {self.freqs.size} frequencies")

       # TODO: deal with the fact that f_low should really be the minimum frequency in self.freqs even if its sparse
        # Ensures we are masking out the frequencies below which f(t=0)
        # Keep everything on GPU - use searchsorted on the device
        start_idx = int(jnp.searchsorted(self.freqs, f0))
        
        # Slice directly on GPU instead of boolean indexing
        filtered_freqs = self.freqs[start_idx:]
        
        # Downsample using slice indexing (stays on GPU)
        freqs_sparse = filtered_freqs[::self.downsampling_factor]
        
        t_masking = time.perf_counter()
        print(f"[BENCH F2] Frequency masking: {(t_masking - t_start)*1e3:.3f} ms")
        print(f"[BENCH F2] Sparse frequencies: {freqs_sparse.size}, Dense frequencies: {filtered_freqs.size}")

        Amp,Phi,times = TaylorF2ecc.CPU_old.waveform_construct_sparse(
                m1,
                m2,
                e0,
                D,
                freqs_sparse,
                s1,
                s2,
                f0,
                coallesence_phase)
        
        t_waveform = time.perf_counter()
        print(f"[BENCH F2] Waveform generation: {(t_waveform - t_masking)*1e3:.3f} ms")
        
        
        time_mask = times<=self.Tobs
        times = times[time_mask]
        Amp = Amp[time_mask]
        Phi = Phi[time_mask]
        freqs_sparse = freqs_sparse[time_mask]
        
        t_time_mask = time.perf_counter()
        print(f"[BENCH F2] Time masking: {(t_time_mask - t_waveform)*1e3:.3f} ms ({np.count_nonzero(time_mask)}/{time_mask.size} points kept)")

        # phi ref is an additional phase rotation which is not needed since we have applied all the rotations inside the waveform phase
        phi_ref = 0.

        # Sky locations
        beta = beta
        lam = lam
    
        #Number of binaries currently hard set 1 
        num_bin_all = 1

        # Length of sparse frequency array for computation 
        length = freqs_sparse.size

        # Only one harmonic for now 
        num_modes = 1
    
        # params are amp, phase, tf, transferL1re, transferL1im, transferL2re, transferL2im, transferL3re, transferL3im
        num_interp_params = 9 
        
        # Allocate buffer once and fill it directly
        out_buffer = cp.zeros(num_interp_params*length*num_modes*num_bin_all)
        out_buffer = out_buffer.reshape(num_interp_params, num_bin_all, num_modes, length)
        
        # Converting from h_plus to h_lm and transfer JAX arrays to CuPy efficiently
        # Use cp.from_dlpack for zero-copy transfer from JAX to CuPy (both on GPU)
        amp_factor = 1.0/np.sqrt(5/(64*np.pi))
        out_buffer[0,0,0,:]= cp.from_dlpack(Amp) * amp_factor
        out_buffer[1,0,0,:]= cp.from_dlpack(Phi) 
        out_buffer[2,0,0,:]= cp.from_dlpack(times)+97729939.827664 
    
        out_buffer = out_buffer.flatten().copy()

        dense_freqs_length = filtered_freqs.size
        
        # Convert JAX array to CuPy using zero-copy transfer
        freqs_sparse = cp.from_dlpack(freqs_sparse)
        
        t_buffer_setup = time.perf_counter()
        print(f"[BENCH F2] Buffer setup: {(t_buffer_setup - t_time_mask)*1e3:.3f} ms")

        # Generate response 
        self.response(freqs_sparse,
                    inc,
                    lam,
                    beta,
                    psi,
                    phi_ref,
                    length,
                    out_buffer=out_buffer,
                    modes = [(2,2)])
        
        t_response = time.perf_counter()
        print(f"[BENCH F2] BBHx response calculation: {(t_response - t_buffer_setup)*1e3:.3f} ms")
        
        # setup interpolant
        # spline = CubicSplineInterpolant(
        #     freqs_sparse,
        #     out_buffer,
        #     length=length,
        #     num_interp_params=num_interp_params,
        #     num_modes=num_modes,
        #     num_bin_all=num_bin_all,
        #     force_backend=self.force_backend,
        # )
        
        t_spline = time.perf_counter()
        print(f"[BENCH F2] Spline setup: {(t_spline - t_response)*1e3:.3f} ms")
        
        # Convert filtered_freqs JAX array to CuPy using zero-copy transfer
        filtered_freqs = cp.from_dlpack(filtered_freqs)

        template_channels = self.interp_response(filtered_freqs,
                                                #  spline.container,
                                                 np.array([97729939.827664]),# Start time 
                                                 np.array([self.Tobs+97729939.827664]),# End time  # https://nextcloud-dcc-fi-csc-okd-exchange1.2.rahtiapp.fi/apps/files/files/16326?dir=/dcc-fi-csc-okd-exchange1-globalstorage/validation/mojito_light_v1_0_0/SOBHB/L1_0p4Hz&editing=false&openfile=true
                                                 length,
                                                 num_modes,
                                                 3)
        
        print('Start time: ', 97729939.827664)
        print('End time: ', (self.Tobs+97729939.827664))
        
        t_interp = time.perf_counter()
        print(f"[BENCH F2] Interpolation to dense grid: {(t_interp - t_spline)*1e3:.3f} ms")

        # combine into one data stream
        data_out = cp.zeros((3, dense_freqs_length), dtype=complex)
        for temp, start_i, length_i in zip(
            template_channels,
            self.interp_response.start_inds,
            self.interp_response.lengths,
        ):
            data_out[:, start_i : start_i + length_i] = temp
        
        t_combine = time.perf_counter()
        print(f"[BENCH F2] Combining channels: {(t_combine - t_interp)*1e3:.3f} ms")
    

        t_end = time.perf_counter()
        print(f"[BENCH F2] TOTAL F2wrapper: {(t_end - t_start)*1e3:.3f} ms")

        return(data_out)

class F2wrapper_custom_jax:

    """ Wrapper for TaylorF2ecc waveform generator to be used with jax response (basically a reimplementation of BBHx with jax and some small variations):
     
    Args:
        freqs: Array of frequencies at which to generate the waveform.
        downsampling_factor: Factor by which to downsample the frequencies for waveform generation. NOT BEING USED IN THIS FUNCTION YET. 
        Tobs: Observation time in years.
        TDI: Type of TDI channels to use ('AET' or 'XYZ').
        force_backend: Backend to use for computation ('cuda12x' or 'cpu', see https://github.com/mikekatz04/BBHx/blob/dev/src/bbhx/response/fastfdresponse.py#L52-L53).
        TDIversion: Version of TDI to use (1 or 2). Currently this is always set to 2, but the response does gen 1.5 which is good "enough" approximation to 2nd gen for now. 
        dt_orbit_reinterp: Time step in seconds for orbit resampling from cubic interpolation, needed for LISA response calculation.

    """
    def __init__(self,
                freqs: np.ndarray,
                downsampling_factor: Optional[int] = 1000, # Does absolutely nothing here. 
                Tobs : Optional[float] = 1.0,
                TDI: Optional[str] = 'AET',
                force_backend: Optional[str] = 'cuda12x',
                TDIversion: Optional[int] = 2,
                dt_orbit_reinterp: Optional[float] = 600.0,
                dt : Optional[float] = 10.0,
                ):

        self.freqs = freqs
        self.downsampling_factor = downsampling_factor
        self.Tobs = Tobs*YRSID_SI
        self.force_backend = force_backend
        
        self.p_splines = LISA_response.Zero_order_response_fast_locally_constant.generate_mojito_orbit_splines(
            '/data/diganta/Global_fit/Initial_Debugging/Orbits/esa-trailing-orbits-mojito_validation_test_2.h5',self.Tobs,dt=dt_orbit_reinterp)
        
        if TDI == "AET":
            TDITag = 'AET'
        elif TDI == 'XYZ':
            TDITag = 'XYZ'    
        else:
            raise ValueError("TDIType must be 'AET', 'XYZ' for BBHx reponse ")
        print('Using TDI channels: ',TDITag)
    
    @jax.jit(static_argnums=[0,12])
    def __call__(self,
                 m1,
                 m2,
                 e0, 
                 D, 
                 inc, 
                 f0,
                 s1,
                 s2,
                 psi,
                 lam,
                 beta,
                 coallesence_phase=0.0,
                 **kwargs) -> Any:
    
        
        """ Generate the waveform using TaylorF2ecc waveform generator, then pass it through BBHx for response. 
        
        TODO: Batched waveform calls with BBHx response for speedup. Maybe can do this because we are only evaluating waveforms on a sparse frequency grid.

        Args:
            m1: Mass of the primary black hole in solar masses.
            m2: Mass of the secondary black hole in solar masses.
            e0: Initial eccentricity.
            D: Distance to the source in pc.
            inc: Inclination angle in radians.
            f0: Initial frequency in Hz.
            s1: Dimensionless spin of the primary black hole. (Spin always aligned with orbital angular momentum).
            s2: Dimensionless spin of the secondary black hole. (Spin always aligned with orbital angular momentum).
            psi: Polarization angle in radians.
            lam: Ecliptic longitude in radians.
            beta: Ecliptic latitude in radians.
            coallesence_phase: Phase at coalescence (default: 0.0).
            **kwargs: Additional keyword arguments (e.g., T, dt from ResponseWrapper).
            
        Returns:
            TDIchannels: Time-delay interferometry channels after passing the waveform through BBHx response.

        """
        # import time
        # t_start = time.perf_counter()
        
        # print(f"[BENCH F2_custom_jax] Starting F2wrapper_custom_jax for {self.freqs.size} frequencies")

        Amp, Phi, times = TaylorF2ecc.CPU_old.waveform_construct_sparse(
                m1,
                m2,
                e0,
                D,
                self.freqs,
                s1,
                s2,
                f0,
                coallesence_phase)
        
        # t_waveform = time.perf_counter()
        # print(f"[BENCH F2_custom_jax] Waveform generation: {(t_waveform - t_start)*1e3:.3f} ms")
        
        # # Create mask - use numpy arrays throughout for faster boolean operations
        # time_and_freq_mask = (times <= self.Tobs) & (self.freqs >= f0)

        # Find where freq >= f0 (start index)
        start_idx = jnp.searchsorted(self.freqs, f0).astype(int)

        # Find where time > Tobs (end index) - times is monotonic with freq
        end_idx = jnp.searchsorted(times, self.Tobs, side='right').astype(int)
        
        # t_masking = time.perf_counter()
        # print(f"[BENCH F2_custom_jax] Index masking: {(t_masking - t_waveform)*1e3:.3f} ms")
        # print(f"[BENCH F2_custom_jax] Valid frequency range: start_idx={start_idx}, end_idx={end_idx}")
    
        T = LISA_response.Zero_order_response_fast_locally_constant.Zero_order_response_15_splined(
                                                      self.freqs,
                                                      times,
                                                      beta,
                                                      lam,
                                                      psi,
                                                      inc,
                                                      self.p_splines,)
        
        # t_response = time.perf_counter()
        # print(f"[BENCH F2_custom_jax] LISA response calculation: {(t_response - t_masking)*1e3:.3f} ms")

        # Conjugate reverses the fourier transform definition to the usual, correct one
        XYZ_15 = jax.numpy.conj(T * Amp / jax.numpy.sqrt(5/(64*jax.numpy.pi)) * jax.numpy.exp(1j*Phi))
        
        # t_xyz = time.perf_counter()
        # print(f"[BENCH F2_custom_jax] XYZ computation: {(t_xyz - t_response)*1e3:.3f} ms")
        
        AET_15 = LISA_response.Zero_order_response_fast_locally_constant.XYZ2AET(XYZ_15[0], XYZ_15[1], XYZ_15[2])

        # t_end = time.perf_counter()
        # print(f"[BENCH F2_custom_jax] XYZ to AET conversion: {(t_end - t_xyz)*1e3:.3f} ms")
        # print(f"[BENCH F2_custom_jax] TOTAL F2wrapper_custom_jax: {(t_end - t_start)*1e3:.3f} ms")

        # Return only the masked data
        # This eliminates the cost of creating large zero-filled arrays
        return AET_15, (start_idx, end_idx)



class SOBBHTDIWaveform(AETTDIWaveform):
    """Generate SOBBH waveforms with the TDI LISA Response.

    Args:
        T: Observation time in years.
        dt: Time cadence in seconds.
        sobbh_waveform_args: Arguments for SOBBH waveform generator.
        sobbh_waveform_kwargs: Keyword arguments for SOBBH waveform generator.
        response_kwargs: Keyword arguments for :class:`ResponseWrapper`.
        freqs: Frequencies at which to evaluate the waveform and response, only used for frequency domain waveforms not for Time-domain. If None, will be generated based on T and dt.
        frequency_bounds: (minimum,maximum) frequency for waveform (and response generation), only used for frequency domain waveforms not for Time-domain. 

    """

    def __init__(
        self,
        T: Optional[float] = 1.0,
        dt: Optional[float] = 10.0,
        sobbh_waveform_args: Optional[tuple] = ('F2',),
        sobbh_waveform_kwargs: Optional[dict] = {},
        response_kwargs: Optional[dict] = td_default_response_kwargs,
        freqs = None, 
        frequency_bounds: Optional[tuple]= (1.e-3,1.e-1),
    ): 
        ##### t0 what should it be. 


        self.times = jax.numpy.arange(0, T*YRSID_SI+dt, dt)
        
        # Want to jit the waveform wrt whateer frequencies are used here. 
        if freqs is not None:
            # If provided frequencies from the waveform generator use these. 
            self.freqs = jax.numpy.asarray(freqs)
        else:
            # If not generate from times. 
            self.freqs = jax.numpy.fft.rfftfreq(self.times.size,d=dt)

        if sobbh_waveform_args[0] == 'F2_custom':
            # Frequency domain (F2 + personal implementation of BBHx response with old analytic orbits!) NOTE: TEMPORARY!!!!
            
            # # resolution of FFT grid 
            # df = 1/(T*YRSID_SI)

            # self.freqs = jax.numpy.arange(frequency_bounds[0],frequency_bounds[1],df)
            self.freqs = self.freqs[(self.freqs>=frequency_bounds[0]) & (self.freqs<=frequency_bounds[1])]

            self.response = F2wrapper_custom_jax(
                                    self.freqs,
                                    downsampling_factor= sobbh_waveform_kwargs.get('downsampling_factor',1000),
                                    Tobs = T,
                                    TDI = sobbh_waveform_kwargs.get('TDI','AET'),
                                    force_backend = sobbh_waveform_kwargs.get('force_backend','cuda12x'),
                                    TDIversion = response_kwargs.get('TDIversion',2),
                                    ) 
            
        elif sobbh_waveform_args[0] == 'F2':
            # Frequency domain (F2 + BBHx response)
            # self.freqs = jax.numpy.arange(frequency_bounds[0],frequency_bounds[1],df)
            self.freqs = self.freqs[(self.freqs>=frequency_bounds[0]) & (self.freqs<=frequency_bounds[1])]

            print('Frequency domain SoBBH waveform number of points: ',self.freqs.size)
            print('TDI version: ', sobbh_waveform_kwargs.get('TDIversion',2))
            
            self.response = F2wrapper(
                                    self.freqs,
                                    downsampling_factor= sobbh_waveform_kwargs.get('downsampling_factor',1000),
                                    Tobs = T,
                                    TDI = sobbh_waveform_kwargs.get('TDI','AET'),
                                    force_backend = sobbh_waveform_kwargs.get('force_backend','cuda12x'),
                                    TDIversion = sobbh_waveform_kwargs.get('TDIversion',2),
                                    orbit_class= response_kwargs.get('orbit_class',EqualArmlengthOrbits())
                                    )        
            
        elif sobbh_waveform_args[0] == 'T2': 
            # Time domain (T2 (non-spinning for now (need to add the spin corrections))+ Fastlisaresponse)
            gen_wave = T2wrapper(self.times)

            for key in td_default_response_kwargs:
                response_kwargs[key] = response_kwargs.get(
                    key, td_default_response_kwargs[key]
                )
            
            # Sky parameters in the arguments supplied to the response. 
            index_lambda = 7
            index_beta = 8

            self.response = ResponseWrapper(
                        gen_wave,
                        T,
                        dt,
                        index_lambda,
                        index_beta,
                        flip_hx=True,  # set to True if waveform is h+ - ihx
                        remove_sky_coords=True,
                        is_ecliptic_latitude=True,
                        remove_garbage="zero",  # removes the beginning of the signal that has bad information
                        **response_kwargs)

            # # Storing the clipped times incase the user wants them (outside of this time the orbit interpolation is not good)
            # tdi_start_ind = int(response_kwargs['t0']/dt)

            # # Done in a way that is consistent with ResponseWrapper internals
            # actual_n = self.response.n  # or self.response.response_model.num_pts
            # self.times_ = np.arange(0, actual_n * dt, dt)[tdi_start_ind : -tdi_start_ind]            

        else: 
            raise ValueError("sobbh_waveform_args[0] must be either 'F2' or 'T2', these are the only waveforms implemented right now")
        
    @property
    def dt(self) -> float:
        """timestep"""
        return self.response.dt

    @property
    def clipped_times(self) -> np.ndarray:
        """Clipped times after removing garbage at the start and end."""
        return self.times_
    @property
    def clipped_freqs(self) -> np.ndarray:
        """Clipped frequencies after removing garbage at the start and end."""
        return self.freqs

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.response(*args, **kwargs)
        # try:
        #     return self.response(*args, **kwargs)
        # except Exception as e:
        #     print(e)
        #     breakpoint()    