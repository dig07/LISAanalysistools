import h5py
import numpy as np
import shutil
from copy import deepcopy

try:
    import cupy as cp
    gpu_available = True
except (ModuleNotFoundError, ImportError) as e:
    import numpy as cp
    gpu_available = True

from eryn.moves.tempering import TemperatureControl, make_ladder

from lisatools.detector import EqualArmlengthOrbits
from eryn.moves import TemperatureControl
from lisatools.utils.constants import *
from eryn.state import BranchSupplemental

from lisatools.globalfit.hdfbackend import GFHDFBackend
from lisatools.globalfit.utils import SetupInfoTransfer, AllSetupInfoTransfer
from lisatools.globalfit.run import CurrentInfoGlobalFit, GlobalFit

from lisatools.globalfit.state import GFBranchInfo, AllGFBranchInfo

from lisatools.globalfit.stock.erebor import SOBBHSetup, SOBBHSettings

from eryn.prior import uniform_dist
from eryn.utils import TransformContainer
from eryn.prior import ProbDistContainer

from eryn.moves import StretchMove, GaussianMove
from lisatools.sampling.moves.skymodehop import SkyMove

from eryn.moves import CombineMove
from lisatools.globalfit.moves import ResidualAddOneRemoveOneMove
from lisatools.utils.utility import tukey

from lisatools.globalfit.engine import GlobalFitSettings, GeneralSetup, GeneralSettings, RankInfo
from lisatools.globalfit.recipe import Recipe, RecipeStep

# SOBBH specific imports
from lisatools.sources.sobbh.waveform import SOBBHTDIWaveform


############################
###   RECIPE STEPS       ###
############################


class SOBBHSearchRecipeStep(RecipeStep):
    """Placeholder search step — immediately stops since we initialise at injection parameters."""

    def setup_run(self, iteration, last_sample, sampler):
        sampler.moves = self.moves
        sampler.weights = self.weights

    def stopping_function(self, iteration, last_sample, sampler):
        print("Starting around the true injection parameters, so stopping search immediately")
        return True


class SOBBHPERecipeStep(RecipeStep):
    """PE step — runs indefinitely until externally stopped."""

    def __init__(self, *args, moves=None, weights=None, **kwargs):
        super().__init__(moves=moves, weights=weights)

    def setup_run(self, iteration, last_sample, sampler):
        sampler.moves = self.moves
        sampler.weights = self.weights

    def stopping_function(self, iteration, last_sample, sampler):
        return False


############################
###   WAVEFORM WRAPPER   ###
############################


class WrapSOBBH:
    """Wraps SOBBHTDIWaveform (F2_custom) for use in the global fit MCMC.

    F2wrapper_custom_jax returns a *sparse* representation:
        (AET_15, (start_idx, end_idx))
    where AET_15 has shape (3, len(freqs)) and start_idx/end_idx indicate
    which frequency bins within that array are valid (freq >= f0 and time <= Tobs).

    This wrapper embeds the sparse output into a full-size array of shape
    (nchannels, data_length) that the likelihood / residual machinery expects.

    Args:
        waveform_gen: An instantiated SOBBHTDIWaveform object.
        nchannels: Number of TDI channels (typically 3 for AET).
        data_length: Length of the frequency-domain data array that the
            AnalysisContainerSomething (acs) holds.
        freq_offset: Index offset between the waveform's internal frequency
            grid and the acs frequency grid. This is the index in the acs
            array at which the waveform's first frequency bin sits.
            For a standard FD domain with min_freq applied, this equals
            the index of frequency_bounds[0] in the full FFT grid.
    """

    def __init__(self, waveform_gen, nchannels, data_length, freq_offset=0):
        self.waveform_gen = waveform_gen
        self.nchannels = nchannels
        self.data_length = data_length
        self.freq_offset = freq_offset

    def __call__(self, *args, **kwargs):
        
        #Waveform generator call 
        result = self.waveform_gen(*args, **kwargs)
        
        # start_idx and end_idx define the valid slice of the output AET array
        AET, (start_idx, end_idx) = result

        start_idx = int(start_idx)
        end_idx = int(end_idx)
        
        # Results array 
        AET_full = cp.zeros((self.nchannels, self.data_length), dtype=complex)

        # Place the valid slice into the full array at the correct offset
        dest_start = self.freq_offset + start_idx
        dest_end = self.freq_offset + end_idx

        AET_full[:, dest_start:dest_end] = cp.asarray(AET[:, start_idx:end_idx])

        return AET_full

############################
###   SETUP RECIPE       ###
############################


def setup_recipe(recipe, engine_info, curr, acs, priors, state):
    """Build the SOBBH search + PE recipe.

    Called by the framework with the standard 6-argument signature:
        (recipe, engine_info, curr, acs, priors, state)

    All domain information (frequencies, Tobs, dt) is obtained from
    curr.general_info and curr.source_info["sobbh"].
    """

    sobbh_info = curr.source_info["sobbh"]
    general_info = curr.general_info
    nwalkers = general_info.nwalkers
    ntemps = general_info.ntemps

    gpus = general_info.gpus
    cp.cuda.runtime.setDevice(gpus[0])

    # ----- Instantiate the waveform generator via initialize_kwargs -----
    # initialize_kwargs were built in get_sobbh_erebor_settings and contain
    # everything SOBBHTDIWaveform needs (waveform type, freqs, bounds, etc.)
    raw_waveform = SOBBHTDIWaveform(**sobbh_info.initialize_kwargs)

    # Compute the frequency offset: index in the acs freq grid where the
    # waveform's frequency grid starts. The waveform is already clipped to
    # frequency_bounds, so its first freq aligns with frequency_bounds[0].
    freq_bounds = sobbh_info.initialize_kwargs.get("frequency_bounds", (1e-3, 0.1))

    # Index of the closest frequency bin in the acs grid to frequency_bounds[0]
    waveform_freq_offset = int(np.round(freq_bounds[0] / acs.df))

    wave_gen = WrapSOBBH(
        raw_waveform,
        nchannels=acs.nchannels,
        data_length=acs.data_length,
        freq_offset=waveform_freq_offset,
    )

    # ----- Subtract injected sources from the residual -----
    # For each active leaf, compute the waveform at injection params
    # and subtract it from the data residual so the residual is
    # r = d - sum_i h(theta_i_true).
    if np.any(sobbh_inds := state.branches_inds["sobbh"][0]):
        for leaf in range(sobbh_inds.shape[-1]):
            if sobbh_inds[0, leaf]:
                assert np.all(sobbh_inds[:, leaf]), \
                    f"All walkers must agree on leaf {leaf} being active"
                inj_coords = state.branches_coords["sobbh"][0, :, leaf]
                inj_coords_in = sobbh_info.transform.both_transforms(inj_coords)

                AET = cp.zeros((inj_coords.shape[0], acs.nchannels, acs.data_length), dtype=complex)
                for i in range(inj_coords.shape[0]):
                    AET[i] = wave_gen(*inj_coords_in[i], **sobbh_info.waveform_kwargs)
                # add_signal_to_residual subtracts the signal: r' = r - h
                acs.add_signal_to_residual(AET)

    # ----- Temperature ladder -----
    # Each leaf gets its own temperature ladder (shape: nleaves_max x ntemps).
    # For fixed-source PE with ntemps=1 this is trivially all ones. TODO: fix this
    betas_all = np.tile(
        make_ladder(sobbh_info.ndim, ntemps=ntemps),
        (sobbh_info.nleaves_max, 1),
    )
    state.sub_states["sobbh"].betas_all = betas_all

    tempering_kwargs = dict(ntemps=ntemps, Tmax=np.inf, permute=False)
    coords_shape = (ntemps, nwalkers, sobbh_info.nleaves_max, sobbh_info.ndim)

    inner_moves = sobbh_info.inner_moves.copy()

    # ----- Search step (skipped immediately since we start at truth) -----
    sobbh_search_move_args = (
        "sobbh",
        coords_shape,
        wave_gen,
        tempering_kwargs,
        sobbh_info.waveform_kwargs.copy(),  # waveform_gen_kwargs
        sobbh_info.waveform_kwargs.copy(),  # waveform_like_kwargs
        acs,
        1,  # num_repeats = 1 for search (since we are skipping it)
        sobbh_info.transform,
        priors,
        inner_moves,
        acs.df,
    )
    
    sobbh_search_move = ResidualAddOneRemoveOneMove(*sobbh_search_move_args)
    sobbh_search_move.accepted = np.zeros((ntemps, nwalkers), dtype=int)
    recipe.add_recipe_component(
        SOBBHSearchRecipeStep(moves=[sobbh_search_move]),
        name="sobbh search",
    )

    # ----- PE step -----
    sobbh_pe_move_args = (
        "sobbh",
        coords_shape,
        wave_gen,
        tempering_kwargs,
        sobbh_info.waveform_kwargs.copy(),
        sobbh_info.waveform_kwargs.copy(),
        acs,
        sobbh_info.num_prop_repeats,
        sobbh_info.transform,
        priors,
        inner_moves,
        acs.df,
    )
    sobbh_pe_move = ResidualAddOneRemoveOneMove(*sobbh_pe_move_args)
    sobbh_pe_move.accepted = np.zeros((ntemps, nwalkers), dtype=int)
    recipe.add_recipe_component(
        SOBBHPERecipeStep(moves=[sobbh_pe_move]),
        name="sobbh pe",
    )


############################
###   SOURCE SETTINGS    ###
############################


def get_sobbh_erebor_settings(
    general_set: GeneralSetup,
    injection_parameters_file: str,
    nsources: int = 6,
    num_prop_repeats: int = 200,
) -> SOBBHSetup:
    """Build SOBBHSetup for fixed-source PE.

    Args:
        general_set: The GeneralSetup containing domain info (Tobs, dt, frequencies).
        injection_parameters_file: Path to .npz file containing injection parameters.
            Must have key 'injection_params' with shape (nsources, 11).
            Parameter order: m1, m2, e0, D, inc, f0, s1, s2, psi, lam, beta.
        nsources: Number of SOBBH sources (fixed, no RJMCMC).
        num_prop_repeats: Number of inner MCMC steps per global fit iteration.

    Returns:
        SOBBHSetup ready to be plugged into GlobalFitSettings.
    """

    # ----- Load injection parameters -----
    inj_data = np.load(injection_parameters_file)
    injection_params = inj_data["injection_params"]  # shape: (nsources, 11)
    assert injection_params.shape == (nsources, 11), \
        f"Expected injection_params shape ({nsources}, 11), got {injection_params.shape}"

    # ----- Get frequency grid from the domain -----
    # The domain settings on general_set provide the frequency array and df
    # that the likelihood is computed on. We pass the *full* domain frequency
    # grid to SOBBHTDIWaveform; it will internally clip to frequency_bounds.
    domain_settings = general_set.sensitivity_backend.basis_settings
    frequencies = domain_settings.f_arr
    df = domain_settings.df

    # ----- Build initialize_kwargs for SOBBHTDIWaveform -----
    # These are stored on the Setup and used in setup_recipe to instantiate
    # the waveform generator.
    frequency_bounds = (1e-3, 0.1)

    initialize_kwargs = dict(
        sobbh_waveform_args=("F2_custom",),
        sobbh_waveform_kwargs={"TDIversion": 2},
        T=general_set.Tobs / YRSID_SI,  # Tobs in years
        dt=general_set.dt,
        freqs=frequencies,
        frequency_bounds=frequency_bounds,
    )

    # ----- Build SOBBHSettings -----
    sobbh_settings = SOBBHSettings(
        Tobs=general_set.Tobs,
        dt=general_set.dt,
        initialize_kwargs=initialize_kwargs,
        waveform_kwargs={},  # F2_custom takes no extra kwargs beyond positional params
        injection=injection_params,
        waveform_type="F2_custom",
        frequency_bounds=frequency_bounds,
        nleaves_max=nsources,
        nleaves_min=nsources,  # fixed number of sources
        ndim=11,
        num_prop_repeats=num_prop_repeats,
    )

    return SOBBHSetup(sobbh_settings)


############################
###   GENERAL SETTINGS   ###
############################


def get_general_erebor_settings(
    data_processor_class,
    processor_init_kwargs: dict,
    data_file_dir: str,
    base_file_name: str = "sobbh_pe",
    file_store_dir: str = "/data/diganta/Global_fit/global_fit_output/",
    Tobs_years: float = 2.0,
    dt: float = 5.0,
    gpus: list = None,
    nwalkers: int = 24,
    ntemps: int = 1,
    tukey_alpha: float = 0.05,
    basis_domain: str = "fd",
    start_freq: float = None,
    end_freq: float = None,
) -> GeneralSetup:
    """Build GeneralSetup for SOBBH-only PE.

    Args:
        data_processor_class: A BaseProcessingStep subclass (e.g. L1ProcessingStep).
        processor_init_kwargs: kwargs to instantiate data_processor_class.
        data_file_dir: Not used directly — processor_init_kwargs should contain paths.
        base_file_name: Prefix for output files.
        file_store_dir: Directory for output chain files.
        Tobs_years: Observation time in years.
        dt: Sampling cadence in seconds.
        gpus: List of GPU indices to use.
        nwalkers: Number of ensemble walkers.
        ntemps: Number of temperatures (1 for PE without tempering).
        tukey_alpha: Tukey window parameter.
        basis_domain: 'fd' for frequency domain.
        start_freq: Minimum frequency for the domain (Hz). None = use all.
        end_freq: Maximum frequency for the domain (Hz). None = use all.

    Returns:
        GeneralSetup with data loaded and domain configured.
    """
    if gpus is None:
        gpus = [0]

    Tobs = YRSID_SI * Tobs_years

    cp.cuda.runtime.setDevice(gpus[0])

    orbits = EqualArmlengthOrbits()
    gpu_orbits = EqualArmlengthOrbits(force_backend="cuda12x")

    general_settings = GeneralSettings(
        Tobs=Tobs,
        dt=dt,
        file_store_dir=file_store_dir,
        base_file_name=base_file_name,
        orbits=orbits,
        gpu_orbits=gpu_orbits,
        basis_domain=basis_domain,
        start_freq=start_freq,
        end_freq=end_freq,
        random_seed=103209,
        backup_iter=5,
        nwalkers=nwalkers,
        ntemps=ntemps,
        tukey_alpha=tukey_alpha,
        gpus=gpus,
        data_processor=data_processor_class,
        processor_init_kwargs=processor_init_kwargs,
        sensitivity_init_kwargs={"force_backend": "cuda12x"},
    )

    general_setup = GeneralSetup(general_settings)
    return general_setup


############################
###   ENTRY POINT        ###
############################


def get_global_fit_settings(
    data_processor_class,
    processor_init_kwargs: dict,
    injection_parameters_file: str,
    nsources: int = 6,
    file_store_dir: str = "/data/diganta/Global_fit/global_fit_output/",
    base_file_name: str = "sobbh_pe",
    Tobs_years: float = 2.0,
    dt: float = 5.0,
    gpus: list = None,
    nwalkers: int = 24,
    ntemps: int = 1,
    num_prop_repeats: int = 200,
    copy_settings_file: bool = False,
) -> CurrentInfoGlobalFit:
    """Build the full global fit configuration for SOBBH-only PE.

    Args:
        data_processor_class: A BaseProcessingStep subclass for loading data.
        processor_init_kwargs: kwargs for instantiating the data processor.
        injection_parameters_file: Path to .npz with 'injection_params' key,
            shape (nsources, 11).
        nsources: Fixed number of SOBBH sources.
        file_store_dir: Output directory for chain files.
        base_file_name: Prefix for output files.
        Tobs_years: Observation time in years.
        dt: Sampling cadence in seconds.
        gpus: GPU indices.
        nwalkers: Number of walkers.
        ntemps: Number of temperatures.
        num_prop_repeats: Inner MCMC steps per iteration.
        copy_settings_file: If True, copy this settings file to the output dir.

    Returns:
        CurrentInfoGlobalFit ready to be passed to GlobalFit.run().
    """
    if gpus is None:
        gpus = [0]

    # ----- General setup (data loading, domain, PSD) -----
    general_setup = get_general_erebor_settings(
        data_processor_class=data_processor_class,
        processor_init_kwargs=processor_init_kwargs,
        data_file_dir="",
        base_file_name=base_file_name,
        file_store_dir=file_store_dir,
        Tobs_years=Tobs_years,
        dt=dt,
        gpus=gpus,
        nwalkers=nwalkers,
        ntemps=ntemps,
    )

    if copy_settings_file:
        shutil.copy(
            __file__,
            general_setup.file_store_dir
            + general_setup.base_file_name
            + "_"
            + __file__.split("/")[-1],
        )

    # ----- Rank setup -----
    rank_info = RankInfo(head_rank=1, main_rank=0)

    # ----- SOBBH source settings -----
    sobbh_setup = get_sobbh_erebor_settings(
        general_setup,
        injection_parameters_file=injection_parameters_file,
        nsources=nsources,
        num_prop_repeats=num_prop_repeats,
    )

    # ----- Assemble global fit settings -----
    global_settings = GlobalFitSettings(
        source_info={"sobbh": sobbh_setup},
        general_info=general_setup,
        rank_info=rank_info,
        setup_function=setup_recipe,
    )

    curr_info = CurrentInfoGlobalFit(global_settings)
    return curr_info


if __name__ == "__main__":
    # Example usage with L1 data processor:
    #
    # from lisatools.globalfit.preprocessing import L1ProcessingStep
    #
    # settings = get_global_fit_settings(
    #     data_processor_class=L1ProcessingStep,
    #     processor_init_kwargs=dict(
    #         L1_folder="/path/to/L1/data/",
    #         source_types=["sobhb"],
    #     ),
    #     injection_parameters_file="/data/diganta/Global_fit/dev/Sobbh_PE/injection_params.npz",
    #     nsources=6,
    #     gpus=[0],
    # )
    # breakpoint()
    print("Run with appropriate data_processor_class and injection_parameters_file.")
