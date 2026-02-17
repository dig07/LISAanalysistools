import h5py
import numpy as np
import shutil

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
from gbgpu.utils.utility import get_fdot
from eryn.state import BranchSupplemental
from lisatools.globalfit.hdfbackend import GFHDFBackend, GBHDFBackend, MBHHDFBackend, EMRIHDFBackend
from lisatools.globalfit.utils import SetupInfoTransfer, AllSetupInfoTransfer
from lisatools.globalfit.run import CurrentInfoGlobalFit, GlobalFit
# from global_fit_input.global_fit_settings import get_global_fit_settings

from lisatools.globalfit.state import GFBranchInfo, AllGFBranchInfo
from lisatools.globalfit.state import MBHState, EMRIState, GBState

from bbhx.utils.transform import *

from lisatools.globalfit.generatefuncs import *
from lisatools.utils.utility import AET
from lisatools.sampling.prior import SNRPrior, AmplitudeFromSNR, AmplitudeFrequencySNRPrior, GBPriorWrap

from lisatools.globalfit.stock.erebor import (
    GalForSetup, GalForSettings, PSDSetup, PSDSettings,
    MBHSetup, MBHSettings, GBSetup, GBSettings, EMRISetup, EMRISettings, SOBBHSetup,
)

from eryn.prior import uniform_dist
from eryn.utils import TransformContainer
from eryn.prior import ProbDistContainer

from eryn.moves import StretchMove, GaussianMove
from lisatools.sampling.moves.skymodehop import SkyMove

from eryn.moves import CombineMove
from lisatools.globalfit.moves import GBSpecialStretchMove, GBSpecialRJRefitMove, GBSpecialRJSearchMove, GBSpecialRJPriorMove, PSDMove, MBHSpecialMove, ResidualAddOneRemoveOneMove, GBSpecialRJSerialSearchMCMC, GFCombineMove
from lisatools.globalfit.galaxyglobal import make_gmm
from lisatools.globalfit.moves import GlobalFitMove
from lisatools.utils.utility import tukey


# import few
from lisatools.globalfit.engine import GlobalFitSettings, GeneralSetup, GeneralSettings


from eryn.utils.updates import Update

from lisatools.globalfit.recipe import Recipe, RecipeStep
import time


# SOBBH specific imports
from lisatools.sources.sobbh.waveform import SOBBHTDIWaveform
from lisatools.globalfit.stock.erebor import SOBBHSetup, SOBBHSettings

################

### DEFINE RECIPE

#############


class PSDSearchRecipeStep(RecipeStep):
    def setup_run(self, iteration, last_sample, sampler):
        # making sure
        sampler.moves = self.moves
        sampler.weights = self.weights

    def stopping_function(self, iteration, last_sample, sampler):
        # this will already be converged to max logl
        return True


class PSDPERecipeStep(RecipeStep):
    def setup_run(self, iteration, last_sample, sampler):
        # making sure
        sampler.moves = self.moves
        sampler.weights = self.weights

    def stopping_function(self, iteration, last_sample, sampler):
        # this will already be converged to max logl
        return False
    
class SOBBHSearchRecipeStep(RecipeStep):
    """
    Placeholder for the SoBBH search step. Currently we will initialise the PE around the true parameters. 
    """
    def setup_run(self, iteration, last_sample, sampler):
        # making sure
        sampler.moves = self.moves
        sampler.weights = self.weights

    def stopping_function(self, iteration, last_sample, sampler):
        # this will already be converged to max logl
        print('Starting around the true injection parameters, so stopping search immediately')
        return True
    
class SOBBHPERecipeStep(RecipeStep):
    def __init__(self, *args, moves=None, weights=None, **kwargs):
        super().__init__(moves=moves, weights=weights)
    
    def setup_run(self, iteration, last_sample, sampler):
        # making sure
        sampler.moves = self.moves
        sampler.weights = self.weights

    def stopping_function(self, iteration, last_sample, sampler):
        return False


from lisatools.sampling.stopping import SearchConvergeStopping


################

### DEFINE RECIPE

#############


def setup_recipe(recipe, engine_info, curr, acs, priors, stat,frequencies,df,Tobs,dt):

    from lisatools.sources.sobbh import SOBBHTDIWaveform  
   
    sobbh_info = curr.source_info["sobbh"]
    # TODO: adjust this indide current info
    general_info = curr.general_info
    nwalkers = curr.general_info.nwalkers
    ntemps = curr.general_info.ntemps

    gpus = curr.general_info.gpus
    cp.cuda.runtime.setDevice(gpus[0])

    waveform_gen = SOBBHTDIWaveform(sobbh_waveform_args=('F2_custom',), # Custom is the response, jaxified, slightly better BBHx
                        sobbh_waveform_kwargs={'TDIversion':2,}, # TDI 1.5 rescaling to TDI 2
                        T = Tobs/YRSID_SI, 
                        dt = dt,
                        freqs=frequencies,# Using the frequencies from the domain. 
                        frequency_bounds=(1.e-3, 0.1),) # There will be no (detectable) SoBBH outside this frequency range anyway

    wave_gen = WrapSOBBH(SOBBHTDIWaveform(**sobbh_info.initialize_kwargs), 
                         curr.general_info.start_freq_ind, curr.general_info.end_freq_ind)

    # Check if there are any of the leaves within the SoBBH branch. 
    if np.any(sobbh_inds := state.branches_inds["sobbh"][0]):
        # For each source in this branch 
        for leaf in range(sobbh_inds.shape[-1]):
            # In theory since we are using a reversible jump (although not actually)
            # Need to check that the leaf is active, i.e there is a source in this index
            if sobbh_inds[0, leaf]:
                # Dont know what this is doing
                assert np.all(sobbh_inds[:, leaf])
                # Injection paramters for this source
                inj_coords = state.branches_coords["sobbh"][0, :, leaf]
                # Right now the transform is not doing anything but we will change this. 
                inj_coords_in = sobbh_info.transform.both_transforms(inj_coords)
                # Number of walkers, # of channels, data length
                AET = cp.zeros((inj_coords.shape[0], acs.nchannels, acs.data_length), dtype=complex)
                for i in range(inj_coords.shape[0]):
                    AET[i] = wave_gen(*inj_coords_in[i],  **sobbh_info.waveform_kwargs)
                acs.add_signal_to_residual(AET) # ADDING signal to residual meaning r'= r - sum_i h(theta_i)
                # Adding to residual is subtracting from data, and subtracting from residuals is adding to data. 
    
    # What on earth is going on here? 
    betas_all = np.tile(make_ladder(sobbh_info.ndim, ntemps=ntemps), (sobbh_info.nleaves_max, 1))

    # to make the states work (what is going on here?)
    betas = betas_all[0]
    state.sub_states["sobbh"].betas_all = betas_all

    tempering_kwargs = dict(ntemps=ntemps, Tmax=np.inf, permute=False)
    
    coords_shape = (ntemps, nwalkers, sobbh_info.nleaves_max, sobbh_info.ndim)
    
    inner_moves = sobbh_info.inner_moves.copy()

    # No search needed for the sobbh now can we skip this?
    # This is search args
    sobbh_search_move_args = (
        "sobbh",
        coords_shape,
        wave_gen,
        tempering_kwargs,
        sobbh_info.waveform_kwargs.copy(),
        sobbh_info.waveform_kwargs.copy(),
        acs,
        1,
        sobbh_info.transform,
        priors,
        inner_moves,  # skip stretch for search
        acs.df
    )

    # Is this doing the right thing i have no idea?
    sobbh_search_move = ResidualAddOneRemoveOneMove(*sobbh_search_move_args)
    sobbh_search_move.accepted = np.zeros((ntemps, nwalkers), dtype=int)
    recipe.add_recipe_component(SOBBHSearchRecipeStep(moves=[sobbh_search_move]), name="sobbh search")

    # This is PE args 
    sobbh_move_args = (
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
        acs.df
    )
    sobbh_pe_move = ResidualAddOneRemoveOneMove(*sobbh_move_args)
    sobbh_pe_move.accepted = np.zeros((ntemps, nwalkers), dtype=int)
    recipe.add_recipe_component(SOBBHPERecipeStep(moves=[sobbh_pe_move]), name="sobbh pe")

    # So the recipe now contains a search step followed by a PE step. 

########################## 
##### SETTINGS ###########
##########################

class WrapSOBBH:
    def __init__(self, waveform_gen_fd, start_freq_ind, end_freq_ind):
        '''
        Convinience wrapper class that for now wraps SOBBH waveform generation to produce the AET channels in the frequency domain. 
        Currently using custom implementation of BBHx response in jax. 
        
        Args:
            waveform_gen_fd: waveform generator in the frequency domain that takes in parameters and outputs AET in the frequency domain.
            start_freq_ind: index of the starting frequency bin to keep for the likelihood calculation
            end_freq_ind: index of the ending frequency bin to keep for the likelihood calculation
        '''
        self.waveform_gen_fd = waveform_gen_fd
        self.start_freq_ind, self.end_freq_ind = start_freq_ind, end_freq_ind

    def __call__(self, *args, **kwargs):
        AET_f = cp.asarray(self.waveform_gen_fd(*args, **kwargs))
        AET_f = AET_f[:, self.start_freq_ind: self.end_freq_ind]
        return AET_f

def get_sobbh_erebor_settings(general_set: GeneralSetup) -> SOBBHSetup:
    # Where do i use this?
    injection_parameters_file = '/data/diganta/Global_fit/dev/Sobbh_PE/injection_params.npz'

    # Grab the frequencies from the domain settings to use for the waveform
    if general_set.basis_domain == "fd":
        frequencies = general_set.sensitivity_backend.settings.f_arr
        df = general_set.sensitivity_backend.settings.df

    sobbh_settings = SOBBHSettings()
                                                    
    return SOBBHSetup(sobbh_settings,frequencies,df,general_set.Tobs,general_set.dt)


def get_general_erebor_settings() -> GeneralSetup:

    # limits on parameters
    delta_safe = 1e-5
    # now with negative fdots
    
    from lisatools.utils.constants import YRSID_SI
    Tobs = YRSID_SI * 2
    dt = 5.0

    emri_source_file = "/data/asantini/packages/LISAanalysistools/emri_sangria_injection.h5"
    base_file_name = "emri_only_5th_try"
    file_store_dir = "/data/asantini/packages/LISAanalysistools/global_fit_output/"

    # TODO: connect LISA to SSB for MBHs to numerical orbits

    gpus = [1]
    cp.cuda.runtime.setDevice(gpus[0])
    # few.get_backend('cuda12x')
    nwalkers = 24
    ntemps = 1

    tukey_alpha = 0.05

    orbits = EqualArmlengthOrbits()
    gpu_orbits = EqualArmlengthOrbits(force_backend="cuda12x")

    general_settings = GeneralSettings(
        Tobs=Tobs,
        dt=dt,
        file_store_dir=file_store_dir,
        base_file_name=base_file_name,
        data_input_path=emri_source_file,
        orbits=orbits,
        gpu_orbits=gpu_orbits, 
        start_freq_ind=0,
        end_freq_ind=None,
        random_seed=103209,
        backup_iter=5,
        nwalkers=nwalkers,
        ntemps=ntemps,
        tukey_alpha=tukey_alpha,
        gpus=gpus,
        remove_from_data=["noise", "dgb", "igb", "vgb", "mbhb"],
        fixed_psd_kwargs=dict(model=sangria)
    )

    general_setup = GeneralSetup(general_settings)
    return general_setup


from lisatools.globalfit.engine import RankInfo


def get_global_fit_settings(copy_settings_file=False):

    general_setup = get_general_erebor_settings()

    # file_information["past_file_for_start"] = file_store_dir + "rework_6th_run_through" + "_parameter_estimation_main.h5"
    if copy_settings_file:
        shutil.copy(__file__, general_setup.file_store_dir + general_setup.base_file_name + "_" + __file__.split("/")[-1])

    ###############################
    ###############################
    ######    Rank/GPU setup  #####
    ###############################
    ###############################

    head_rank = 1

    main_rank = 0
    
    # run results rank will be next available rank if used
    # gmm_ranks will be all other ranks

    rank_info = RankInfo(
        head_rank=head_rank,
        main_rank=main_rank
    )

    ##################################
    ##################################
    ###  EMRI Settings  ##############
    ##################################
    ##################################

    emri_setup = get_emri_erebor_settings(general_setup)

    ##############
    ## READ OUT ##
    ##############


    global_settings = GlobalFitSettings(
        source_info={
            "emri": emri_setup,
        },
        general_info=general_setup,
        rank_info=rank_info,
        setup_function=setup_recipe,
    )

    curr_info = CurrentInfoGlobalFit(global_settings)

    return curr_info



if __name__ == "__main__":
    settings = get_global_fit_settings()
    breakpoint()
