# GUI modules
import time

# start_time = time.time()


# from .gui_data import sv_ttk
import hashlib
import json

import os
import pickle
import psutil

import queue

import subprocess


from tkinter import messagebox
from collections import Counter
from uvr.__version__ import VERSION, PATCH, PATCH_MAC, PATCH_LINUX

from datetime import datetime
from .gui_data.constants import *
from .gui_data.app_size_values import *

from .gui_data.old_data_check import file_check, remove_unneeded_yamls, remove_temps

from .lib_v5.vr_network.model_param_init import ModelParameters

from .separate import (
    cuda_available,
    mps_available,  # directml_available,
)


import sys
import yaml


# if not is_macos:
#     import torch_directml

# is_choose_arch = cuda_available and directml_available
# is_opencl_only = not cuda_available and directml_available
# is_cuda_only = cuda_available and not directml_available

is_gpu_available = cuda_available or mps_available  # or directml_available

# Change the current working directory to the directory
# this file sits in
if getattr(sys, "frozen", False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app
    # path into variable _MEIPASS'.
    BASE_PATH = sys._MEIPASS
else:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# os.chdir(BASE_PATH)  # Change the current working directory to the base path

SPLASH_DOC = os.path.join(BASE_PATH, "tmp", "splash.txt")

if os.path.isfile(SPLASH_DOC):
    os.remove(SPLASH_DOC)


def get_execution_time(function, name):
    start = time.time()
    function()
    end = time.time()
    time_difference = end - start
    print(f"{name} Execution Time: ", time_difference)


PREVIOUS_PATCH_WIN = "UVR_Patch_10_6_23_4_27"

is_dnd_compatible = True
banner_placement = -2

if OPERATING_SYSTEM == "Darwin":
    OPEN_FILE_func = lambda input_string: subprocess.Popen(["open", input_string])
    dnd_path_check = MAC_DND_CHECK
    banner_placement = -8
    current_patch = PATCH_MAC
    is_windows = False
    is_macos = True
    right_click_button = "<Button-2>"
    application_extension = ".dmg"
elif OPERATING_SYSTEM == "Linux":
    OPEN_FILE_func = lambda input_string: subprocess.Popen(["xdg-open", input_string])
    dnd_path_check = LINUX_DND_CHECK
    current_patch = PATCH_LINUX
    is_windows = False
    is_macos = False
    right_click_button = "<Button-3>"
    application_extension = ".zip"
elif OPERATING_SYSTEM == "Windows":
    OPEN_FILE_func = lambda input_string: os.startfile(input_string)
    dnd_path_check = WINDOWS_DND_CHECK
    current_patch = PATCH
    is_windows = True
    is_macos = False
    right_click_button = "<Button-3>"
    application_extension = ".exe"


def right_click_release_linux(window, top_win=None):
    if OPERATING_SYSTEM == "Linux":
        root.bind("<Button-1>", lambda e: window.destroy())
        if top_win:
            top_win.bind("<Button-1>", lambda e: window.destroy())


if not is_windows:
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context
else:
    from ctypes import windll, wintypes


def close_process(q: queue.Queue):
    def close_splash():
        name = "UVR_Launcher.exe"
        for process in psutil.process_iter(attrs=["name"]):
            process_name = process.info.get("name")

            if process_name == name:
                try:
                    process.terminate()
                    q.put(f"{name} terminated.")  # Push message to queue
                    break
                except psutil.NoSuchProcess as e:
                    q.put(f"Error terminating {name}: {e}")  # Push error to queue

                    try:
                        with open(SPLASH_DOC, "w") as f:
                            f.write("1")
                    except:
                        print("No splash screen.")

    thread = KThread(target=close_splash)
    thread.start()


def save_data(data):
    """
    Saves given data as a .pkl (pickle) file

    Paramters:
        data(dict):
            Dictionary containing all the necessary data to save
    """
    # Open data file, create it if it does not exist
    with open("data.pkl", "wb") as data_file:
        pickle.dump(data, data_file)


def load_data() -> dict:
    """
    Loads saved pkl file and returns the stored data

    Returns(dict):
        Dictionary containing all the saved data
    """
    try:
        with open("data.pkl", "rb") as data_file:  # Open data file
            data = pickle.load(data_file)

        return data
    except (ValueError, FileNotFoundError):
        # Data File is corrupted or not found so recreate it

        save_data(data=DEFAULT_DATA)

        return load_data()


def load_model_hash_data(dictionary):
    """Get the model hash dictionary"""
    with open(dictionary, "r") as d:
        return json.load(d)


def font_checker(font_file):
    chosen_font_name = None
    chosen_font_file = None

    try:
        if os.path.isfile(font_file):
            with open(font_file, "r") as d:
                chosen_font = json.load(d)

            chosen_font_name = chosen_font["font_name"]
            if chosen_font["font_file"]:
                chosen_font_file = os.path.join(
                    OTHER_FONT_PATH, chosen_font["font_file"]
                )
                chosen_font_file = (
                    chosen_font_file if os.path.isfile(chosen_font_file) else None
                )
    except Exception as e:
        print(e)

    chosen_font = chosen_font_name, chosen_font_file

    return chosen_font


debugger = []

# --Constants--
# Models
MODELS_DIR = os.path.join(BASE_PATH, "models")
VR_MODELS_DIR = os.path.join(MODELS_DIR, "VR_Models")
MDX_MODELS_DIR = os.path.join(MODELS_DIR, "MDX_Net_Models")
DEMUCS_MODELS_DIR = os.path.join(MODELS_DIR, "Demucs_Models")
DEMUCS_NEWER_REPO_DIR = os.path.join(DEMUCS_MODELS_DIR, "v3_v4_repo")
MDX_MIXER_PATH = os.path.join(BASE_PATH, "lib_v5", "mixer.ckpt")

# Cache & Parameters
VR_HASH_DIR = os.path.join(VR_MODELS_DIR, "model_data")
VR_HASH_JSON = os.path.join(VR_MODELS_DIR, "model_data", "model_data.json")
MDX_HASH_DIR = os.path.join(MDX_MODELS_DIR, "model_data")
MDX_HASH_JSON = os.path.join(MDX_HASH_DIR, "model_data.json")
MDX_C_CONFIG_PATH = os.path.join(MDX_HASH_DIR, "mdx_c_configs")

DEMUCS_MODEL_NAME_SELECT = os.path.join(
    DEMUCS_MODELS_DIR, "model_data", "model_name_mapper.json"
)
MDX_MODEL_NAME_SELECT = os.path.join(
    MDX_MODELS_DIR, "model_data", "model_name_mapper.json"
)
ENSEMBLE_CACHE_DIR = os.path.join(BASE_PATH, "gui_data", "saved_ensembles")
SETTINGS_CACHE_DIR = os.path.join(BASE_PATH, "gui_data", "saved_settings")
VR_PARAM_DIR = os.path.join(BASE_PATH, "lib_v5", "vr_network", "modelparams")
SAMPLE_CLIP_PATH = os.path.join(BASE_PATH, "temp_sample_clips")
ENSEMBLE_TEMP_PATH = os.path.join(BASE_PATH, "ensemble_temps")
DOWNLOAD_MODEL_CACHE = os.path.join(BASE_PATH, "gui_data", "model_manual_download.json")

# CR Text
CR_TEXT = os.path.join(BASE_PATH, "gui_data", "cr_text.txt")

# Style
ICON_IMG_PATH = os.path.join(BASE_PATH, "gui_data", "img", "GUI-Icon.ico")
if not is_windows:
    MAIN_ICON_IMG_PATH = os.path.join(BASE_PATH, "gui_data", "img", "GUI-Icon.png")

OWN_FONT_PATH = os.path.join(BASE_PATH, "gui_data", "own_font.json")

MAIN_FONT_NAME = "Montserrat"
SEC_FONT_NAME = "Century Gothic"
FONT_PATH = os.path.join(
    BASE_PATH, "gui_data", "fonts", "Montserrat", "Montserrat.ttf"
)  #
SEC_FONT_PATH = os.path.join(
    BASE_PATH, "gui_data", "fonts", "centurygothic", "GOTHIC.ttf"
)  #
OTHER_FONT_PATH = os.path.join(BASE_PATH, "gui_data", "fonts", "other")  #

FONT_MAPPER = {MAIN_FONT_NAME: FONT_PATH, SEC_FONT_NAME: SEC_FONT_PATH}

# Other
COMPLETE_CHIME = os.path.join(BASE_PATH, "gui_data", "complete_chime.wav")
FAIL_CHIME = os.path.join(BASE_PATH, "gui_data", "fail_chime.wav")
CHANGE_LOG = os.path.join(BASE_PATH, "gui_data", "change_log.txt")

DENOISER_MODEL_PATH = os.path.join(VR_MODELS_DIR, "UVR-DeNoise-Lite.pth")
DEVERBER_MODEL_PATH = os.path.join(VR_MODELS_DIR, "UVR-DeEcho-DeReverb.pth")

MODEL_DATA_URLS = [
    VR_MODEL_DATA_LINK,
    MDX_MODEL_DATA_LINK,
    MDX_MODEL_NAME_DATA_LINK,
    DEMUCS_MODEL_NAME_DATA_LINK,
]
MODEL_DATA_FILES = [
    VR_HASH_JSON,
    MDX_HASH_JSON,
    MDX_MODEL_NAME_SELECT,
    DEMUCS_MODEL_NAME_SELECT,
]

file_check(os.path.join(MODELS_DIR, "Main_Models"), VR_MODELS_DIR)
file_check(os.path.join(DEMUCS_MODELS_DIR, "v3_repo"), DEMUCS_NEWER_REPO_DIR)
remove_unneeded_yamls(DEMUCS_MODELS_DIR)

remove_temps(ENSEMBLE_TEMP_PATH)
remove_temps(SAMPLE_CLIP_PATH)
remove_temps(os.path.join(BASE_PATH, "img"))

if not os.path.isdir(ENSEMBLE_TEMP_PATH):
    os.mkdir(ENSEMBLE_TEMP_PATH)

if not os.path.isdir(SAMPLE_CLIP_PATH):
    os.mkdir(SAMPLE_CLIP_PATH)

model_hash_table = {}
data = load_data()


def drop(event, accept_mode: str = "files"):
    path = event.data
    if accept_mode == "folder":
        path = path.replace("{", "").replace("}", "")
        if not os.path.isdir(path):
            messagebox.showerror(
                parent=root,
                title=INVALID_FOLDER_ERROR_TEXT[0],
                message=INVALID_FOLDER_ERROR_TEXT[1],
            )
            return
        root.export_path_var.set(path)
    elif accept_mode in ["files", FILE_1, FILE_2, FILE_1_LB, FILE_2_LB]:
        path = path.replace("{", "").replace("}", "")
        for dnd_file in dnd_path_check:
            path = path.replace(f" {dnd_file}", f";{dnd_file}")
        path = path.split(";")
        path[-1] = path[-1].replace(";", "")

        if accept_mode == "files":
            root.inputPaths = tuple(path)
            root.process_input_selections()
            root.update_inputPaths()
        elif accept_mode in [FILE_1, FILE_2]:
            if len(path) == 2:
                root.select_audiofile(path[0])
                root.select_audiofile(path[1], is_primary=False)
                root.DualBatch_inputPaths = []
                root.check_dual_paths()
            elif len(path) == 1:
                if accept_mode == FILE_1:
                    root.select_audiofile(path[0])
                else:
                    root.select_audiofile(path[0], is_primary=False)

        elif accept_mode in [FILE_1_LB, FILE_2_LB]:
            return path
    else:
        return


class ModelData:
    def __init__(
        self,
        model_name: str,
        selected_process_method=ENSEMBLE_MODE,
        is_secondary_model=False,
        primary_model_primary_stem=None,
        is_primary_model_primary_stem_only=False,
        is_primary_model_secondary_stem_only=False,
        is_pre_proc_model=False,
        is_dry_check=False,
        is_change_def=False,
        is_get_hash_dir_only=False,
        is_vocal_split_model=False,
    ):

        device_set = root.device_set_var.get()
        self.DENOISER_MODEL = DENOISER_MODEL_PATH
        self.DEVERBER_MODEL = DEVERBER_MODEL_PATH
        self.is_deverb_vocals = (
            root.is_deverb_vocals_var.get()
            if os.path.isfile(DEVERBER_MODEL_PATH)
            else False
        )
        self.deverb_vocal_opt = DEVERB_MAPPER[root.deverb_vocal_opt_var.get()]
        self.is_denoise_model = (
            True
            if root.denoise_option_var.get() == DENOISE_M
            and os.path.isfile(DENOISER_MODEL_PATH)
            else False
        )
        self.is_gpu_conversion = 0 if root.is_gpu_conversion_var.get() else -1
        self.is_normalization = root.is_normalization_var.get()  #
        self.is_use_opencl = (
            False  # True if is_opencl_only else root.is_use_opencl_var.get()
        )
        self.is_primary_stem_only = root.is_primary_stem_only_var.get()
        self.is_secondary_stem_only = root.is_secondary_stem_only_var.get()
        self.is_denoise = (
            True if not root.denoise_option_var.get() == DENOISE_NONE else False
        )
        self.is_mdx_c_seg_def = root.is_mdx_c_seg_def_var.get()  #
        self.mdx_batch_size = (
            1
            if root.mdx_batch_size_var.get() == DEF_OPT
            else int(root.mdx_batch_size_var.get())
        )
        self.mdxnet_stem_select = root.mdxnet_stems_var.get()
        self.overlap = (
            float(root.overlap_var.get())
            if not root.overlap_var.get() == DEFAULT
            else 0.25
        )
        self.overlap_mdx = (
            float(root.overlap_mdx_var.get())
            if not root.overlap_mdx_var.get() == DEFAULT
            else root.overlap_mdx_var.get()
        )
        self.overlap_mdx23 = int(float(root.overlap_mdx23_var.get()))
        self.semitone_shift = float(root.semitone_shift_var.get())
        self.is_pitch_change = False if self.semitone_shift == 0 else True
        self.is_match_frequency_pitch = root.is_match_frequency_pitch_var.get()
        self.is_mdx_ckpt = False
        self.is_mdx_c = False
        self.is_mdx_combine_stems = root.is_mdx23_combine_stems_var.get()  #
        self.mdx_c_configs = None
        self.mdx_model_stems = []
        self.mdx_dim_f_set = None
        self.mdx_dim_t_set = None
        self.mdx_stem_count = 1
        self.compensate = None
        self.mdx_n_fft_scale_set = None
        self.wav_type_set = root.wav_type_set  #
        self.device_set = (
            device_set.split(":")[-1].strip() if ":" in device_set else device_set
        )
        self.mp3_bit_set = root.mp3_bit_set_var.get()
        self.save_format = root.save_format_var.get()
        self.is_invert_spec = root.is_invert_spec_var.get()  #
        self.is_mixer_mode = False  #
        self.demucs_stems = root.demucs_stems_var.get()
        self.is_demucs_combine_stems = root.is_demucs_combine_stems_var.get()
        self.demucs_source_list = []
        self.demucs_stem_count = 0
        self.mixer_path = MDX_MIXER_PATH
        self.model_name = model_name
        self.process_method = selected_process_method
        self.model_status = (
            False
            if self.model_name == CHOOSE_MODEL or self.model_name == NO_MODEL
            else True
        )
        self.primary_stem = None
        self.secondary_stem = None
        self.primary_stem_native = None
        self.is_ensemble_mode = False
        self.ensemble_primary_stem = None
        self.ensemble_secondary_stem = None
        self.primary_model_primary_stem = primary_model_primary_stem
        self.is_secondary_model = True if is_vocal_split_model else is_secondary_model
        self.secondary_model = None
        self.secondary_model_scale = None
        self.demucs_4_stem_added_count = 0
        self.is_demucs_4_stem_secondaries = False
        self.is_4_stem_ensemble = False
        self.pre_proc_model = None
        self.pre_proc_model_activated = False
        self.is_pre_proc_model = is_pre_proc_model
        self.is_dry_check = is_dry_check
        self.model_samplerate = 44100
        self.model_capacity = 32, 128
        self.is_vr_51_model = False
        self.is_demucs_pre_proc_model_inst_mix = False
        self.manual_download_Button = None
        self.secondary_model_4_stem = []
        self.secondary_model_4_stem_scale = []
        self.secondary_model_4_stem_names = []
        self.secondary_model_4_stem_model_names_list = []
        self.all_models = []
        self.secondary_model_other = None
        self.secondary_model_scale_other = None
        self.secondary_model_bass = None
        self.secondary_model_scale_bass = None
        self.secondary_model_drums = None
        self.secondary_model_scale_drums = None
        self.is_multi_stem_ensemble = False
        self.is_karaoke = False
        self.is_bv_model = False
        self.bv_model_rebalance = 0
        self.is_sec_bv_rebalance = False
        self.is_change_def = is_change_def
        self.model_hash_dir = None
        self.is_get_hash_dir_only = is_get_hash_dir_only
        self.is_secondary_model_activated = False
        self.vocal_split_model = None
        self.is_vocal_split_model = is_vocal_split_model
        self.is_vocal_split_model_activated = False
        self.is_save_inst_vocal_splitter = (
            root.is_save_inst_set_vocal_splitter_var.get()
        )
        self.is_inst_only_voc_splitter = root.check_only_selection_stem(INST_STEM_ONLY)
        self.is_save_vocal_only = root.check_only_selection_stem(IS_SAVE_VOC_ONLY)

        if selected_process_method == ENSEMBLE_MODE:
            self.process_method, _, self.model_name = model_name.partition(
                ENSEMBLE_PARTITION
            )
            self.model_and_process_tag = model_name
            self.ensemble_primary_stem, self.ensemble_secondary_stem = (
                root.return_ensemble_stems()
            )

            is_not_secondary_or_pre_proc = (
                not is_secondary_model and not is_pre_proc_model
            )
            self.is_ensemble_mode = is_not_secondary_or_pre_proc

            if root.ensemble_main_stem_var.get() == FOUR_STEM_ENSEMBLE:
                self.is_4_stem_ensemble = self.is_ensemble_mode
            elif (
                root.ensemble_main_stem_var.get() == MULTI_STEM_ENSEMBLE
                and root.chosen_process_method_var.get() == ENSEMBLE_MODE
            ):
                self.is_multi_stem_ensemble = True

            is_not_vocal_stem = self.ensemble_primary_stem != VOCAL_STEM
            self.pre_proc_model_activated = (
                root.is_demucs_pre_proc_model_activate_var.get()
                if is_not_vocal_stem
                else False
            )

        if self.process_method == VR_ARCH_TYPE:
            self.is_secondary_model_activated = (
                root.vr_is_secondary_model_activate_var.get()
                if not is_secondary_model
                else False
            )
            self.aggression_setting = float(
                int(root.aggression_setting_var.get()) / 100
            )
            self.is_tta = root.is_tta_var.get()
            self.is_post_process = root.is_post_process_var.get()
            self.window_size = int(root.window_size_var.get())
            self.batch_size = (
                1
                if root.batch_size_var.get() == DEF_OPT
                else int(root.batch_size_var.get())
            )
            self.crop_size = int(root.crop_size_var.get())
            self.is_high_end_process = (
                "mirroring" if root.is_high_end_process_var.get() else "None"
            )
            self.post_process_threshold = float(root.post_process_threshold_var.get())
            self.model_capacity = 32, 128
            self.model_path = os.path.join(VR_MODELS_DIR, f"{self.model_name}.pth")
            self.get_model_hash()
            if self.model_hash:
                self.model_hash_dir = os.path.join(
                    VR_HASH_DIR, f"{self.model_hash}.json"
                )
                if is_change_def:
                    self.model_data = self.change_model_data()
                else:
                    self.model_data = (
                        self.get_model_data(VR_HASH_DIR, root.vr_hash_MAPPER)
                        if not self.model_hash == WOOD_INST_MODEL_HASH
                        else WOOD_INST_PARAMS
                    )
                if self.model_data:
                    vr_model_param = os.path.join(
                        VR_PARAM_DIR,
                        "{}.json".format(self.model_data["vr_model_param"]),
                    )
                    self.primary_stem = self.model_data["primary_stem"]
                    self.secondary_stem = secondary_stem(self.primary_stem)
                    self.vr_model_param = ModelParameters(vr_model_param)
                    self.model_samplerate = self.vr_model_param.param["sr"]
                    self.primary_stem_native = self.primary_stem
                    if (
                        "nout" in self.model_data.keys()
                        and "nout_lstm" in self.model_data.keys()
                    ):
                        self.model_capacity = (
                            self.model_data["nout"],
                            self.model_data["nout_lstm"],
                        )
                        self.is_vr_51_model = True
                    self.check_if_karaokee_model()

                else:
                    self.model_status = False

        if self.process_method == MDX_ARCH_TYPE:
            self.is_secondary_model_activated = (
                root.mdx_is_secondary_model_activate_var.get()
                if not is_secondary_model
                else False
            )
            self.margin = int(root.margin_var.get())
            self.chunks = 0
            self.mdx_segment_size = int(root.mdx_segment_size_var.get())
            self.get_mdx_model_path()
            self.get_model_hash()
            if self.model_hash:
                self.model_hash_dir = os.path.join(
                    MDX_HASH_DIR, f"{self.model_hash}.json"
                )
                if is_change_def:
                    self.model_data = self.change_model_data()
                else:
                    self.model_data = self.get_model_data(
                        MDX_HASH_DIR, root.mdx_hash_MAPPER
                    )
                if self.model_data:

                    if "config_yaml" in self.model_data:
                        self.is_mdx_c = True
                        config_path = os.path.join(
                            MDX_C_CONFIG_PATH, self.model_data["config_yaml"]
                        )
                        if os.path.isfile(config_path):
                            with open(config_path) as f:
                                config = ConfigDict(
                                    yaml.load(f, Loader=yaml.FullLoader)
                                )

                            self.mdx_c_configs = config

                            if self.mdx_c_configs.training.target_instrument:
                                # Use target_instrument as the primary stem and set 4-stem ensemble to False
                                target = self.mdx_c_configs.training.target_instrument
                                self.mdx_model_stems = [target]
                                self.primary_stem = target
                            else:
                                # If no specific target_instrument, use all instruments in the training config
                                self.mdx_model_stems = (
                                    self.mdx_c_configs.training.instruments
                                )
                                self.mdx_stem_count = len(self.mdx_model_stems)

                                # Set primary stem based on stem count
                                if self.mdx_stem_count == 2:
                                    self.primary_stem = self.mdx_model_stems[0]
                                else:
                                    self.primary_stem = self.mdxnet_stem_select

                                # Update mdxnet_stem_select based on ensemble mode
                                if self.is_ensemble_mode:
                                    self.mdxnet_stem_select = self.ensemble_primary_stem
                        else:
                            self.model_status = False
                    else:
                        self.compensate = (
                            self.model_data["compensate"]
                            if root.compensate_var.get() == AUTO_SELECT
                            else float(root.compensate_var.get())
                        )
                        self.mdx_dim_f_set = self.model_data["mdx_dim_f_set"]
                        self.mdx_dim_t_set = self.model_data["mdx_dim_t_set"]
                        self.mdx_n_fft_scale_set = self.model_data[
                            "mdx_n_fft_scale_set"
                        ]
                        self.primary_stem = self.model_data["primary_stem"]
                        self.primary_stem_native = self.model_data["primary_stem"]
                        self.check_if_karaokee_model()

                    self.secondary_stem = secondary_stem(self.primary_stem)
                else:
                    self.model_status = False

        if self.process_method == DEMUCS_ARCH_TYPE:
            self.is_secondary_model_activated = (
                root.demucs_is_secondary_model_activate_var.get()
                if not is_secondary_model
                else False
            )
            if not self.is_ensemble_mode:
                self.pre_proc_model_activated = (
                    root.is_demucs_pre_proc_model_activate_var.get()
                    if not root.demucs_stems_var.get() in [VOCAL_STEM, INST_STEM]
                    else False
                )
            self.margin_demucs = int(root.margin_demucs_var.get())
            self.chunks_demucs = 0
            self.shifts = int(root.shifts_var.get())
            self.is_split_mode = root.is_split_mode_var.get()
            self.segment = root.segment_var.get()
            self.is_chunk_demucs = root.is_chunk_demucs_var.get()
            self.is_primary_stem_only = (
                root.is_primary_stem_only_var.get()
                if self.is_ensemble_mode
                else root.is_primary_stem_only_Demucs_var.get()
            )
            self.is_secondary_stem_only = (
                root.is_secondary_stem_only_var.get()
                if self.is_ensemble_mode
                else root.is_secondary_stem_only_Demucs_var.get()
            )
            self.get_demucs_model_data()
            self.get_demucs_model_path()

        if self.model_status:
            self.model_basename = os.path.splitext(os.path.basename(self.model_path))[0]
        else:
            self.model_basename = None

        self.pre_proc_model_activated = (
            self.pre_proc_model_activated if not self.is_secondary_model else False
        )

        self.is_primary_model_primary_stem_only = is_primary_model_primary_stem_only
        self.is_primary_model_secondary_stem_only = is_primary_model_secondary_stem_only

        is_secondary_activated_and_status = (
            self.is_secondary_model_activated and self.model_status
        )
        is_demucs = self.process_method == DEMUCS_ARCH_TYPE
        is_all_stems = root.demucs_stems_var.get() == ALL_STEMS
        is_valid_ensemble = not self.is_ensemble_mode and is_all_stems and is_demucs
        is_multi_stem_ensemble_demucs = self.is_multi_stem_ensemble and is_demucs

        if is_secondary_activated_and_status:
            if (
                is_valid_ensemble
                or self.is_4_stem_ensemble
                or is_multi_stem_ensemble_demucs
            ):
                for key in DEMUCS_4_SOURCE_LIST:
                    self.secondary_model_data(key)
                    self.secondary_model_4_stem.append(self.secondary_model)
                    self.secondary_model_4_stem_scale.append(self.secondary_model_scale)
                    self.secondary_model_4_stem_names.append(key)

                self.demucs_4_stem_added_count = sum(
                    i is not None for i in self.secondary_model_4_stem
                )
                self.is_secondary_model_activated = any(
                    i is not None for i in self.secondary_model_4_stem
                )
                self.demucs_4_stem_added_count -= (
                    1 if self.is_secondary_model_activated else 0
                )

                if self.is_secondary_model_activated:
                    self.secondary_model_4_stem_model_names_list = [
                        i.model_basename if i is not None else None
                        for i in self.secondary_model_4_stem
                    ]
                    self.is_demucs_4_stem_secondaries = True
            else:
                primary_stem = (
                    self.ensemble_primary_stem
                    if self.is_ensemble_mode and is_demucs
                    else self.primary_stem
                )
                self.secondary_model_data(primary_stem)

        if self.process_method == DEMUCS_ARCH_TYPE and not is_secondary_model:
            if self.demucs_stem_count >= 3 and self.pre_proc_model_activated:
                self.pre_proc_model = root.process_determine_demucs_pre_proc_model(
                    self.primary_stem
                )
                self.pre_proc_model_activated = True if self.pre_proc_model else False
                self.is_demucs_pre_proc_model_inst_mix = (
                    root.is_demucs_pre_proc_model_inst_mix_var.get()
                    if self.pre_proc_model
                    else False
                )

        if self.is_vocal_split_model and self.model_status:
            self.is_secondary_model_activated = False
            if self.is_bv_model:
                primary = (
                    BV_VOCAL_STEM
                    if self.primary_stem_native == VOCAL_STEM
                    else LEAD_VOCAL_STEM
                )
            else:
                primary = (
                    LEAD_VOCAL_STEM
                    if self.primary_stem_native == VOCAL_STEM
                    else BV_VOCAL_STEM
                )
            self.primary_stem, self.secondary_stem = primary, secondary_stem(primary)

        self.vocal_splitter_model_data()

    def vocal_splitter_model_data(self):
        if not self.is_secondary_model and self.model_status:
            self.vocal_split_model = root.process_determine_vocal_split_model()
            self.is_vocal_split_model_activated = (
                True if self.vocal_split_model else False
            )

            if self.vocal_split_model:
                if self.vocal_split_model.bv_model_rebalance:
                    self.is_sec_bv_rebalance = True

    def secondary_model_data(self, primary_stem):
        secondary_model_data = root.process_determine_secondary_model(
            self.process_method,
            primary_stem,
            self.is_primary_stem_only,
            self.is_secondary_stem_only,
        )
        self.secondary_model = secondary_model_data[0]
        self.secondary_model_scale = secondary_model_data[1]
        self.is_secondary_model_activated = False if not self.secondary_model else True
        if self.secondary_model:
            self.is_secondary_model_activated = (
                False
                if self.secondary_model.model_basename == self.model_basename
                else True
            )

        # print("self.is_secondary_model_activated: ", self.is_secondary_model_activated)

    def check_if_karaokee_model(self):
        if IS_KARAOKEE in self.model_data.keys():
            self.is_karaoke = self.model_data[IS_KARAOKEE]
        if IS_BV_MODEL in self.model_data.keys():
            self.is_bv_model = self.model_data[IS_BV_MODEL]  #
        if IS_BV_MODEL_REBAL in self.model_data.keys() and self.is_bv_model:
            self.bv_model_rebalance = self.model_data[IS_BV_MODEL_REBAL]  #

    def get_mdx_model_path(self):

        if self.model_name.endswith(CKPT):
            self.is_mdx_ckpt = True

        ext = "" if self.is_mdx_ckpt else ONNX

        for file_name, chosen_mdx_model in root.mdx_name_select_MAPPER.items():
            if self.model_name in chosen_mdx_model:
                if file_name.endswith(CKPT):
                    ext = ""
                self.model_path = os.path.join(MDX_MODELS_DIR, f"{file_name}{ext}")
                break
        else:
            self.model_path = os.path.join(MDX_MODELS_DIR, f"{self.model_name}{ext}")

        self.mixer_path = os.path.join(MDX_MODELS_DIR, f"mixer_val.ckpt")

    def get_demucs_model_path(self):

        demucs_newer = self.demucs_version in {DEMUCS_V3, DEMUCS_V4}
        demucs_model_dir = DEMUCS_NEWER_REPO_DIR if demucs_newer else DEMUCS_MODELS_DIR

        for file_name, chosen_model in root.demucs_name_select_MAPPER.items():
            if self.model_name == chosen_model:
                self.model_path = os.path.join(demucs_model_dir, file_name)
                break
        else:
            self.model_path = os.path.join(
                DEMUCS_NEWER_REPO_DIR, f"{self.model_name}.yaml"
            )

    def get_demucs_model_data(self):

        self.demucs_version = DEMUCS_V4

        for key, value in DEMUCS_VERSION_MAPPER.items():
            if value in self.model_name:
                self.demucs_version = key

        if DEMUCS_UVR_MODEL in self.model_name:
            self.demucs_source_list, self.demucs_source_map, self.demucs_stem_count = (
                DEMUCS_2_SOURCE,
                DEMUCS_2_SOURCE_MAPPER,
                2,
            )
        else:
            self.demucs_source_list, self.demucs_source_map, self.demucs_stem_count = (
                DEMUCS_4_SOURCE,
                DEMUCS_4_SOURCE_MAPPER,
                4,
            )

        if not self.is_ensemble_mode:
            self.primary_stem = (
                PRIMARY_STEM if self.demucs_stems == ALL_STEMS else self.demucs_stems
            )
            self.secondary_stem = secondary_stem(self.primary_stem)

    def get_model_data(self, model_hash_dir, hash_mapper: dict):
        model_settings_json = os.path.join(model_hash_dir, f"{self.model_hash}.json")

        if os.path.isfile(model_settings_json):
            with open(model_settings_json, "r") as json_file:
                return json.load(json_file)
        else:
            for hash, settings in hash_mapper.items():
                if self.model_hash in hash:
                    return settings

            return self.get_model_data_from_popup()

    def change_model_data(self):
        if self.is_get_hash_dir_only:
            return None
        else:
            return self.get_model_data_from_popup()

    def get_model_data_from_popup(self):
        if self.is_dry_check:
            return None

        if not self.is_change_def:
            confirm = messagebox.askyesno(
                title=UNRECOGNIZED_MODEL[0],
                message=f'"{self.model_name}"{UNRECOGNIZED_MODEL[1]}',
                parent=root,
            )
            if not confirm:
                return None

        if self.process_method == VR_ARCH_TYPE:
            root.pop_up_vr_param(self.model_hash)
            return root.vr_model_params
        elif self.process_method == MDX_ARCH_TYPE:
            root.pop_up_mdx_model(self.model_hash, self.model_path)
            return root.mdx_model_params

    def get_model_hash(self):
        self.model_hash = None

        if not os.path.isfile(self.model_path):
            self.model_status = False
            self.model_hash is None
        else:
            if model_hash_table:
                for key, value in model_hash_table.items():
                    if self.model_path == key:
                        self.model_hash = value
                        break

            if not self.model_hash:
                try:
                    with open(self.model_path, "rb") as f:
                        f.seek(-10000 * 1024, 2)
                        self.model_hash = hashlib.md5(f.read()).hexdigest()
                except:
                    self.model_hash = hashlib.md5(
                        open(self.model_path, "rb").read()
                    ).hexdigest()

                table_entry = {self.model_path: self.model_hash}
                model_hash_table.update(table_entry)

        # print(self.model_name," - ", self.model_hash)
