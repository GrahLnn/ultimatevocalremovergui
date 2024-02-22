# import pickle
from dataclasses import dataclass

from .separate import SeperateMDXC

# from ultimatevocalremovergui.uvr.UVR import ModelData


@dataclass
class ModelData:
    DENOISER_MODEL: str
    DEVERBER_MODEL: str
    is_deverb_vocals: bool
    deverb_vocal_opt: str
    is_denoise_model: bool
    is_gpu_conversion: int
    is_normalization: bool
    is_use_opencl: bool
    is_primary_stem_only: bool
    is_secondary_stem_only: bool
    is_denoise: bool
    is_mdx_c_seg_def: bool
    mdx_batch_size: int
    mdxnet_stem_select: str
    overlap: float
    overlap_mdx: str
    overlap_mdx23: int
    semitone_shift: float
    is_pitch_change: bool
    is_match_frequency_pitch: bool
    is_mdx_ckpt: bool
    is_mdx_c: bool
    is_mdx_combine_stems: bool
    mdx_c_configs: dict
    mdx_model_stems: list
    mdx_dim_f_set: str
    mdx_dim_t_set: str
    mdx_stem_count: int
    compensate: str
    mdx_n_fft_scale_set: str
    wav_type_set: str
    device_set: str
    mp3_bit_set: str
    save_format: str
    is_invert_spec: bool
    is_mixer_mode: bool
    demucs_stems: str
    is_demucs_combine_stems: bool
    demucs_source_list: list
    demucs_stem_count: int
    mixer_path: str
    model_name: str
    process_method: str
    model_status: bool
    primary_stem: str
    secondary_stem: str
    primary_stem_native: str
    is_ensemble_mode: bool
    ensemble_primary_stem: str
    ensemble_secondary_stem: str
    primary_model_primary_stem: str
    is_secondary_model: bool
    secondary_model: str
    secondary_model_scale: str
    demucs_4_stem_added_count: int
    is_demucs_4_stem_secondaries: bool
    is_4_stem_ensemble: bool
    pre_proc_model: str
    pre_proc_model_activated: bool
    is_pre_proc_model: bool
    is_dry_check: bool
    model_samplerate: int
    model_capacity: tuple
    is_vr_51_model: bool
    is_demucs_pre_proc_model_inst_mix: bool
    manual_download_Button: str
    secondary_model_4_stem: list
    secondary_model_4_stem_scale: list
    secondary_model_4_stem_names: list
    secondary_model_4_stem_model_names_list: list
    all_models: list
    secondary_model_other: str
    secondary_model_scale_other: str
    secondary_model_bass: str
    secondary_model_scale_bass: str
    secondary_model_drums: str
    secondary_model_scale_drums: str
    is_multi_stem_ensemble: bool
    is_karaoke: bool
    is_bv_model: bool
    bv_model_rebalance: int
    is_sec_bv_rebalance: bool
    is_change_def: bool
    model_hash_dir: str
    is_get_hash_dir_only: bool
    is_secondary_model_activated: bool
    vocal_split_model: str
    is_vocal_split_model: bool
    is_vocal_split_model_activated: bool
    is_save_inst_vocal_splitter: bool
    is_inst_only_voc_splitter: bool
    is_save_vocal_only: bool
    margin: int
    chunks: int
    mdx_segment_size: int
    model_path: str
    model_hash: str
    model_data: dict
    model_basename: str
    is_primary_model_primary_stem_only: bool
    is_primary_model_secondary_stem_only: bool


# import os

# os.chdir("c:\\Users\\yvliu\\ultimatevocalremovergui")
# print(os.getcwd())

# with open("./my_object.pkl", "rb") as f:
#     process_data = pickle.load(f)


# current_model: ModelData = process_data["model_data"]
# print(dir(current_model))
# print(current_model.__dict__)
# current_model.model_path = r"C:\Users\yvliu\AppData\Local\Programs\Ultimate Vocal Remover\models\MDX_Net_Models\MDX23C-8KFFT-InstVoc_HQ.ckpt"
# print(current_model.model_path)
# print(process_data)
# seperator = SeperateMDXC(current_model, process_data)
# seperator.seperate()
# https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/MDX23C-8KFFT-InstVoc_HQ_2.ckpt


def infer(
    audio,
    output_path,
    mdx23c_model_path,
    denoise_model_path,
    deEcho_model_path,
    process_data,
):
    ### only suport MDX23C-8KFFT-InstVoc_HQ
    # with open("./my_object.pkl", "rb") as f:
    #     process_data = pickle.load(f)
    current_model: ModelData = process_data["model_data"]
    current_model.model_path = mdx23c_model_path
    current_model.DENOISER_MODEL = denoise_model_path
    current_model.DEVERBER_MODEL = deEcho_model_path
    process_data["export_path"] = output_path
    process_data["audio_file"] = audio
    seperator = SeperateMDXC(current_model, process_data)
    seperator.seperate()
