import pickle
from .separate import SeperateMDXC
from .UVR import ModelData

# import os

# os.chdir("c:\\Users\\yvliu\\ultimatevocalremovergui")
# print(os.getcwd())

# with open("./my_object.pkl", "rb") as f:
#     process_data = pickle.load(f)


# current_model: ModelData = process_data["model_data"]
# current_model.model_path = (
#     r"c:\Users\yvliu\ultimatevocalremovergui\MDX23C-8KFFT-InstVoc_HQ.ckpt"
# )
# print(current_model.model_path)
# print(process_data)
# seperator = SeperateMDXC(current_model, process_data)
# seperator.seperate()
# https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/MDX23C-8KFFT-InstVoc_HQ_2.ckpt


def infer(audio, output_path, model_path):
    ### only suport MDX23C-8KFFT-InstVoc_HQ
    with open("./my_object.pkl", "rb") as f:
        process_data = pickle.load(f)
    current_model: ModelData = process_data["model_data"]
    current_model.model_path = model_path
    process_data["export_path"] = output_path
    process_data["audio_file"] = audio
    seperator = SeperateMDXC(current_model, process_data)
    seperator.seperate()
