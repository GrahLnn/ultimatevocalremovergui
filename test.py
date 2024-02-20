import pickle
from separate import SeperateMDXC
from UVR import ModelData

with open("my_object.pkl", "rb") as f:
    process_data = pickle.load(f)

current_model: ModelData = process_data["model_data"]

seperator = SeperateMDXC(current_model, process_data)
seperator.seperate()
