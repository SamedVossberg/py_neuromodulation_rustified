import pickle
import os
from scipy import io

PATH = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/d_out_patient_across_class_10s_seglength_480_all_CB_all_labels.pkl"

with open(PATH, "rb") as f:
    data = pickle.load(f)
    data = data["all"]["ecog_stn"]

# write data to mat file
io.savemat(
    os.path.join(
        os.path.dirname(PATH),
        "d_out_patient_across_class_10s_seglength_480_all_CB_all_labels.mat",
    ),
    data,
)
