import pandas as pd
import os
from scipy.io import loadmat as sio_loadmat

PATH_ = r"E:\Downloads\drive-download-20240624T093854Z-001"

def _check_keys(dict):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    from scipy.io.matlab import mat_struct

    for key in dict:
        if isinstance(dict[key], mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj) -> dict:
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    from scipy.io.matlab import mat_struct

    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

list_coords = []
for file in os.listdir(PATH_):
    if file.endswith(".mat"):
        
        if file[:5] in ["rcs09", "rcs10", "rcs14", "rcs19"]:
            loc = "GP"
        else: 
            loc = "STN" 

        data = sio_loadmat(os.path.join(PATH_, file), struct_as_record=False, squeeze_me=True)
        data = _check_keys(data)
        coords_left = data["reco"]["mni"]["coords_mm"][0]
        coords_right = data["reco"]["mni"]["coords_mm"][1]

        for coord_idx, coord in enumerate(coords_left):
            list_coords.append({
                "name" : file[:5],
                "ch" : file[:5]+ f"l_ch_{coord_idx}",
                "x" : coord[0],
                "y" : coord[1],
                "z" : coord[2],
                "loc" : loc,
            })


        for coord_idx, coord in enumerate(coords_right):
            list_coords.append({
                "name" : file[:5],
                "ch" : file[:5]+ f"r_ch_{coord_idx}",
                "x" : coord[0],
                "y" : coord[1],
                "z" : coord[2],
                "loc" : loc,
            })

pd.DataFrame(list_coords).to_csv("mni_coords_subcortex.csv", index=False)
        