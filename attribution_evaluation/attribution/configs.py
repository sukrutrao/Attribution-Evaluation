"""
Specify different configurations for each attribution method in this file.
For RISE, specify the path to save and load the random masks.
"""
attributor_configs = {
    "RISE": {
        "test": {"n": 1000, "mask_path": "/path/to/save/and/load/rise/masks/"},
    },
    "Occlusion": {
        "Occ5_2": {"ks": 5,
                   "stride": 2},
        "Occ16_8": {"ks": 16, "stride": 8},
    },
    "IxG": {
        "test": {}
    },
    "GB": {
        "test": {"apply_abs": True}
    },
    "Grad": {
        "test": {"abs": True},
    },
    "IntGrad": {
        "test": {}
    },
    "GradCam": {
        "test": {},
    },
    "LayerCam": {
        "test": {},
    },
    "GradCamPlusPlus": {
        "test": {},
    },
    "ScoreCam": {
        "test": {},
    },
    "AblationCam": {
        "test": {},
    },
}
