from ast import excepthandler
import pandas as pd


class Extraction:
    
    def __init__(self, path, omics_colormaps):
        bitsinfo = pd.read_csv("%s\\omics_color.csv"%path)
        bitsinfo['colors'] = bitsinfo.Subtypes.map(omics_colormaps)
        self.bitsinfo = bitsinfo
        self.colormaps = omics_colormaps
        try:
            self.scaleinfo = pd.read_pickle("%s\\omics_scale.cfg"%path)
        except:
            pass