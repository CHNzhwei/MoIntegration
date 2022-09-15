import pandas as pd
import numpy as np
from collections import OrderedDict







RPPA_256_colormaps = {"Cell membrane": "#F9836D", 
                      "Cytoplasm": "#F99B9C", 
                      "Cytosol": "#CDE7E2", 
                      "Golgi apparatus or Endoplasmic reticulum": "#FEEFBC", 
                      "Mitochondrion": "#FFE6E0", 
                      "Nucleus": "#FEBCA8", 
                      "Others": "#FEF100"
                    }
z
#import seaborn as sns
#sns.palplot(olormaps.values())

class Extraction:
    
    def __init__(self, map = 'RPPA', feature_dict = {}):
        """        
        parameters
        -----------------------
        feature_dict: dict parameters for the corresponding descriptors, say: {'Property':['MolWeight', 'MolSLogP']}
        """

        ''' orihinal
        if feature_dict == {}:
            factory = mapkey
            feature_dict = _subclass_
            self.flag = 'all'
        else:
            factory = {key:mapkey[key] for key in set(feature_dict.keys()) & set(mapkey)}
            feature_dict = feature_dict
            self.flag = 'auto'
        
        assert factory != {}, 'types of feature %s can be used' % list(mapkey.keys())
        self.factory = factory
        self.feature_dict = feature_dict
        keys = []
        for key, lst in self.feature_dict.items():
            if not lst:
                nlst = _subclass_.get(key)
            else:
                nlst = lst
            keys.extend([(v, key) for v in nlst])
        bitsinfo = pd.DataFrame(keys, columns=['IDs', 'Subtypes'])
        bitsinfo['colors'] = bitsinfo.Subtypes.map(colormaps)
        self.bitsinfo = bitsinfo
        self.colormaps = colormaps
        self.scaleinfo = load_config('descriptor','scale')
        '''
        bitsinfo = pd.read_csv("/public/home/zhangwei/scRNA-seq/csv/%s-color.csv"%map)
        bitsinfo['colors'] = bitsinfo.Subtypes.map(RPPA_256_colormaps)
        self.bitsinfo = bitsinfo
        self.colormaps = RPPA_256_colormaps
        # self.scaleinfo = pd.read_pickle("/public/home/zhangwei/MolMapNet/bidd-molmap/molmap/config/gene_scale.cfg")
        self.scaleinfo = pd.read_pickle("/public/home/zhangwei/MolMapNet/bidd-molmap/molmap/config/%s_scale.cfg"%map)

        
    def _transform_mol(self, mol):
        """
        mol" rdkit mol object
        """
        _all = OrderedDict()
        
        for key, func in self.factory.items():
            flist = self.feature_dict.get(key)
            dict_res = func(mol)
            if (self.flag == 'all') | (not flist):
                _all.update(dict_res)
            else:
                for k in flist:
                    _all.update({k:dict_res.get(k)})
        arr = np.fromiter(_all.values(), dtype=float)
        arr[np.isinf(arr)] = np.nan #convert inf value with nan     
        # try:
        #     print(arr.shape)
        # except:
        #     print("Error!!!")
        # finally:
        #     pass     
        return arr
    
    