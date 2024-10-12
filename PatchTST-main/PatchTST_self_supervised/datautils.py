

import numpy as np
import pandas as pd
import torch
from torch import nn
import sys

from src.data.datamodule import DataLoaders
from src.data.pred_dataset import *

DSETS = ['DE', 'IT', 'FR', 'GB', 'ES']

def get_dls(params):
    
    assert params.dset in DSETS, f"Unrecognized dset (`{params.dset}`). Options include: {DSETS}"
    if not hasattr(params,'use_time_features'): params.use_time_features = False
    
    df_name = f'{params.dset}_data.csv'
    root_path = './datasets/'
    size = [params.context_points, 0, params.target_points]
    dls = DataLoaders(
            datasetCls=Dataset_Custom,
            dataset_kwargs={
            'root_path': root_path,
            'data_path': df_name,
            'features': params.features,
            'scale': True,
            'size': size,
            'use_time_features': params.use_time_features,
            'overlapping_windows': params.overlapping_windows,  # New argument
            'scaler_type': params.scaler_type                   # New argument
            },
            batch_size=params.batch_size,
            workers=params.num_workers,
            )
        
    # dataset is assume to have dimension len x nvars
    dls.vars, dls.len = dls.train.dataset[0][0].shape[1], params.context_points
    dls.c = dls.train.dataset[0][1].shape[0]
    return dls



if __name__ == "__main__":
    class Params:
        dset= 'DE'
        context_points= 384
        target_points= 96
        batch_size= 64
        num_workers= 8
        with_ray= False
        features='M'
        overlapping_windows = True  
        scaler_type = 'standard'   
    params = Params 
    dls = get_dls(params)
    for i, batch in enumerate(dls.valid):
        print(i, len(batch), batch[0].shape, batch[1].shape)
    breakpoint()
