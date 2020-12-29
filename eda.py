#%%
import os
import re
import pandas as pd
import numpy as np
DATA_DIR = '../ranzcr-clip-catheter-line-classification'

# %%
def get_best_checkpoint(dir, score_pattern=r"auc=(0\.\d*)"):
    chekcpoints = []
    scores= []
    for root, _, files in os.walk(dir):
        for file in files:
            if '.ckpt' in file: 
                chekcpoints.append(os.path.join(root, file))
                result = re.search(score_pattern, file)
                if result: scores.append(float(result.group(0).split('=')[1]))
                else: scores.append(0.0)
    assert len(chekcpoints) == len(scores)
    if chekcpoints: return chekcpoints[np.argmax(scores)]         
    else: return None
#%%
#get_best_checkpoint('raznr', r"val_auc=(0\.\d*)")
# %%
