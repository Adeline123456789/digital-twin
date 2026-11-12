"""Small adapter to Kornia's existing LoFTR, returning RGB points only.

Run in the existing tote_insert Python with .local/loftr_runtime on PYTHONPATH.
The downloaded checkpoint and image-derived arrays stay outside public Git.
"""
import numpy as np
import torch
from kornia.feature import LoFTR


def load_matcher(checkpoint):
    model=LoFTR(pretrained=None)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['state_dict'])
    return model.eval()


def match_region(model, reference, target, mask):
    # Keep the full RGB context; select ROI observations after matching.
    with torch.inference_mode():
        result=model({name:torch.from_numpy(image).float()[None,None]/255
                      for name,image in [('image0',reference),('image1',target)]})
    p=result['keypoints0'].cpu().numpy();q=result['keypoints1'].cpu().numpy()
    confidence=result['confidence'].cpu().numpy()
    xy=np.rint(p).astype(int)
    keep=mask[xy[:,1],xy[:,0]]>0
    return p[keep],q[keep],confidence[keep]
