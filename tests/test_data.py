from pathlib import Path
import numpy as np
from skimage import io

from ctrobust.data import FolderSliceDataset


def test_folder_dataset(tmp_path: Path):
    for cls in ("benign","malignant"):
        d=tmp_path/cls; d.mkdir()
        io.imsave(d/"x.png",(np.ones((8,8))*127).astype(np.uint8),check_contrast=False)
    ds=FolderSliceDataset(tmp_path,image_size=16,input_mode="normalized")
    assert len(ds)==2
    x,y=ds[0]
    assert x.shape==(1,16,16)
    assert y.item() in (0,1)
