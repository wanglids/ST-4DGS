from torch.utils.data import Dataset
from scene.cameras import Camera
import torch
from utils.graphics_utils import focal2fov
class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args
    ):
        self.dataset = dataset
        self.args = args
    def __getitem__(self, index):


        image, flow, w2c, focal, time, per_time = self.dataset[index]
        R,T = w2c

        width = image.shape[2]
        height = image.shape[1]

        focal_length_x = focal[0]
        focal_length_y = focal[1]

        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)

        return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,flow=flow,focal = focal,gt_alpha_mask=None,
                          image_name=f"{index}",uid=index,data_device=torch.device("cuda"),time=time,per_time = per_time)
    def __len__(self):
        
        return len(self.dataset)
