import os
from typing import NamedTuple
from scene.colmap_loader import read_points3D_binary
from scene.external import storePly
import torchvision.transforms as transforms
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
from plyfile import PlyData, PlyElement
from scene.gaussian_model import BasicPointCloud
from tqdm import tqdm


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    flow: np.array
    focal: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time : float
    per_time : float
   
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    video_cameras: list
    nerf_normalization: dict
    ply_path: str
    maxtime: int

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def format_infos(dataset,split):
    # loading
    cameras = []
    image = dataset[0][0]
    flow = dataset[0][1]
    focal = dataset[0][2]
    width = image.shape[2]
    height = image.shape[1]
    if split == "train":
        for idx in tqdm(range(len(dataset))):
            image_path = None
            image_name = f"{idx}"
            time = dataset.image_times[idx]
            per_time = dataset.per_image_times[idx]
            focal = dataset.focal[idx]
            R, T = dataset.load_pose(idx)       
            
            FovX = focal2fov(focal[0], width)
            FovY = focal2fov(focal[1], height)
            
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,flow=flow,focal = focal,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                time = time,per_time = per_time))

    return cameras


def format_render_poses(poses,data_infos):
    cameras = []
    tensor_to_pil = transforms.ToPILImage()
    len_poses = len(poses)
    times = [i/len_poses for i in range(len_poses)]
    image = data_infos[0][0]
    flow = data_infos[0][1]
    focal = data_infos[0][3]
    width = image.shape[2]
    height = image.shape[1]

    for idx, p in tqdm(enumerate(poses)):

        image_path = None
        image_name = f"{idx}"
        time = times[idx]
        pose = np.eye(4)
        pose[:3,:] = p[:3,:]
        R = pose[:3,:3]
        R[:,0] = -R[:,0]
        T = pose[:3,3]
        FovX = focal2fov(focal[0], width)
        FovY = focal2fov(focal[1], height)

        cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,flow = flow,focal=focal,
                            image_path=image_path, image_name=image_name, width=width, height=height,
                            time = time,per_time=0))
    return cameras


def readdynerfInfo(datadir,args):
    # loading all the data follow hexplane format
    # ply_path = os.path.join(datadir, "points3d.ply")
    ply_path = os.path.join(datadir, 'colmap/sparse/0/points3D.ply')
    bin_path = os.path.join(datadir, 'colmap/sparse/0/points3D.bin')

    from scene.neural_3D_dataset_NDC import Neural3D_NDC_Dataset
    train_dataset = Neural3D_NDC_Dataset(
    datadir,
    "train",
    1.0,
    time_scale=1,
    scene_bbox_min=[-2.5, -2.0, -1.0],
    scene_bbox_max=[2.5, 2.0, 1.0],
    eval_index=args.eval_index,
    is_short = args.is_short
    )

    test_dataset = Neural3D_NDC_Dataset(
    datadir,
    "test",
    1.0,
    time_scale=1,
    scene_bbox_min=[-2.5, -2.0, -1.0],
    scene_bbox_max=[2.5, 2.0, 1.0],
    eval_index=args.eval_index,
    is_short = args.is_short
    )

    train_cam_infos = format_infos(train_dataset,"train")

    val_cam_infos = format_render_poses(test_dataset.val_poses,test_dataset)
    nerf_normalization = getNerfppNorm(train_cam_infos)

    xyz, rgb, _ = read_points3D_binary(bin_path)
    storePly(ply_path, xyz, rgb)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_dataset,
                           test_cameras=test_dataset,
                           video_cameras=val_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=300
                           )
    return scene_info

sceneLoadTypeCallbacks = {
    "dynerf" : readdynerfInfo,
}