import concurrent.futures

import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


from scene.colmap_loader import read_extrinsics_binary,read_intrinsics_binary,qvec2rotmat,load_colmap_data
from utils.graphics_utils import focal2fov, fov2focal
import sys
from typing import NamedTuple

class CameraCOLMAP(NamedTuple):
    uid: int
    focal: np.array
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    width: int
    height: int

def readColmapCameras(cam_extrinsics, cam_intrinsics,root_path):
    cam_infos = []

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        # print(intr)
        # exit()

        image_path = os.path.join(root_path, "colmap/input", os.path.basename(extr.name))
        image = Image.open(image_path)
        width = image.size[0]
        height = image.size[1]

        uid = int(os.path.basename(image_path).split(".")[0][3:])
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)


        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)

        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        cam_info = CameraCOLMAP(uid=uid,focal=intr.params, R=R, T=T, FovY=FovY, FovX=FovX,width=width,height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[
        :3
    ] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = np.concatenate(
        [poses, last_row], 1
    )  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    #     poses_centered = poses_centered  @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.0])

    for theta in np.linspace(0.0, 2.0 * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses



def process_video(video_data_save, video_path, img_wh, downsample, transform):
    """
    Load video_path data to video_data_save tensor.
    """
    video_frames = cv2.VideoCapture(video_path)
    count = 0
    video_images_path = video_path.split('.')[0]
    image_path = os.path.join(video_images_path,"images")

    if not os.path.exists(image_path):
        os.makedirs(image_path)
        while video_frames.isOpened():
            ret, video_frame = video_frames.read()
            if ret:
                video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
                video_frame = Image.fromarray(video_frame)
                if downsample != 1.0:
                    
                    img = video_frame.resize(img_wh, Image.LANCZOS)
                img.save(os.path.join(image_path,"%04d.png"%count))

                img = transform(img)
                video_data_save[count] = img.permute(1,2,0)
                count += 1
            else:
                break
      
    else:
        images_path = os.listdir(image_path)
        images_path.sort()
        
        for path in images_path:
            img = Image.open(os.path.join(image_path,path))
            if downsample != 1.0:  
                img = img.resize(img_wh, Image.LANCZOS)
                img = transform(img)
                video_data_save[count] = img.permute(1,2,0)
                count += 1
        
    video_frames.release()
    print(f"Video {video_path} processed.")
    return None


# define a function to process all videos
def process_videos(videos, skip_index, img_wh, downsample, transform, num_workers=1):
    """
    A multi-threaded function to load all videos fastly and memory-efficiently.
    To save memory, we pre-allocate a tensor to store all the images and spawn multi-threads to load the images into this tensor.
    """
    all_imgs = torch.zeros(len(videos) - 1, 300, img_wh[-1] , img_wh[-2], 3)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # start a thread for each video
        current_index = 0
        futures = []
        for index, video_path in enumerate(videos):
            # skip the video with skip_index (eval video)
            if index == skip_index:
                continue
            else:
                future = executor.submit(
                    process_video,
                    all_imgs[current_index],
                    video_path,
                    img_wh,
                    downsample,
                    transform,
                )
                futures.append(future)
                current_index += 1
    return all_imgs

def get_spiral(c2ws_all, near_fars,camera_idx, rads_scale=1.0, N_views=120):
    """
    Generate a set of poses using NeRF's spiral camera trajectory as validation poses.
    """
    # center pose
    print(c2ws_all.shape)
    c2w = average_poses(c2ws_all)
    c2w = c2ws_all[camera_idx]

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)

    # Get radii for spiral path
    zdelta = near_fars.min() * 0.2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 10, 0) * rads_scale
    render_poses = render_path_spiral(
        c2w, up, rads, focal, zdelta, zrate=0.5, N=N_views
    )
    return np.stack(render_poses)



def get_near_fars(datapath):
    poses,pts3d,perm = load_colmap_data(datapath)

    pts_arr = []
    vis_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
        cams = [0] * poses.shape[-1]
        for ind in pts3d[k].image_ids:
            if len(cams) < ind - 1:
                print('ERROR: the correct camera poses for current points cannot be accessed')
                return
            cams[ind - 1] = 1
        vis_arr.append(cams)

    pts_arr = np.array(pts_arr)
    vis_arr = np.array(vis_arr)
    print('Points', pts_arr.shape, 'Visibility', vis_arr.shape)

    zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2, 0, 1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
    valid_z = zvals[vis_arr == 1]
    print('Depth stats', valid_z.min(), valid_z.max(), valid_z.mean())

    nearfars_arr = []
    for i in perm:
        vis = vis_arr[:, i]
        zs = zvals[:, i]
        zs = zs[vis == 1]
        close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)
        nearfars_arr.append(np.array([close_depth, inf_depth]))
    nearfars_arr = np.array(nearfars_arr)

    return nearfars_arr




class Neural3D_NDC_Dataset(Dataset):
    def __init__(
        self,
        datadir,
        split="train",
        downsample=1.0,
        is_stack=True,
        cal_fine_bbox=False,
        N_vis=-1,
        time_scale=1.0,
        scene_bbox_min=[-1.0, -1.0, -1.0],
        scene_bbox_max=[1.0, 1.0, 1.0],
        N_random_pose=1000,
        bd_factor=0.75,
        eval_step=1,
        eval_index=0,
        is_short = False,
        sphere_scale=1.0,
    ):
        self.img_wh = (
            int(1920 / downsample),
            int(1014 / downsample),
        )  # According to the neural 3D paper, the default resolution is 1024x768
        self.root_dir = datadir
        self.split = split
        # self.downsample = 2704 / self.img_wh[0]
        self.is_stack = is_stack
        self.N_vis = N_vis
        self.time_scale = time_scale
        self.scene_bbox = torch.tensor([scene_bbox_min, scene_bbox_max])

        self.world_bound_scale = 1.1
        self.bd_factor = bd_factor
        self.eval_step = eval_step
        self.eval_index = eval_index
        self.blender2opencv = np.eye(4)
        self.transform = T.ToTensor()

        self.near = 0.0
        self.far = 1.0
        self.near_far = [self.near, self.far]  # NDC near far is [0, 1.0]
        self.white_bg = False
        self.ndc_ray = True
        self.depth_data = False
        self.is_short = is_short


        self.load_meta()
        print(f"meta data loaded, total image:{len(self)}")

    def load_meta(self):
        """
        Load meta data from the dataset.
        """
        # Read poses and video file paths.
        # poses_arr = np.load(os.path.join(self.root_dir, "poses_bounds.npy"))[:12,:]
        # poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)


        # poses_arr1 = np.load(os.path.join(self.root_dir, "poses_bounds.npy"))#[:12, :]

        cameras_extrinsic_file = os.path.join(self.root_dir, "colmap/sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(self.root_dir, "colmap/sparse/0", "cameras.bin")

        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)


        cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,root_path=self.root_dir)

        self.cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: int(x.uid))


        self.near_fars = get_near_fars(os.path.join(self.root_dir, "colmap/sparse/0"))

        self.height = self.cam_infos[0].height
        self.width = self.cam_infos[0].width

        poses = []
        for i,key in enumerate(self.cam_infos):
            poses.append(np.concatenate((self.cam_infos[i].R,self.cam_infos[i].T[:,np.newaxis]),1))
        poses = np.stack(poses)


        videos = glob.glob(os.path.join(self.root_dir, "cam*"))
        videos = sorted(videos)
        assert len(videos) == poses.shape[0]


        near_original = self.near_fars.min()
        scale_factor = near_original * 0.75
        self.near_fars *= (
            scale_factor  # rescale nearest plane so that it is at z = 4/3.
        )

        N_views = 120
        self.val_poses = get_spiral(poses, self.near_fars,self.eval_index, N_views=N_views)
        # self.val_poses = self.directions
        W, H = self.img_wh
        poses_i_train = []

        for i in range(len(poses)):
            if i != self.eval_index:
                poses_i_train.append(i)
        self.poses = poses[poses_i_train]
        self.poses_all = poses
        # self.image_paths, self.image_poses, self.image_times, N_cam, N_time = self.load_images_path(videos, self.split)
        self.image_paths, self.flow_paths, self.image_poses,self.focal, self.image_times,self.per_image_times, N_cam, N_time = self.load_images_path(
            self.root_dir, videos, self.split)

        self.cam_number = N_cam
        self.time_number = N_time
    def get_val_pose(self):
        render_poses = self.val_poses
        render_times = torch.linspace(0.0, 1.0, render_poses.shape[0]) * 2.0 - 1.0
        return render_poses, self.time_scale * render_times
    def load_images_path(self,root_dir,videos,split):
        image_paths = []
        flow_paths = []
        image_poses = []
        image_times = []
        per_image_times = []
        focals = []
        N_cams = 0
        N_time = 0
        # countss = 100

        for index, video_path in enumerate(videos):
            
            if index == self.eval_index:
                if split =="train":
                    continue
            else:
                if split == "test":
                    continue
            N_cams +=1
            count = 0
            video_images_path = video_path.split('.')[0]
            image_path = os.path.join(video_images_path,"images") #
            flow_path = os.path.join(video_images_path, "flow")
            if self.is_short:
                countss = 100
            else:
                countss = len(os.listdir(os.path.join(video_images_path, "images")))

                    
            images_name = os.listdir(image_path)
            flows_name = os.listdir(flow_path)

            images_name.sort(key=lambda x: int(x[:-4]))
            flows_name.sort(key=lambda x: int(x[:-4]))

            this_count = 0
            for idx, path in enumerate(images_name):
                if this_count >=countss:break
                image_paths.append(os.path.join(image_path,path))
                flow_paths.append(os.path.join(flow_path, flows_name[idx]))

                R = self.cam_infos[index].R
                T = self.cam_infos[index].T
                image_poses.append((R, T))
                focals.append(self.cam_infos[index].focal)

                image_times.append(idx / countss)
                if idx == 0:
                    per_image_times.append(idx / countss)
                else:
                    per_image_times.append((idx - 1) / countss)

                this_count+=1
            N_time = this_count


        return image_paths,flow_paths, image_poses,focals, image_times,per_image_times, N_cams, N_time
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self,index):

        img = Image.open(self.image_paths[index])
        img = self.transform(img)
        flow = np.load(self.flow_paths[index])

        return img, flow, self.image_poses[index], self.focal[index], self.image_times[index], self.per_image_times[index]
    def load_pose(self,index):
        return self.image_poses[index]

