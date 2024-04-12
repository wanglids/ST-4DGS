import copy
import numpy as np
import cv2
import torch
import open3d as o3d
from scene.KDTree import KDTree
from plyfile import PlyElement,PlyData
import os


def build_rotation(q):
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3), device='cuda')
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def world_to_view_screen(pts3D, K, RT_cam2):
    # print("pts3D",pts3D.max(), pts3D.min())
    wrld_X = RT_cam2.bmm(pts3D)
    xy_proj = K.bmm(wrld_X)


    # And finally we project to get the final result
    mask = (xy_proj[:, 2:3, :].abs() < 1E-2).detach()
    mask = mask.to(pts3D.device)
    mask.requires_grad_(False)

    zs = xy_proj[:, 2:3, :]

    mask_unsq = mask.unsqueeze(0).unsqueeze(0)

    if True in mask_unsq:
        zs[mask] = 1E-2
    sampler = torch.cat((xy_proj[:, 0:2, :] / zs, xy_proj[:, 2:3, :]), 1)

    # Remove invalid zs that cause nans
    if True in mask_unsq:
        sampler[mask.repeat(1, 3, 1)] = -10
    return sampler

def get_pixel_grids(height, width):
    with torch.no_grad():
        # texture coordinate
        x_linspace = torch.linspace(0, width - 1, width).view(1, width).expand(height, width)
        y_linspace = torch.linspace(0, height - 1, height).view(height, 1).expand(height, width)
        x_coordinates = x_linspace.contiguous().view(-1)
        y_coordinates = y_linspace.contiguous().view(-1)
        ones = torch.ones(height * width)
        indices_grid = torch.stack([x_coordinates, y_coordinates, ones, torch.ones(height * width)], dim=0)
    return indices_grid

def my_view_to_world_coord(pts3D, K_inv, RTinv_cam1, xyzs):
    # PERFORM PROJECTION
    # Project the world points into the new view
    projected_coors = xyzs * pts3D
    projected_coors[:, -1, :] = 1
    cam1_X = K_inv.bmm(projected_coors)
    wrld_X = RTinv_cam1.bmm(cam1_X)
    return wrld_X

def get_add_point(view_camera,xyz):
    # Motion-aware Splitting. Establish a connection between candidate points and Gaussian.


    temp_R = copy.deepcopy(view_camera.R)
    temp_T = copy.deepcopy(view_camera.T)

    temp_R = np.transpose(temp_R)
    R = np.eye(4)
    R[:3, :3] = temp_R
    R[:3, 3] = temp_T

    H, W = view_camera.original_image.shape[1], view_camera.original_image.shape[2]

    K = np.eye(4)
    K[0, 2] = W / 2
    K[1, 2] = H / 2
    K[0, 0] = view_camera.focal[0]
    K[1, 1] = view_camera.focal[1]

    
    K = torch.FloatTensor(K).unsqueeze(0).cuda()
    R = torch.FloatTensor(R).unsqueeze(0).cuda()

    src_xyz_t = xyz
    src_xyz_t = src_xyz_t.unsqueeze(0).permute(0, 2, 1)
    tempdata = torch.ones((src_xyz_t.shape[0], 1, src_xyz_t.shape[2])).cuda()
    src_xyz = torch.cat((src_xyz_t, tempdata), dim=1)

    xyz_sampler = world_to_view_screen(src_xyz, RT_cam2=R, K=K)


    sampler = xyz_sampler[0, 0:2].transpose(1, 0)
    depth_sampler = xyz_sampler[0, 2:].transpose(1, 0)

    sampler_t = sampler.detach().cpu().numpy().astype(int)
    sampler_mask = np.ones((sampler_t.shape[0],1))

    sampler_mask[sampler_t[:, 1] >= H] = 0
    sampler_mask[sampler_t[:, 0] >= W] = 0
    sampler_mask[sampler_t[:, 1] < 0] = 0
    sampler_mask[sampler_t[:, 0] < 0] = 0

    sampler_t[sampler_t[:, 1] >= H, 1] = H - 1
    sampler_t[sampler_t[:, 0] >= W, 0] = W - 1
    sampler_t[sampler_t<0] = 0

    sampler_w = np.zeros_like(sampler_t)
    sampler_w[:, 0] = sampler_t[:, 1]
    sampler_w[:, 1] = sampler_t[:, 0]

    mask = np.zeros((H,W))
    mask[sampler_t[:,1],sampler_t[:,0]] = 255

    x_linspace = torch.linspace(0, W - 1, W).view(1, W).expand(H, W)
    y_linspace = torch.linspace(0, H - 1, H).view(H, 1).expand(H, W)
    xyzs_big = torch.stack([y_linspace, x_linspace], dim=2)  # H W 2

    flow_t = copy.deepcopy(view_camera.flow)

    flow_p = np.sum(flow_t,axis=2)
    flow_p[flow_p>0.5] = 1
    flow_p[flow_p<-0.5] = 1
    flow_p[flow_p!=1] = 0
    flow_p = np.array(flow_p,dtype= np.uint8)  # motion regions
    
    kernel=np.ones((5,5),np.uint8)
    flow_p=cv2.dilate(flow_p,kernel,iterations=1) # Morphological operation


    flow_mask = flow_p[sampler_t[:, 1], sampler_t[:, 0]]

    wind = 3

    image1 = np.zeros((H,W))
    image2 = np.zeros((H,W))
    x_linspace1 = np.linspace(0,H-1,int(H/wind)-1).astype(int)
    y_linspace1 = np.linspace(0,W-1,int(W/wind)-1).astype(int)


    image1[x_linspace1,:] = 1
    image2[:,y_linspace1] = 1
    image = image1+image2
    image_indx = image[flow_p!=0]

    xyz_mask = xyzs_big[flow_p!=0]
    sampler_mask = xyz_mask[image_indx==2]
    sampler_w = sampler_w[flow_mask!=0]

    sampler_w = np.array(sampler_w)
    sampler_mask = np.array(sampler_mask)   

    num_train = sampler_mask.shape[0]

    kdtree = KDTree(sampler_w)
    root = kdtree.createKDTree(sampler_w, 0)

    near_point_mask = torch.zeros((num_train),dtype = torch.long) 
    for i in range(num_train):
        pt, minDis = kdtree.getNearestPt(root, sampler_mask[i])
        index1,_ = np.where(sampler_w == pt)
        near_point_mask[i] = index1[0]

    return near_point_mask



def get_sample_point(view_camera,xyz):

    temp_R = copy.deepcopy(view_camera.R)
    temp_T = copy.deepcopy(view_camera.T)

    temp_R = np.transpose(temp_R)
    R = np.eye(4)
    R[:3, :3] = temp_R
    R[:3, 3] = temp_T

    H, W = view_camera.original_image.shape[1], view_camera.original_image.shape[2]

    K = np.eye(4)
    K[0, 2] = view_camera.focal[2]
    K[1, 2] = view_camera.focal[3]
    K[0, 0] = view_camera.focal[1]
    K[1, 1] = view_camera.focal[0]

    src_xyz_t = xyz
    src_xyz_t = src_xyz_t.unsqueeze(0).permute(0, 2, 1)
    tempdata = torch.ones((src_xyz_t.shape[0], 1, src_xyz_t.shape[2])).cuda()
    src_xyz = torch.cat((src_xyz_t, tempdata), dim=1)

    K = torch.FloatTensor(K).unsqueeze(0).cuda()
    R = torch.FloatTensor(R).unsqueeze(0).cuda()

    xyz_sampler = world_to_view_screen(src_xyz, RT_cam2=R, K=K)

    sampler = xyz_sampler[0, 0:2].transpose(1, 0)
    temp_depth = xyz_sampler[:,2].transpose(1, 0)




    sampler_t = sampler.detach().cpu().numpy().astype(int)
    sampler_mask = np.ones((sampler_t.shape[0],1))
    sampler_mask[sampler_t[:, 1] >= H] = 0
    sampler_mask[sampler_t[:, 0] >= W] = 0
    sampler_mask[sampler_t[:, 1] < 0] = 0
    sampler_mask[sampler_t[:, 0] < 0] = 0

    sampler_t[sampler_t[:, 1] >= H, 1] = H - 1
    sampler_t[sampler_t[:, 0] >= W, 0] = W - 1
    sampler_t[sampler_t<0] = 0

    mask = np.zeros((H,W))
    mask[sampler_t[:,1],sampler_t[:,0]] = 255

    x_linspace = torch.linspace(0, W - 1, W).view(1, W).expand(H, W)
    y_linspace = torch.linspace(0, H - 1, H).view(H, 1).expand(H, W)
    xyzs_big = torch.stack([y_linspace, x_linspace], dim=2)

    flow_t = copy.deepcopy(view_camera.flow)

    flow_axis = flow_t+xyzs_big.numpy()

    flow_p = np.sum(flow_t,axis=2)
    flow_p[flow_p>0.5] = 1
    flow_p[flow_p<-0.5] = 1
    flow_p[flow_p!=1] = 0
    flow_mask = flow_p[sampler_t[:, 1], sampler_t[:, 0]]

    flow_sampler = flow_axis[sampler_t[:, 1], sampler_t[:, 0]]

    return mask,sampler_t,sampler_mask,torch.tensor(flow_sampler,dtype=torch.float,device="cuda"),torch.tensor(flow_mask,dtype=torch.float,device="cuda")

  

def warp_point(view_camera, xyz):
    temp_R = copy.deepcopy(view_camera.R)
    temp_T = copy.deepcopy(view_camera.T)

    temp_T[0] *= -1


    R = np.eye(4)
    R[:3, :3] = temp_R
    R[:3, 3] = temp_T
    R = np.linalg.inv(R)
    H, W = view_camera.original_image.shape[1], view_camera.original_image.shape[2]

    K = np.eye(4)
    K[0, 2] = W / 2
    K[1, 2] = H / 2
    K[0, 0] = view_camera.focal[0]
    K[1, 1] = view_camera.focal[1]

    src_xyz_t = xyz
    src_xyz_t = src_xyz_t.unsqueeze(0).permute(0, 2, 1)
    tempdata = torch.ones((src_xyz_t.shape[0], 1, src_xyz_t.shape[2])).cuda()
    src_xyz = torch.cat((src_xyz_t, tempdata), dim=1)

    K = torch.FloatTensor(K).unsqueeze(0).cuda()
    R = torch.FloatTensor(R).unsqueeze(0).cuda()

    xyz_sampler = world_to_view_screen(src_xyz, RT_cam2=R, K=K)

    sampler = xyz_sampler[0, 0:2].transpose(1, 0)

    sampler_t = torch.zeros_like(sampler)
    sampler_t[:,0] = sampler[:,1]
    sampler_t[:,1] = sampler[:,0]


    return sampler_t


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T

def o3d_knn(pts, num_knn):
    indices = []
    sq_dists = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for p in pcd.points:
        [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)
        indices.append(i[1:])
        sq_dists.append(d[1:])
    return np.array(sq_dists), np.array(indices)


def calculate_total_size_of_files(folders):
    total_size = 0

    for folder_name in folders:
        deformation_path = os.path.join(folder_name, "./point_cloud/coarse_iteration_3000/deformation.pth")
        point_cloud_path = os.path.join(folder_name, "./point_cloud/coarse_iteration_3000/point_cloud.ply")
        # print(point_cloud_path)
        if os.path.exists(deformation_path):
            deformation_size = os.path.getsize(deformation_path) / (1024 * 1024)
            total_size += deformation_size

        if os.path.exists(point_cloud_path):
            point_cloud_size = os.path.getsize(point_cloud_path) / (1024 * 1024)
            total_size += point_cloud_size

    return total_size
