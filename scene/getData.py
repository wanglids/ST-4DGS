import torch
from scene.gaussian_model import GaussianModel

def get_model_data( pc : GaussianModel, time, stage="fine"):
    means3D = pc.get_xyz
    opacity = pc._opacity

    scales = pc._scaling
    rotations = pc._rotation
    deformation_point = pc._deformation_table

    if stage == "coarse" :
        means3D_deform, scales_deform, rotations_deform, opacity_deform = means3D, scales, rotations, opacity
    else:
        means3D_deform, scales_deform, rotations_deform, opacity_deform = pc._deformation(means3D[deformation_point], scales[deformation_point],
                                                                         rotations[deformation_point], opacity[deformation_point],
                                                                         time[deformation_point])

    if stage == "fine":
        with torch.no_grad():
            pc._deformation_accum[deformation_point] += torch.abs(means3D_deform-means3D[deformation_point])

    means3D_final = torch.zeros_like(means3D)
    rotations_final = torch.zeros_like(rotations)
    scales_final = torch.zeros_like(scales)
    opacity_final = torch.zeros_like(opacity)
    means3D_final[deformation_point] =  means3D_deform
    rotations_final[deformation_point] =  rotations_deform
    scales_final[deformation_point] =  scales_deform
    opacity_final[deformation_point] = opacity_deform
    means3D_final[~deformation_point] = means3D[~deformation_point]
    rotations_final[~deformation_point] = rotations[~deformation_point]
    scales_final[~deformation_point] = scales[~deformation_point]
    opacity_final[~deformation_point] = opacity[~deformation_point]

    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity)

    return means3D_final,scales_final,rotations_final,opacity