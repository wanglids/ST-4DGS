
import numpy as np
import random
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch

from random import randint
from utils.loss_utils import l1_loss, ssim,lpips_loss,scale_loss,weighted_l2_loss_v2
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from scene.external import get_sample_point,warp_point,get_add_point,quat_mult,build_rotation,o3d_knn
from scene.getData import get_model_data
import lpips
from utils.scene_utils import render_training_image


to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter,timer):
    first_iter = 0
    viewpoint_stack = None

    if checkpoint:
        first_iter = checkpoint_iterations[0]
        gaussians.restore(os.path.join(dataset.source_path,dataset.model_path),checkpoint_iterations[0])


    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)


    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    iter_finetune = 0.75

    final_iter = train_iter
    
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    lpips_model = lpips.LPIPS(net="alex").cuda()
    video_cams = scene.getVideoCameras()
    for iteration in range(first_iter, final_iter+1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer, ts = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, stage="stage")["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras()
            batch_size = 1
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,shuffle=True,num_workers=0,collate_fn=list)
            loader = iter(viewpoint_stack_loader)
        if opt.dataloader:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                print("reset dataloader")
                batch_size = 1
                loader = iter(viewpoint_stack_loader)
        else:
            idx = randint(0, len(viewpoint_stack)-1)
            viewpoint_cams = [viewpoint_stack[idx]]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        gt_images = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            images.append(image.unsqueeze(0))
            gt_image = viewpoint_cam.original_image.cuda()
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
        

        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor = torch.cat(images,0)
        gt_image_tensor = torch.cat(gt_images,0)
        # Loss
        losses = {}
        losses['image'] = l1_loss(image_tensor, gt_image_tensor)



        if stage == "fine" and hyper.time_smoothness_weight != 0:
            losses['tv'] = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.plane_tv_weight, hyper.l1_time_planes)

        # if opt.lambda_dssim != 0 and iteration > opt.densify_until_iter:
        #     losses['simm'] = 1.0-ssim(image_tensor,gt_image_tensor)

        # if opt.lambda_lpips !=0:
        #     losses['lpips'] = lpips_loss(image_tensor,gt_image_tensor,lpips_model)

        if opt.lambda_ani != 0:
            losses['ani'] = scale_loss(gaussians.get_scaling, thr=10.0)

        if stage == "fine" and iteration > final_iter*iter_finetune and viewpoint_cams[0].time!=0:

            with torch.no_grad():

                move_mask = torch.zeros((gaussians.get_xyz.shape[0]),
                                          dtype=torch.float,device="cuda")

                flow_samplers = []
                sampler_masks = []

                for viewpoint in viewpoint_cams:
                    per_time = torch.tensor(viewpoint.per_time).to(gaussians.get_xyz.device).repeat(
                        gaussians.get_xyz.shape[0], 1)
                    per_xyz, per_scales, per_rotations, per_opacity = get_model_data(gaussians, per_time,
                                                                                     stage="get_data")

                    mask, sampler, sampler_mask, flow_sampler, flow_mask = get_sample_point(viewpoint,per_xyz)
                    sampler_masks.append(sampler_mask)

                    flow_samplers.append(flow_sampler)

                    move_mask += flow_mask

                move_mask[move_mask!=0] = 1
                for fm in range(len(flow_samplers)):
                    flow_samplers[fm][move_mask==0] = 0


            if opt.lambda_tem != 0:
                segs = []
                for viewpoint_cam in viewpoint_cams:
                    seg = warp_point(viewpoint_cam, per_xyz)
                    seg[move_mask == 0] = 0
                    segs.append(seg)

                segs_all = torch.cat(segs,0)
                flow_samplers_all = torch.cat(flow_samplers,0)
                losses['tem'] = l1_loss(segs_all, flow_samplers_all)


            if opt.lambda_loc != 0:

                fg_pts = gaussians.get_xyz[move_mask>0.5]
                fg_rot = gaussians.get_rotation[move_mask>0.5]

                prev_inv_rot_fg = per_rotations[move_mask>0.5].detach()
                per_fg_xyz = per_xyz[move_mask>0.5].detach()

                if per_fg_xyz.shape[0] == 0:
                    losses['loc'] = 0
                else:
                    rel_rot = quat_mult(fg_rot, prev_inv_rot_fg)
                    rot = build_rotation(rel_rot)

                    neighbor_sq_dist, neighbor_indice = o3d_knn(per_fg_xyz.cpu().numpy(), 20)
                    neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
                    neighbor_dist = np.sqrt(neighbor_sq_dist)

                    neighbor_indices = torch.tensor(neighbor_indice).cuda().long().contiguous()
                    neighbor_weight = torch.tensor(neighbor_weight).cuda().float().contiguous()
                    neighbor_pts = fg_pts[neighbor_indices]

                    curr_offset = neighbor_pts - fg_pts[:, None]
                    curr_offset_in_prev_coord = (rot.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]).squeeze(-1)

                    prev_offset = per_fg_xyz[neighbor_indices] - per_fg_xyz[:, None]

                    loss_rigid = weighted_l2_loss_v2(curr_offset_in_prev_coord, prev_offset,
                                                        neighbor_weight)

                    loss_rot = weighted_l2_loss_v2(rel_rot[neighbor_indices], rel_rot[:, None],
                                                        neighbor_weight)

                    losses['loc'] = loss_rigid+loss_rot




        loss_weights = {'image': 1.0, 'tv': 1.0, 'simm': opt.lambda_dssim, 'lpips': opt.lambda_lpips, 'ani': opt.lambda_ani,
                        'tem': opt.lambda_tem,'loc': opt.lambda_loc}

        loss = sum([loss_weights[k] * v for k, v in losses.items()])
        loss.backward()

        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            # Progress bar

            psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            timer.pause()
            training_report(tb_writer, iteration, losses['image'], loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background], stage)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 1) \
                    or (iteration < 3000 and iteration % 50 == 1) \
                        or (iteration < 10000 and iteration %  100 == 1) \
                            or (iteration < 60000 and iteration % 100 ==1):

                    render_training_image(scene, gaussians, video_cams, render, pipe, background, stage, iteration-1,timer.get_elapsed_time())

            timer.start()

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    print("reset opacity")
                    gaussians.reset_opacity()

                if stage == "coarse":
                    if iteration > opt.densify_from_iter and iteration % 500 == 0:
                        gaussians.geometry_prune_point(opt.coarse_neighbors,opt.coarse_std)
                else:
                    if iteration > 1500 and iteration % 1000 == 0 and iteration < opt.densify_until_iter-3000:
                        gaussians.geometry_prune_point(opt.fine_neighbors,opt.fine_std)

            if (iteration == 6000 or iteration == 12000) and stage == "fine":
                for viewpoint_cam in viewpoint_cams:
                    # with torch.no_grad():
                    t_time = torch.tensor(viewpoint_cams[0].per_time).to(gaussians.get_xyz.device).repeat(gaussians.get_xyz.shape[0], 1) 
                    add_xyz,_,_,_= get_model_data(gaussians, t_time, stage="get_data")
                    near_mask = get_add_point(viewpoint_cam,add_xyz)
                    gaussians.motion_splitting(near_mask)



            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname):
    # first_iter = 0
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=None)
    timer.start()
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                             checkpoint_iterations, checkpoint, debug_from,
                             gaussians, scene, "coarse", tb_writer, opt.coarse_iterations,timer)
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, "fine", tb_writer, opt.iterations,timer)

def prepare_output_and_logger(expname):    
    if not args.model_path:
        unique_str = expname
        args.model_path = os.path.join("./output/", unique_str)

    # Set up output folder
    print("Output folder: {}".format(os.path.join(args.source_path,args.model_path)))
    os.makedirs(os.path.join(args.source_path,args.model_path), exist_ok = True)
    with open(os.path.join(os.path.join(args.source_path,args.model_path), "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(os.path.join(args.source_path,args.model_path))
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, stage):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)
        
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(stage + "/"+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        
        torch.cuda.empty_cache()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # Set up command line argument parser
    # torch.set_default_tensor_type('torch.FloatTensor')
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i*500 for i in range(0,120)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3000,15000, 20000, 30_000,45000,60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", action="store_true")
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")

    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmengine
        from utils.params_utils import merge_hparams
        config = mmengine.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname)

    # All done
    print("\nTraining complete.")
