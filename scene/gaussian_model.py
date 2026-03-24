#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
import json
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scipy.spatial.transform import Rotation as R
from utils.pose_utils import rotation2quad, get_camera_from_tensor, get_tensor_from_camera
from utils.graphics_utils import fov2focal, getWorld2View2
from utils.semantic_utils import append_semantic_fields, build_selected_mask, estimate_plane_outline, slugify_label
from scene.per_point_adam import PerPointAdam


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.semantic_positive_views = torch.empty(0, dtype=torch.int32, device="cuda")
        self.semantic_visible_views = torch.empty(0, dtype=torch.int32, device="cuda")
        self.semantic_label_name = ""
        self.semantic_label_slug = ""
        self.semantic_min_visible_views = 1
        self.semantic_min_positive_views = 1
        self.semantic_min_score = 0.5
        self.semantic_vote_min_depth = 1e-4
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.P,
            self.semantic_positive_views,
            self.semantic_visible_views,
            self.semantic_label_name,
            self.semantic_label_slug,
            self.semantic_min_visible_views,
            self.semantic_min_positive_views,
            self.semantic_min_score,
            self.semantic_vote_min_depth,
        )

    def restore(self, model_args, training_args):
        if len(model_args) == 13:
            (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            self.P) = model_args
            self._ensure_semantic_state(self._xyz.shape[0])
        else:
            (self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
            self.P,
            self.semantic_positive_views,
            self.semantic_visible_views,
            self.semantic_label_name,
            self.semantic_label_slug,
            self.semantic_min_visible_views,
            self.semantic_min_positive_views,
            self.semantic_min_score,
            self.semantic_vote_min_depth) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def configure_semantics(self, label_name: str, min_visible_views: int, min_positive_views: int, min_score: float, min_depth: float):
        self.semantic_label_name = label_name or ""
        self.semantic_label_slug = slugify_label(self.semantic_label_name) if self.semantic_label_name else ""
        self.semantic_min_visible_views = int(min_visible_views)
        self.semantic_min_positive_views = int(min_positive_views)
        self.semantic_min_score = float(min_score)
        self.semantic_vote_min_depth = float(min_depth)
        self._ensure_semantic_state(self.get_xyz.shape[0])

    def _ensure_semantic_state(self, count: int):
        if self.semantic_positive_views.numel() == count and self.semantic_visible_views.numel() == count:
            return
        device = self.get_xyz.device if self.get_xyz.numel() > 0 else torch.device("cuda")
        self.semantic_positive_views = torch.zeros((count,), dtype=torch.int32, device=device)
        self.semantic_visible_views = torch.zeros((count,), dtype=torch.int32, device=device)

    def _semantic_selected_mask(self):
        positive_views = self.semantic_positive_views.detach().cpu().numpy()
        visible_views = self.semantic_visible_views.detach().cpu().numpy()
        return build_selected_mask(
            positive_views=positive_views,
            visible_views=visible_views,
            min_visible_views=self.semantic_min_visible_views,
            min_positive_views=self.semantic_min_positive_views,
            min_score=self.semantic_min_score,
        )

    def accumulate_semantic_votes(self, camera, pose, visibility_filter, mask_threshold: int):
        if not self.semantic_label_name or camera.semantic_mask is None:
            return

        visible_indices = torch.nonzero(visibility_filter, as_tuple=False).squeeze(-1)
        if visible_indices.numel() == 0:
            return

        camera_rt = get_camera_from_tensor(pose.detach())
        c2w = torch.linalg.inv(camera_rt)
        camera_position = c2w[:3, 3]
        camera_rotation = c2w[:3, :3]

        xyz = self._xyz.detach()[visible_indices]
        xyz_cam = torch.matmul(xyz - camera_position, camera_rotation.transpose(0, 1))
        depth = xyz_cam[:, 2]
        valid = depth > self.semantic_vote_min_depth
        if not torch.any(valid):
            return

        xyz_valid = xyz_cam[valid]
        depth_valid = depth[valid]
        candidate_indices = visible_indices[valid]
        fx = fov2focal(camera.FoVx, camera.image_width)
        fy = fov2focal(camera.FoVy, camera.image_height)
        cx = camera.image_width / 2.0
        cy = camera.image_height / 2.0
        u = torch.round((fx * xyz_valid[:, 0] / depth_valid) + cx).to(torch.int64)
        v = torch.round((fy * xyz_valid[:, 1] / depth_valid) + cy).to(torch.int64)

        in_frame = (u >= 0) & (u < camera.image_width) & (v >= 0) & (v < camera.image_height)
        if not torch.any(in_frame):
            return

        global_indices = candidate_indices[in_frame]
        u = u[in_frame].detach().cpu().numpy()
        v = v[in_frame].detach().cpu().numpy()
        depth_np = depth_valid[in_frame].detach().cpu().numpy()
        pixel_linear = v.astype(np.int64) * int(camera.image_width) + u.astype(np.int64)
        order = np.lexsort((depth_np, pixel_linear))
        pixel_linear_sorted = pixel_linear[order]
        first_per_pixel = np.ones(pixel_linear_sorted.shape[0], dtype=bool)
        first_per_pixel[1:] = pixel_linear_sorted[1:] != pixel_linear_sorted[:-1]
        front_order = order[first_per_pixel]
        if front_order.size == 0:
            return

        global_indices = global_indices[front_order]
        u_tensor = torch.from_numpy(u[front_order]).to(device=camera.semantic_mask.device, dtype=torch.long)
        v_tensor = torch.from_numpy(v[front_order]).to(device=camera.semantic_mask.device, dtype=torch.long)
        masked_hits = camera.semantic_mask[v_tensor, u_tensor] > int(mask_threshold)

        self.semantic_visible_views[global_indices] += 1
        if torch.any(masked_hits):
            self.semantic_positive_views[global_indices[masked_hits]] += 1

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def init_RT_seq(self, cam_list):
        poses =[]
        for cam in cam_list[1.0]:
            p = get_tensor_from_camera(cam.world_view_transform.transpose(0, 1)) # R T -> quat t
            poses.append(p)
        poses = torch.stack(poses)
        self.P = poses.cuda().requires_grad_(True)

    def get_RT(self, idx):
        pose = self.P[idx]
        return pose
    
    def get_RT_test(self, idx):
        pose = self.test_P[idx]
        return pose

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, scale_gaussian: np.ndarray = None):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        if scale_gaussian is not None:
            mean3_sq_dist = torch.from_numpy(scale_gaussian**2).float().cuda()
            dist2 = torch.min(mean3_sq_dist, dist2)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._ensure_semantic_state(self.get_xyz.shape[0])

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self._ensure_semantic_state(self.get_xyz.shape[0])
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr * 10, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0 * 10, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr * 10, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr * 10, "name": "rotation"},
        ]

        l_cam = [{'params': [self.P],'lr': training_args.rotation_lr*0.1, "name": "pose"},]
        l += l_cam

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.cam_scheduler_args = get_expon_lr_func(
                                                    lr_init=training_args.rotation_lr*0.1,
                                                    lr_final=training_args.rotation_lr*0.001,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.iterations)
        
    # per-point optimizer
    def training_setup_pp(self, training_args, confidence_lr=None):
        self.percent_dense = training_args.percent_dense
        self._ensure_semantic_state(self.get_xyz.shape[0])
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.per_point_lr = confidence_lr

        l = [
            {'params': [self._xyz], 'per_point_lr': self.per_point_lr, 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr * 10, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0 * 10, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr * 10, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr * 10, "name": "rotation"}
        ]

        l_cam = [{'params': [self.P],'lr': training_args.rotation_lr*0.1, "name": "pose"},]
        l += l_cam

        self.optimizer = PerPointAdam(l, lr=0, betas=(0.9, 0.999), eps=1e-15, weight_decay=0.0)

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        self.cam_scheduler_args = get_expon_lr_func(
                                                    lr_init=training_args.rotation_lr*0.1,
                                                    lr_final=training_args.rotation_lr*0.001,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.iterations)
    
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "pose":
                lr = self.cam_scheduler_args(iteration)
                # print("pose learning rate", iteration, lr)
                param_group['lr'] = lr
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
        # return lr


    def construct_list_of_attributes(self, include_semantic=False):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if include_semantic:
            l.extend([
                "semantic_label",
                "semantic_positive_views",
                "semantic_visible_views",
                "semantic_score",
            ])
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        vertex_data = elements
        semantic_output_dir = os.path.join(os.path.dirname(path), "semantic")
        if self.semantic_label_name:
            selected_mask, score = self._semantic_selected_mask()
            vertex_data = append_semantic_fields(
                vertex_data=elements,
                selected_mask=selected_mask,
                positive_views=self.semantic_positive_views.detach().cpu().numpy(),
                visible_views=self.semantic_visible_views.detach().cpu().numpy(),
                score=score,
            )
            mkdir_p(semantic_output_dir)
            selected_vertex_data = vertex_data[selected_mask]
            selected_ply_path = os.path.join(semantic_output_dir, f"point_cloud_{self.semantic_label_slug}.ply")
            PlyData([PlyElement.describe(selected_vertex_data, 'vertex')]).write(selected_ply_path)

            outline_payload = estimate_plane_outline(xyz[selected_mask])
            if outline_payload is not None:
                outline_payload.update(
                    {
                        "label_name": self.semantic_label_name,
                        "label_slug": self.semantic_label_slug,
                        "selected_count": int(selected_mask.sum()),
                        "gaussian_count": int(xyz.shape[0]),
                    }
                )
                with open(os.path.join(semantic_output_dir, f"{self.semantic_label_slug}_outline.json"), "w", encoding="utf-8") as handle:
                    json.dump(outline_payload, handle, indent=2)

            summary = {
                "label_name": self.semantic_label_name,
                "label_slug": self.semantic_label_slug,
                "gaussian_count": int(xyz.shape[0]),
                "selected_count": int(selected_mask.sum()),
                "selected_pct": float((selected_mask.sum() / max(1, xyz.shape[0])) * 100.0),
                "thresholds": {
                    "min_visible_views": self.semantic_min_visible_views,
                    "min_positive_views": self.semantic_min_positive_views,
                    "min_score": self.semantic_min_score,
                    "min_depth": self.semantic_vote_min_depth,
                },
                "outputs": {
                    "selected_ply": selected_ply_path,
                    "outline_json": os.path.join(semantic_output_dir, f"{self.semantic_label_slug}_outline.json"),
                },
            }
            with open(os.path.join(semantic_output_dir, f"{self.semantic_label_slug}_summary.json"), "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)

        el = PlyElement.describe(vertex_data, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((
                        np.array(plydata.elements[0]["x"], copy=True),
                        np.array(plydata.elements[0]["y"], copy=True),
                        np.array(plydata.elements[0]["z"], copy=True)),  axis=1)
        opacities = np.array(plydata.elements[0]["opacity"], copy=True)[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.array(plydata.elements[0]["f_dc_0"], copy=True)
        features_dc[:, 1, 0] = np.array(plydata.elements[0]["f_dc_1"], copy=True)
        features_dc[:, 2, 0] = np.array(plydata.elements[0]["f_dc_2"], copy=True)

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.array(plydata.elements[0][attr_name], copy=True)
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.array(plydata.elements[0][attr_name], copy=True)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.array(plydata.elements[0][attr_name], copy=True)

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        semantic_names = {p.name for p in plydata.elements[0].properties}
        if {"semantic_positive_views", "semantic_visible_views"}.issubset(semantic_names):
            self.semantic_positive_views = torch.tensor(
                np.array(plydata.elements[0]["semantic_positive_views"], copy=True),
                dtype=torch.int32,
                device="cuda",
            )
            self.semantic_visible_views = torch.tensor(
                np.array(plydata.elements[0]["semantic_visible_views"], copy=True),
                dtype=torch.int32,
                device="cuda",
            )
        else:
            self._ensure_semantic_state(xyz.shape[0])

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                # breakpoint()
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.semantic_positive_views = self.semantic_positive_views[valid_points_mask]
        self.semantic_visible_views = self.semantic_visible_views[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_semantic_positive_views=None, new_semantic_visible_views=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        if new_semantic_positive_views is None:
            new_semantic_positive_views = torch.zeros((new_xyz.shape[0],), dtype=torch.int32, device="cuda")
        if new_semantic_visible_views is None:
            new_semantic_visible_views = torch.zeros((new_xyz.shape[0],), dtype=torch.int32, device="cuda")
        self.semantic_positive_views = torch.cat((self.semantic_positive_views, new_semantic_positive_views), dim=0)
        self.semantic_visible_views = torch.cat((self.semantic_visible_views, new_semantic_visible_views), dim=0)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_semantic_positive_views = self.semantic_positive_views[selected_pts_mask].repeat(N)
        new_semantic_visible_views = self.semantic_visible_views[selected_pts_mask].repeat(N)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_semantic_positive_views,
            new_semantic_visible_views,
        )

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_semantic_positive_views = self.semantic_positive_views[selected_pts_mask]
        new_semantic_visible_views = self.semantic_visible_views[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_semantic_positive_views,
            new_semantic_visible_views,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # self.densify_and_clone(grads, max_grad, extent)
        # self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
