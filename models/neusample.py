import numpy as np
import jittor as jt
from jittor import nn
import time
from collections import OrderedDict

im2mse = lambda x, y: jt.mean((x - y) ** 2)
mse2psnr = lambda x : -10 * jt.log(x) / jt.log(jt.Var([10]))

def raw2outputs(densities, colors, z_vals, rays_dir, alpha_noise_std, white_bkgd):
    def process_alphas(densities, dists, act_fn=nn.relu):
        return 1 - jt.exp(-act_fn(densities) * dists)

    # Computes distances
    dists = z_vals[..., 1:] - z_vals[..., :-1]  # dists: (B,N-1)

    # the distance that starts from the last point is infinity.
    dists = jt.concat([
        dists,
        2e10 * jt.ones(dists[..., :1].shape)
    ], dim=-1)  # [B, n_samples]

    # Multiplies each distance by the norm of its ray direction
    # to convert it to real world distance (accounts for non-unit ray directions).
    dists = dists * jt.norm(rays_dir[..., None, :], dim=-1)

    # [B, n_points, 1] -> [B, n_points]
    densities = densities.squeeze(-1)

    # Adds noise to model's predictions for density. Can be used to
    # regularize network (prevents floater artifacts).
    noise = 0
    if alpha_noise_std > 0:
        noise = jt.randn(densities.shape) * alpha_noise_std

    # Predicts density of point. Higher values imply
    # higher likelihood of being absorbed at this point.
    alphas = process_alphas(densities + noise, dists)  # [B, n_points]

    # Compute weight for RGB of each sample.  A cumprod() is
    # used to express the idea of the ray not having reflected up to this
    # sample yet.
    # [B, n_points]
    weights = alphas * jt.cumprod(
        jt.concat([jt.ones(tuple(alphas.shape[:-1]) + (1,)),
                   1 - alphas[..., :-1] + 1e-10], dim=-1),
        dim=-1
    )# (B,N)
    # Computed weighted color of each sample y
    color_map = jt.sum(weights[..., None] * colors, dim=-2)  # [B, 3]

    # Estimated depth map is expected distance.
    depth_map = jt.sum(weights * z_vals, dim=-1)  # [B]

    # Disparity map is inverse depth.
    disp_map = 1 / jt.maximum(1e-10 * jt.ones_like(depth_map), depth_map / jt.sum(weights, dim=-1))

    # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
    acc_map = jt.sum(weights, dim=-1) # (B,1)

    # To composite onto a white background, use the accumulated alpha map
    if white_bkgd:
        color_map = color_map + (1 - acc_map[..., None])

    outputs = {
        'alphas': alphas,
        'weights': weights,
        'color_map': color_map,
        'depth_map': depth_map,
        'disp_map': disp_map,
        'acc_map': acc_map
    }
    return outputs


class NeuSample(nn.Module):
    def __init__(self,
                 ori_embedder,
                 dir_ray_embedder,
                 xyz_embedder,
                 sample_field,
                 render_params,
                 dir_embedder,
                 radiance_field,
                 **kwargs):
        super().__init__()
        self.ori_embedder = ori_embedder
        self.dir_ray_embedder = dir_ray_embedder
        self.sample_field = sample_field

        self.xyz_embedder = xyz_embedder
        self.dir_embedder = dir_embedder
        self.radiance_field = radiance_field

        self.render_params = render_params
        self.fp16_enabled = False

    def forward_points(self, points, directions):
        shape = tuple(points.shape[:-1])  # [B, n_points]
        # [B, 3] -> [B, n_points, 3]
        directions = directions[..., None, :].expand_as(points)
        directions = directions.reshape((-1, 3))
        dir_embeds = self.dir_embedder(directions)

        points = points.reshape((-1, 3))
        xyz_embeds = self.xyz_embedder(points)
        densities, colors = self.radiance_field(xyz_embeds, dir_embeds)
        densities = densities.reshape(shape + (1,))
        colors = colors.reshape(shape + (3,))

        return densities, colors

    def forward_batchified(self,
                           points,
                           directions,
                           max_rays_num):
        nb_rays = points.shape[0]
        if nb_rays <= max_rays_num or self.is_train:
            return self.forward_points(points, directions)
        else:
            outputs = []
            start = 0
            while start < nb_rays:
                end = min(start + max_rays_num, nb_rays)
                assert start < end, 'start >= end ({:d}, {:d})'.format(start, end)
                output = self.forward_points(points[start:end, ...],
                                             directions[start:end, ...], )
                outputs.append(output)
                start += max_rays_num

            densities_colors = []
            # for out in zip(*outputs):
            #     if out[0] is not None:
            #         out = jt.concat(out, dim=0).sync()
            #     else:
            #         out = None
            #     densities_colors.append(out)

            densities = jt.zeros((nb_rays,points.shape[1],1),dtype='float')
            colors = jt.zeros((nb_rays,points.shape[1],3),dtype='float')
            start = 0
            while start<=nb_rays//max_rays_num:
                densities[start*max_rays_num:start*max_rays_num+outputs[start][0].shape[0],:,:] = outputs[start][0]
                colors[start*max_rays_num:start*max_rays_num+outputs[start][1].shape[0],:,:] = outputs[start][1]
                start =start + 1
                densities.sync()
                colors.sync()

            densities_colors.append(densities)
            densities_colors.append(colors)

            return densities_colors

    def sample_points(self, rays_ori, directions):
        ori_embeds = self.ori_embedder(rays_ori)
        dir_embeds = self.dir_ray_embedder(directions)
        sampled_dists = self.sample_field(ori_embeds, dir_embeds)
        return sampled_dists

    def sample_batchified(self, rays_ori, directions, max_rays_num):
        nb_rays = rays_ori.shape[0]
        if nb_rays <= max_rays_num or self.is_train:
            sampled_points = self.sample_points(rays_ori, directions)
            return sampled_points
        else:
            outputs = []
            start = 0
            while start < nb_rays:
                end = min(start + max_rays_num, nb_rays)
                assert start < end, 'start >= end ({:d}, {:d})'.format(start, end)
                output = self.sample_points(rays_ori[start:end, ...],
                                            directions[start:end, ...], )
                outputs.append(output)
                start += max_rays_num
            sampled_dists = jt.concat(outputs, dim=0)
            return sampled_dists

    def forward_render(self,
                       rays_ori, rays_dir, rays_color,
                       alpha_noise_std, inv_depth,  # render param
                       max_rays_num, near=0.0, far=1.0, white_bkgd=False, **kwargs):

        if isinstance(near, jt.Var) and len(near.shape) > 0:
            near = near[..., 0].item()
            far = far[..., 0].item()

        directions = jt.normalize(rays_dir, p=2, dim=-1)  # p:L2_Norm dim:normalize (,3) in batch

        sampled_dists = self.sample_batchified(
            rays_ori, directions, max_rays_num * 64 * 8 // self.sample_field.nb_layers)

        sorted_idx, t_vals = jt.argsort(sampled_dists, dim=1)

        if not inv_depth:
            z_vals = near * (1 - t_vals) + far * t_vals
        else:
            z_vals = 1 / (1 / near * (1 - t_vals) + 1 / far * t_vals)

        points = rays_ori[..., None, :] + rays_dir[..., None, :] * \
                 z_vals[..., :, None]  # (B,N,3)


        densities, colors = self.forward_batchified(points, directions,  # densities: (B,N,1) colors: (B,N,3)
                                                    max_rays_num=max_rays_num * 192 // points.shape[1])

        outputs = raw2outputs(densities,
                              colors,
                              z_vals,
                              rays_dir,
                              alpha_noise_std,
                              white_bkgd, )

        return {'fine': outputs}


    def execute(self, rays, render_params=None):
        """
        Args:
            rays (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict:
        """

        # ray dict: 5
        # ray_ori: Tensor(b, 3)
        # ray_dir: Tensor(b, 3)
        # rays_color: Tensor(b, 3)
        # near
        # far
        for k, v in rays.items():  # k:dict_name v:dict_content
            if len(v.shape) > 2:
                rays[k] = v.flatten(0, 1)

        if render_params is None:
            render_params = self.render_params
        outputs = self.forward_render(**rays, **render_params)

        im_loss_fine = im2mse(outputs['fine']['color_map'], rays['rays_color'])
        outputs['fine_loss'] = im_loss_fine

        return outputs

def _parse_outputs(outputs):
    loss, log_vars = _parse_losses(outputs)
    log_vars['psnr'] = mse2psnr(outputs['fine_loss']).item()
    outputs.update(dict(loss=loss, log_vars=log_vars))
    outputs['num_samples'] = 1
    return outputs

def _parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if 'loss' not in loss_name:
            continue
        if isinstance(loss_value, jt.Var):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        elif isinstance(loss_value, dict):
            for name, value in loss_value.items():
                log_vars[name] = value
        else:
            raise TypeError(
                f'{loss_name} is not a jittor var or list of jittor vars')

    loss = sum(_value for _key, _value in log_vars.items())
    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars


