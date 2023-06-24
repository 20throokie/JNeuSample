import jittor.init
from jittor import nn
from models.base_embedder import BaseEmbedder
from models.sample_field import SampleField
from models.base_field import BaseField
from models.neusample import NeuSample, _parse_outputs, _parse_losses
from datasets.synthetic_dataset import SyntheticDataset
from datasets.utils import RepeatDataset
import jittor as jt
import os
import lpips
import time
import numpy as np
import time
import tensorflow as tf
import torch
import cv2


def test(logger, args):
    ori_embedder = BaseEmbedder(in_dims=3, n_freqs=10, include_input=True)
    dir_ray_embedder = BaseEmbedder(in_dims=3, n_freqs=10, include_input=True)
    sample_field = SampleField(nb_layers=8, hid_dims=256, ori_emb_dims=2*3*10+3, dir_emb_dims=2*3*10+3, n_samples=192)
    xyz_embedder = BaseEmbedder(in_dims=3, n_freqs=10, include_input=True)
    dir_embedder = BaseEmbedder(in_dims=3, n_freqs=4, include_input=True)
    radiance_field = BaseField(nb_layers=8, hid_dims=256, xyz_emb_dims=2*3*10+3, dir_emb_dims=2*3*4+3, use_dirs=True)
    render_params = dict(alpha_noise_std=0, inv_depth=False, white_bkgd=True, max_rays_num=1024*4,)

    model = NeuSample(ori_embedder,dir_ray_embedder,xyz_embedder,sample_field,render_params,dir_embedder,radiance_field)
    test_loader = SyntheticDataset(base_dir=os.path.join('./dataset/nerf_synthetic', args.obj_class),
                                   half_res=False,
                                   white_bkgd=True,
                                   #precrop_frac=0.5,
                                   testskip=1,
                                   split='val',
                                   batch_size=1,
                                   batchsize=-1,
                                   shuffle=False,
                                   num_workers=0,)
    ckpt = jt.load(args.load_path)
    model.load_parameters(ckpt)
    model.eval()
    lpips_model = lpips.LPIPS(net='vgg')
    loss = 0
    psnr = 0
    lpips_score = 0
    ssim_score = 0
    time_cost = 0

    for i, data in enumerate(test_loader):
        # temp = np.array(data['rays_color'].reshape(800,800,3)*255).astype(np.int)
        # plt.imshow(temp)
        # plt.show()

        with jt.no_grad():
            start_time = time.time()
            outputs = model(data)
            outputs = _parse_outputs(outputs)
            end_time = time.time()

            # save images
            # im = outputs['coarse']['color_map'].reshape(self.im_shape)
            # im = 255 * im.detach().cpu().numpy()
            # # TODO: convert to video
            # cv2.imwrite(osp.join(
            #     self.out_dir, f'iter{runner.iter+1}-id{i}-coarse.png'), im[:,:,::-1])

            # if outputs['fine'] is not None:
            #     im = outputs['fine']['color_map'].reshape(Val_Loader.h, Val_Loader.w, 3)
            #     im = np.array(255 * im)
            #     # TODO: convert to video
            #     cv2.imwrite(os.path.join(
            #         args.work_dir, f'iter{iter}-id{i}-fine.png'), im[:,:,::-1])

            if outputs['fine'] is not None:
                im_ori_fine = outputs['fine']['color_map'].reshape(test_loader.h, test_loader.w, 3)
                im = np.array(255 * im_ori_fine)
                cv2.imwrite(os.path.join(args.work_dir, '-fine.png'), im[:,:,::-1])

            gt_ori = data['rays_color'].reshape(test_loader.h, test_loader.w, 3)
            gt = np.array(255 * gt_ori)
            cv2.imwrite(os.path.join(args.work_dir, f'gt-id{i}.png'), gt[:, :, ::-1])

            gt_lpips = gt_ori.permute([2, 0, 1]) * 2.0 - 1.0
            predict_image_lpips = im_ori_fine.permute([2, 0, 1]).clamp(0, 1) * 2.0 - 1.0
            lpips_score += lpips_model.forward(torch.tensor(np.array(predict_image_lpips)), torch.tensor(np.array(gt_lpips))).item()

            gt_load = tf.image.decode_image(tf.io.read_file(os.path.join(args.work_dir, f'gt-id{i}.png')))
            pred_load = tf.image.decode_image(tf.io.read_file(os.path.join(args.work_dir, '-fine.png')))
            gt_load = tf.expand_dims(gt_load, axis=0)
            pred_load = tf.expand_dims(pred_load, axis=0)
            ssim = tf.image.ssim(gt_load, pred_load, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
            ssim_score += float(ssim[0])

        loss += outputs['log_vars']['loss']
        psnr += outputs['log_vars']['psnr']
        time_cost += end_time - start_time
    if jt.in_mpi:
        loss = loss.mpi_all_reduce()
        psnr = psnr.mpi_all_reduce()
        time_cost = time_cost.mpi_all_reduce()
        lpips_score = lpips_score.mpi_all_reduce()
        ssim_score = ssim_score.mpi_all_reduce()
    loss = loss / len(test_loader)
    psnr = psnr / len(test_loader)
    ssim_score = ssim_score / len(test_loader)
    lpips_score = lpips_score / len(test_loader)
    time_cost = time_cost / len(test_loader)
    if jt.rank == 0:
        logger.info(f'Valid Loss:  {loss}   PSNR:  {psnr}   SSIM:  {ssim_score}   LPIPS:  {lpips_score}   Average Time Cost:  {time_cost}')
    return psnr