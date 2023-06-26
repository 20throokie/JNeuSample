import jittor.init
from jittor import nn
from models.base_embedder import BaseEmbedder
from models.sample_field import SampleField
from models.base_field import BaseField
from models.neusample import NeuSample, _parse_outputs, _parse_losses
from datasets.synthetic_dataset import SyntheticDataset
from datasets.utils import RepeatDataset
import jittor as jt
from jittor.optim import LRScheduler
import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import lpips
import tensorflow as tf
import torch
import math
#from torch.utils.data import DataLoader
metric = []


# TODO: config here
def train_config(args):

    #********************Network******************
    ori_embedder = BaseEmbedder(in_dims=3, n_freqs=10, include_input=True)
    dir_ray_embedder = BaseEmbedder(in_dims=3, n_freqs=10, include_input=True)
    sample_field = SampleField(nb_layers=8, hid_dims=256, ori_emb_dims=2*3*10+3, dir_emb_dims=2*3*10+3, n_samples=192)
    xyz_embedder = BaseEmbedder(in_dims=3, n_freqs=10, include_input=True)
    dir_embedder = BaseEmbedder(in_dims=3, n_freqs=4, include_input=True)
    radiance_field = BaseField(nb_layers=8, hid_dims=256, xyz_emb_dims=2*3*10+3, dir_emb_dims=2*3*4+3, use_dirs=True)
    render_params = dict(alpha_noise_std=1.0, inv_depth=False, max_rays_num=1024*4,)

    model = NeuSample(
        ori_embedder,
        dir_ray_embedder,
        xyz_embedder,
        sample_field,
        render_params,
        dir_embedder,
        radiance_field)


    #********************Dataset******************
    train_dataset = SyntheticDataset(base_dir=os.path.join('./dataset/nerf_synthetic', args.obj_class),
                                     half_res=False,
                                     white_bkgd=True,
                                     precrop_frac=0.5,
                                     testskip=8,
                                     split='train',
                                     batchsize=1024*4,  # set this
                                     batch_size=1,   # fixed
                                     shuffle=True,
                                     num_workers=0,)
    train_loader = RepeatDataset(train_dataset, times=20)
    val_loader = SyntheticDataset(base_dir=os.path.join('./dataset/nerf_synthetic', args.obj_class),
                                   half_res=False,
                                   white_bkgd=True,
                                   #precrop_frac=0.5,
                                   testskip=8,
                                   split='val',
                                   batch_size=1,
                                   batchsize=-1,
                                   shuffle=False,
                                   num_workers=0,)


    #********************Optimizer******************
    lr=5e-4
    epochs=100
    optimizer = jt.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    class PolyLR(LRScheduler):
        def __init__(self, optimizer, base_lr, power, max_epoch, last_epoch=-1):
            self.base_lr = base_lr
            self.power = power
            self.max_epoch = max_epoch
            self.warm_up = 1-1
            super(PolyLR, self).__init__(optimizer, last_epoch)

        def get_lr(self):
            if self.last_epoch == -1:
                return [1e-5 for _ in self.optimizer.param_groups]
            else:
                alpha = (1 - float(self.last_epoch) / float(self.max_epoch)) ** self.power
                return [self.base_lr * alpha for _ in self.optimizer.param_groups]
                # if self.last_epoch <= self.warm_up:
                #     return [self.base_lr * 0.01 for _ in self.optimizer.param_groups]
                # else:
                #     alpha = (1 - (float(self.last_epoch)-float(self.warm_up)) / float(self.max_epoch)) ** self.power
                #     return [self.base_lr * alpha for _ in self.optimizer.param_groups]

    scheduler = PolyLR(optimizer, base_lr=lr, power=1, max_epoch=epochs)


    return model, train_loader, val_loader, optimizer, scheduler, epochs


# start training
def trainrenderer(logger, args):
    model, train_loader, val_loader, optimizer, scheduler, epochnum = train_config(args)
    iter = 0
    best_psnr = 0
    for epoch in range(epochnum):
        model.train()
        for i, data in enumerate(train_loader):
            iter = iter + 1
            outputs = model(data)
            outputs = _parse_outputs(outputs)
            loss = outputs['fine_loss']
            #optimizer.clip_grad_norm()
            optimizer.step(loss)
            cur_lr = scheduler.get_last_lr()

            if jt.rank == 0:
                if iter==1000:
                    old = train_loader.dataset_.precrop_frac
                    train_loader.dataset_.set_attrs(precrop_frac=1)
                    logger.info(f'Iteration{iter} precrop_frac change from {old} to {train_loader.dataset_.precrop_frac}')
                if iter%200 == 0:
                    logger.info(f"Epoch[{epoch+1}][{i+1}/{len(train_loader)}]  Learning Rate{cur_lr}  Loss:{loss}  PSNR:{outputs['log_vars']['psnr']}")
                # if iter%1000 == 0:
                #     _print_grad(model, optimizer, logger)

        psnr = evaluate(model, val_loader, iter, logger, args)
        if jt.rank == 0:
            if psnr > best_psnr:
                old_filename = f'checkpoint_{best_psnr:.2f}.pth'
                if os.path.isfile(os.path.join(args.work_dir, old_filename)):
                    os.remove(os.path.join(args.work_dir, old_filename))
                best_psnr = psnr
                bestname = f'checkpoint_{best_psnr:.2f}.pkl'
                if logger is not None:
                     logger.info(f'Saving best {bestname}')
                #jt.save(model.state_dict(),os.path.join(args.work_dir, bestname))
                model.save(os.path.join(args.work_dir, bestname))
            else:
                logger.info(f'No improvement Current best {bestname}')
            logger.info(f'******************************************************************')

        scheduler.step()


def evaluate(model, Val_Loader, iter, logger, args):
    model.eval()
    loss = 0
    psnr = 0
    time_cost = 0
    model.render_params = dict(alpha_noise_std=0, inv_depth=False, white_bkgd=True, max_rays_num=1024*4,)

    for i, data in enumerate(Val_Loader):
        # temp = np.array(data['rays_color'].reshape(800,800,3)*255).astype(np.int)
        # plt.imshow(temp)
        # plt.show()

        with jt.no_grad():
            start_time = time.clock()
            outputs = model(data)
            outputs = _parse_outputs(outputs)
            end_time = time.clock()

            # save images
            # im = outputs['coarse']['color_map'].reshape(self.im_shape)
            # im = 255 * im.detach().cpu().numpy()
            # # TODO: convert to video
            # cv2.imwrite(osp.join(
            #     self.out_dir, f'iter{runner.iter+1}-id{i}-coarse.png'), im[:,:,::-1])

            if outputs['fine'] is not None:
                im = outputs['fine']['color_map'].reshape(Val_Loader.h, Val_Loader.w, 3)
                im = np.array(255 * im)
                # TODO: convert to video
                cv2.imwrite(os.path.join(
                    args.work_dir, f'iter{iter}-id{i}-fine.png'), im[:,:,::-1])

        loss += outputs['log_vars']['loss']
        psnr += outputs['log_vars']['psnr']
        time_cost += end_time - start_time
    if jt.in_mpi:
        loss = loss.mpi_all_reduce()
        psnr = psnr.mpi_all_reduce()
        time_cost = time_cost.mpi_all_reduce()
    loss = loss / len(Val_Loader)
    psnr = psnr / len(Val_Loader)
    time_cost = time_cost / len(Val_Loader)
    if jt.rank == 0:
        logger.info(f'Valid Loss:  {loss}   PSNR:  {psnr}   Average Time Cost:  {time_cost}')
    return psnr


def fine_tune(logger, args):
    model, train_loader, val_loader, _, _, _ = train_config(args)
    train_loader.dataset_.set_attrs(precrop_frac=1)
    ckpt = jt.load(args.load_path)
    model.load_parameters(ckpt)
    model.sample_field.dist_out = nn.Linear(256, 64)
    jittor.init.kaiming_uniform_(model.sample_field.dist_out.weight, a=math.sqrt(5))
    iter = 0
    best_psnr = 0
    fixed_para = [para for name, para in model.named_parameters() if name not in ['sample_field.dist_out.weight', 'sample_field.dist_out.bias']]
    #optimizer = jt.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.999))
    optimizer = jt.optim.Adam([{'params':fixed_para},
                               {'params':model.sample_field.parameters(), 'lr':5e-4},
                              ], lr=5e-5, betas=(0.9, 0.999))

    for epoch in range(10):
        model.train()
        for i, data in enumerate(train_loader):
            iter = iter + 1
            outputs = model(data)
            outputs = _parse_outputs(outputs)
            loss = outputs['fine_loss']
            optimizer.step(loss)

            if jt.rank == 0:
                if iter%200 == 0:
                    logger.info(f"Epoch[{epoch+1}][{i+1}/{len(train_loader)}]  Loss:{loss}  PSNR:{outputs['log_vars']['psnr']}")

        psnr = evaluate(model, val_loader, iter, logger, args)
        if jt.rank == 0:
            if psnr > best_psnr:
                best_psnr = psnr
                bestname = f'fine_tune_{best_psnr:.2f}.pkl'
                if logger is not None:
                     logger.info(f'Saving best {bestname}')
                #jt.save(model.state_dict(),os.path.join(args.work_dir, bestname))
                model.save(os.path.join(args.work_dir, bestname))
            else:
                logger.info(f'No improvement Current best {bestname}')
            logger.info(f'******************************************************************')


def test2(logger, args):
    ori_embedder = BaseEmbedder(in_dims=3, n_freqs=10, include_input=True)
    dir_ray_embedder = BaseEmbedder(in_dims=3, n_freqs=10, include_input=True)
    sample_field = SampleField(nb_layers=8, hid_dims=256, ori_emb_dims=2*3*10+3, dir_emb_dims=2*3*10+3, n_samples=64)
    xyz_embedder = BaseEmbedder(in_dims=3, n_freqs=10, include_input=True)
    dir_embedder = BaseEmbedder(in_dims=3, n_freqs=4, include_input=True)
    radiance_field = BaseField(nb_layers=8, hid_dims=256, xyz_emb_dims=2*3*10+3, dir_emb_dims=2*3*4+3, use_dirs=True)
    render_params = dict(alpha_noise_std=0, inv_depth=False, white_bkgd=True, max_rays_num=1024*4,)

    model = NeuSample(ori_embedder,dir_ray_embedder,xyz_embedder,sample_field,render_params,dir_embedder,radiance_field)
    test_loader = SyntheticDataset(base_dir=os.path.join('./dataset/nerf_synthetic', args.train_class),
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