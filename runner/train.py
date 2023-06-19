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
import matplotlib.pyplot as plt
#from torch.utils.data import DataLoader

# TODO: config here
def train_config():

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
    train_dataset = SyntheticDataset(base_dir='./dataset/nerf_synthetic/lego',
                                     half_res=False,
                                     white_bkgd=True,
                                     precrop_frac=0.5,
                                     testskip=8,
                                     split='train',
                                     batchsize=2,  # set this
                                     batch_size=1,   # fixed
                                     shuffle=True,
                                     num_workers=0,)
    train_loader = RepeatDataset(train_dataset, times=20)
    val_loader = SyntheticDataset(base_dir='./dataset/nerf_synthetic/lego',
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
            super(PolyLR, self).__init__(optimizer, last_epoch)

        def get_lr(self):
            if self.last_epoch == -1:
                return [1e-5 for _ in self.optimizer.param_groups]
            else:
                alpha = (1 - float(self.last_epoch) / float(self.max_epoch)) ** self.power
                return [self.base_lr * alpha for _ in self.optimizer.param_groups]

    scheduler = PolyLR(optimizer, base_lr=lr, power=1, max_epoch=epochs)


    return model, train_loader, val_loader, model, optimizer, scheduler, epochs


# start training
def traindetector(logger, args):
    model, train_loader, val_loader, model, optimizer, scheduler, epochnum = train_config()
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
                bestname = f'checkpoint_{best_psnr:.2f}.pth'
                if logger is not None:
                     logger.info(f'Saving best {bestname}')
                jt.save(model.state_dict(),os.path.join(args.work_dir, bestname))
            else:
                logger.info(f'No improvement Current best {bestname}')
            logger.info(f'******************************************************************')

        scheduler.step()


def evaluate(model, Val_Loader, iter, logger, args):
    model.eval()
    loss = 0
    psnr = 0
    model.render_params = dict(alpha_noise_std=0, inv_depth=False, white_bkgd=True, max_rays_num=1024*4,)

    for i, data in enumerate(Val_Loader):
        # temp = np.array(data['rays_color'].reshape(800,800,3)*255).astype(np.int)
        # plt.imshow(temp)
        # plt.show()

        with jt.no_grad():
            outputs = model(data)
            outputs = _parse_outputs(outputs)


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
    if jt.in_mpi:
        loss = loss.mpi_all_reduce()
        psnr = psnr.mpi_all_reduce()
    loss = loss / len(Val_Loader)
    psnr = psnr / len(Val_Loader)
    if jt.rank == 0:
        logger.info(f'Valid Loss:  {loss}   PSNR:  {psnr}')
    return psnr

