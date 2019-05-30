# License: MIT
# Author: Karl Stelzner

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import visdom

import os

import model
import datasets
import config
import tqdm

vis = visdom.Visdom()
vis_name = 'monet_clevr'
win_recon = None
win_z_kld = None
win_mask_kld = None
win_images = None


def numpify(tensor):
    return tensor.cpu().detach().numpy()


def visualize_masks(imgs, masks, recons, viz_name=vis_name):
    global win_images
    # print('recons min/max', recons[:, 0].min().item(), recons[:, 0].max().item())
    # print('recons1 min/max', recons[:, 1].min().item(), recons[:, 1].max().item())
    # print('recons2 min/max', recons[:, 2].min().item(), recons[:, 2].max().item())
    recons = np.clip(recons, 0., 1.)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    colors.extend([(c[0]//2, c[1]//2, c[2]//2) for c in colors])
    colors.extend([(c[0]//4, c[1]//4, c[2]//4) for c in colors])
    seg_maps = np.zeros_like(imgs)
    masks = np.argmax(masks, 1)
    for i in range(imgs.shape[0]):
        for y in range(imgs.shape[2]):
            for x in range(imgs.shape[3]):
                seg_maps[i, :, y, x] = colors[masks[i, y, x]]

    seg_maps /= 255.0
    if win_images is None:
        win_images = vis.images(np.concatenate((imgs, seg_maps, recons), 0),
                                env=viz_name+'_reconstruction', nrow=imgs.shape[0])
    else:
        win_images = vis.images(np.concatenate((imgs, seg_maps, recons), 0),
                                win=win_images, env=viz_name+'_reconstruction', nrow=imgs.shape[0])


def viz_lines(iter, recons_loss, z_kld, mask_kld):
    global win_mask_kld, win_recon, win_z_kld

    if win_recon is None:
        win_recon = vis.line(
            X=[iter],
            Y=[recons_loss],
            env=vis_name + '_lines',
            opts=dict(
                width=400,
                height=400,
                xlabel='iteration',
                title='reconstruction loss', ))
    else:
        win_recon = vis.line(
            X=[iter],
            Y=[recons_loss],
            env=vis_name + '_lines',
            win=win_recon,
            update='append',
            opts=dict(
                width=400,
                height=400,
                xlabel='iteration',
                title='reconstruction loss', ))

    if win_z_kld is None:
        win_z_kld = vis.line(
            X=[iter],
            Y=[z_kld],
            env=vis_name + '_lines',
            opts=dict(
                width=400,
                height=400,
                xlabel='iteration',
                title='z kld loss', ))
    else:
        win_z_kld = vis.line(
            X=[iter],
            Y=[z_kld],
            env=vis_name + '_lines',
            win=win_z_kld,
            update='append',
            opts=dict(
                width=400,
                height=400,
                xlabel='iteration',
                title='z kld loss', ))

    if win_mask_kld is None:
        win_mask_kld = vis.line(
            X=[iter],
            Y=[mask_kld],
            env=vis_name + '_lines',
            opts=dict(
                width=400,
                height=400,
                xlabel='iteration',
                title='mask kld loss', ))
    else:
        win_mask_kld = vis.line(
            X=[iter],
            Y=[mask_kld],
            env=vis_name + '_lines',
            win=win_mask_kld,
            update='append',
            opts=dict(
                width=400,
                height=400,
                xlabel='iteration',
                title='mask kld loss', ))


def run_training(monet, conf, trainloader):
    if conf.load_parameters and os.path.isfile(conf.checkpoint_file):
        monet.load_state_dict(torch.load(conf.checkpoint_file))
        print('Restored parameters from', conf.checkpoint_file)
    else:
        for w in monet.parameters():
            std_init = 0.01
            nn.init.normal_(w, mean=0., std=std_init)
        print('Initialized parameters')

    optimizer = optim.RMSprop(monet.parameters(), lr=1e-4)

    for epoch in tqdm.trange(conf.num_epochs):
        running_loss = 0.0
        running_break_loss = np.array([0., 0., 0.])
        optimizer.zero_grad()
        minibatch_loss_sum = 0
        minibatch_loss_break_sum = np.array([0., 0., 0.])
        batch_pbar = tqdm.tqdm(trainloader)
        for minibatch_idx, data in enumerate(batch_pbar, 0):
            images, counts = data
            images = images.cuda()
            output = monet(images)
            loss = torch.mean(output['loss'])
            loss_break = np.asarray([torch.mean(x).detach().cpu().numpy() for x in output['loss_break']])
            loss_break /= conf.subdivs
            loss /= conf.subdivs
            minibatch_loss_sum += loss
            minibatch_loss_break_sum += loss_break
            loss.backward()
            if (minibatch_idx+1) % conf.subdivs == 0:
                optimizer.step()
                optimizer.zero_grad()

                # Show progress
                batch_pbar.set_postfix(dict(loss=minibatch_loss_sum.data.item()))

                running_loss += minibatch_loss_sum.data.item()
                running_break_loss += minibatch_loss_break_sum
                minibatch_loss_sum = 0
                minibatch_loss_break_sum = np.array([0., 0., 0.])

            if (minibatch_idx+1) % (conf.vis_every*conf.subdivs) == 0:
                #print('[%d, %5d] loss: %.3f' %
                #      (epoch + 1, minibatch_idx // conf.subdivs + 1, running_loss / conf.vis_every))
                visualize_masks(numpify(images[:8]),
                                numpify(output['masks'][:8]),
                                numpify(output['reconstructions'][:8]))

                recons_loss, z_kld, mask_kld = running_break_loss / conf.vis_every

                viz_lines(epoch*len(trainloader) + minibatch_idx + 1,
                          recons_loss=recons_loss,
                          z_kld=z_kld,
                          mask_kld=mask_kld)

                running_loss = 0.0
                running_break_loss = np.array([0., 0., 0.])

        torch.save(monet.state_dict(), conf.checkpoint_file)

    print('training done')

def sprite_experiment():
    conf = config.sprite_config
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.float()),
                                    ])
    trainset = datasets.Sprites(conf.data_dir, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=conf.batch_size // conf.subdivs,
                                              shuffle=True, num_workers=2)
    monet = model.Monet(conf, 64, 64).cuda()
    if conf.parallel:
        monet = nn.DataParallel(monet)
    run_training(monet, conf, trainloader)

def clevr_experiment():
    conf = config.clevr_config
    # Crop as described in appendix C
    crop_tf = transforms.Lambda(lambda x: transforms.functional.crop(x, 29, 64, 192, 192))
    drop_alpha_tf = transforms.Lambda(lambda x: x[:3])
    transform = transforms.Compose([crop_tf,
                                    transforms.Resize((128, 128)),
                                    transforms.ToTensor(),
                                    drop_alpha_tf,
                                    transforms.Lambda(lambda x: x.float()),
                                   ])
    trainset = datasets.Clevr(conf.data_dir,
                              transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=conf.batch_size // conf.subdivs,
                                              shuffle=True, num_workers=8)
    monet = model.Monet(conf, 128, 128).cuda()
    if conf.parallel:
        monet = nn.DataParallel(monet)
    run_training(monet, conf, trainloader)

if __name__ == '__main__':
    clevr_experiment()
    # sprite_experiment()

