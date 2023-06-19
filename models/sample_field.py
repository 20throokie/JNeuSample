import jittor as jt
import jittor.init
from jittor import nn
import math


class SampleField(nn.Module):
    def __init__(self, nb_layers=8, hid_dims=256, ori_emb_dims=63,
                 dir_emb_dims=27, n_samples=192):
        super().__init__()
        self.nb_layers = nb_layers
        self.hid_dims = hid_dims
        self.ori_emb_dims = ori_emb_dims
        self.dir_emb_dims = dir_emb_dims
        self.n_samples = n_samples
        self.input_dims = self.ori_emb_dims + self.dir_emb_dims
        self.skips = [nb_layers // 2]

        self.relu = nn.ReLU()
        self.layers = nn.Sequential()
        self.layers.add_module('fc0', nn.Linear(self.input_dims, hid_dims))
        for i in range(1, nb_layers):
            if i in self.skips:
                self.layers.add_module(
                    'fc{}'.format(i),
                    nn.Linear(hid_dims + self.input_dims, hid_dims)
                )
            else:
                self.layers.add_module(
                    'fc{}'.format(i),
                    nn.Linear(hid_dims, hid_dims)
                )

        self.dist_out = nn.Linear(hid_dims, n_samples)
        self.fp16_enabled = False

        for m in self.modules():
            if isinstance(m, nn.Linear):
                jittor.init.kaiming_uniform_(m.weight, a=math.sqrt(5))

    def execute(self, ori_embeds, dir_embeds):
        x = jt.concat([ori_embeds, dir_embeds], dim=1)
        cat_skip = x
        for i in range(self.nb_layers):
            key = 'fc{}'.format(i)
            layer = self.layers[key]
            if i in self.skips:
                x = jt.concat([x, cat_skip], dim=1)
            x = layer(x)
            x = self.relu(x)

        dists = self.dist_out(x)
        dists = jt.sigmoid(dists)
        return dists