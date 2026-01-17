from conv2d_stuff import Generator, StyleVectorizer, image_noise, latent_to_w, Trainer
import torch
import numpy as np

import math

millnames = ['',' Thousand',' Million',' Billion',' Trillion']

def millify(n):
    n = float(n)
    millidx = max(0,min(len(millnames)-1,
                        int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))

    return '{:.0f}{}'.format(n / 10**(3 * millidx), millnames[millidx])
# S = StyleVectorizer(512, 8)
# in_stuff = torch.randn(10,512)
# out = S(in_stuff)
# gen = Generator(image_size=256, latent_dim=512, network_capacity=1)
# ins = image_noise(10, gen.image_size, device=0)
# print(ins.shape, out.shape)
# gen.cuda(0)
# out = out.cuda(0)
#
# w_space = latent_to_w(S, style)
# w_styles = styles_def_to_tensor(w_space)
# # out2 = gen(out, ins)
image_s = 512
latents_d = 512
net = Generator(image_size=image_s, latent_dim=512, fmap_max=256)
model_parameters = filter(lambda p: p.requires_grad, net.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(millify(params))
batch_s = 4
layers_c = net.num_layers
print(layers_c, "LAYERS")

styles = torch.randn(4, layers_c, latents_d)
noises = torch.randn(batch_s, image_s, image_s, 1)
out = net(styles, noises)
print(out.shape)
# a = Trainer()
# a.train()