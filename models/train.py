import torch

from vgg import get_data
from VAE import MulLayer
from losses import LossCriterion
import random
import torch.nn as nn
from tqdm import trange

latent_dimension = 64
n_iters = 5000

style1 = get_data(style1)
style2 = get_data(style2)
content = get_data(content)

model = MulLayer(latent_dimension)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def objective():
    style1batch = next(iter(style1))[0]
    style2batch = next(iter(style2))[0]
    contentbatch = next(iter(content))[0]
    k = random.random()
    out, transmatrix, KL = model(k, contentbatch, style1batch, style2batch)
    return 0.05 *(k * nn.MSELoss(size_average=False)(out, style1batch) +
                  (1 - k) * nn.MSELoss(size_average=False)(out, style2batch)) + \
           nn.MSELoss(size_average=False)(out, contentbatch) + KL


def callback(t, loss_out):
    if t % 100 == 0:
        print("Iteration {} lower bound {}".format(t, loss_out))

def update():
    optimizer.zero_grad()
    loss = objective()
    loss.backward()
    optimizer.step()
    return loss


print("Optimizing variational parameters...")
for t in trange(0, n_iters):
    loss_out = update()
    callback(t, loss_out)
