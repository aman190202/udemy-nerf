import numpy as np
import torch

a = torch.tensor([1.,2.,3.,4.])
b = torch.tensor([5.,12.,21.,32.])



# 5,6,7,8
x = torch.tensor([344.,23.,12.,45.],requires_grad=True)
optimizer = torch.optim.Adam({x},lr=1e-1)

losses = []
for epoch in range(10000):

    loss = ((a*x - b)**2).mean()
    optimizer.zero_grad()
    loss.backward()
    losses.append(loss.item())
    optimizer.step()

    if(epoch % 1000 == 0):
        print(f'Epoch : {epoch} loss : {loss} x : {x}')

import matplotlib.pyplot as plt

plt.plot(losses)
plt.show()