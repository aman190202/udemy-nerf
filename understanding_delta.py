import torch 
tn = 5
tf = 10
n_bins = 5
t = torch.linspace(tn,tf,n_bins)
print(t[1:]-t[:-1])