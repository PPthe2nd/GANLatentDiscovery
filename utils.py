import os
import sys
import json
import torch
from scipy.stats import truncnorm
from scipy.stats import loglaplace
from sklearn.linear_model import LinearRegression


def make_noise(batch, dim, truncation=None):
    if isinstance(dim, int):
        dim = [dim]
    if truncation is None or truncation == 1.0:
        return torch.randn([batch] + dim)
    else:
        return torch.from_numpy(truncated_noise([batch] + dim, truncation)).to(torch.float)
        #return torch.from_numpy(w_space_noise([batch] + dim)).to(torch.float)

      
def is_conditional(G):
    return 'biggan' in G.__class__.__name__.lower()


def one_hot(dims, value, indx):
    vec = torch.zeros(dims)
    vec[indx] = value
    return vec


def save_command_run_params(args):
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'args.json'), 'w') as args_file:
        json.dump(args.__dict__, args_file)
    with open(os.path.join(args.out, 'command.sh'), 'w') as command_file:
        command_file.write(' '.join(sys.argv))
        command_file.write('\n')


def truncated_noise(size, truncation=1.0):
    return truncnorm.rvs(-truncation, truncation, size=size)

def w_space_noise(size):
    return loglaplace.rvs(3.25,loc=-.8,scale=1,size=size)

def reg_brain(train_n_data,train_w_data):
    reg = LinearRegression().fit(train_n_data, train_w_data)
    coef_ = torch.from_numpy(reg.coef_,).transpose(1,0).to(torch.float)
    intercept_ = torch.from_numpy(reg.intercept_).to(torch.float)
    return coef_, intercept_
