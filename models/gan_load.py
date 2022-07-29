import json
import numpy as np
import torch
from torch import nn
from models.BigGAN import BigGAN, utils
from models.ProgGAN.model import Generator as ProgGenerator
from models.SNGAN.load import load_model_from_state_dict
from models.gan_with_shift import gan_with_shift

from utils import reg_brain
try:
    from models.StyleGAN2.model import Generator as StyleGAN2Generator
except Exception as e:
    print('StyleGAN2 load fail: {}'.format(e))

import sys
sys.path.insert(0, '../stylegan_xl/')

import legacy
import dnnlib
    
import h5py
from sklearn.linear_model import LinearRegression
import torch.nn.functional as F
    
class StyleGanXL(nn.Module):
    def __init__(self, G):
        super(StyleGanXL, self).__init__()
        self.style_gan_xl = G
        #self.dim_shift = self.style_gan_xl.w_dim
        #self.dim_z = self.style_gan_xl.w_dim
        
        self.dim_shift = self.style_gan_xl.z_dim
        self.dim_z = self.style_gan_xl.z_dim
        
    def forward(self, input):
        #w = torch.tile(torch.unsqueeze(input,dim=1),[1,self.style_gan_xl.num_ws,1])
        #w_avg = self.style_gan_xl.mapping.w_avg
        #w_avg = torch.mean(w_avg,0)
        #w = w_avg + (w - w_avg) * 0.8
        #return self.style_gan_xl.synthesis(w, noise_mode='const')
        cs = torch.zeros([input.shape[0], self.style_gan_xl.mapping.c_dim])
        cs[:,373] = 1
        w = self.style_gan_xl.mapping(input,cs.cuda())
        return self.style_gan_xl.synthesis(w, noise_mode='const')
        
    def gen_shifted(self, z, shift):
        return self.forward(z + shift)
    
class Brain(nn.Module):
    def __init__(self, G, neural_path, train_ws_path, roi):
        super(Brain, self).__init__()
        self.style_gan_xl = G
  
        ALL_n = np.linspace(0,1023,1024).astype('int')
        V1_n = np.linspace(0,511,512).astype('int')
        V4_n = np.linspace(512,767,256).astype('int')
        IT_n = np.linspace(768,1023,256).astype('int')
        rois = {'ALL':ALL_n, 'V1':V1_n, 'V4':V4_n,'IT':IT_n}
        idx_n = rois[roi]
        n_neurons = idx_n.shape[0]
        self.dim_shift = n_neurons
        self.dim_z = n_neurons
        
        #load neural data
        data_dict = {}
        f = h5py.File(neural_path,'r')
        for k, v in f.items():
            data_dict[k] = np.array(v)
        train_n_data = data_dict['train_MUA'][:,idx_n]
        self.data = train_n_data
        #load stim latents
        train_w_data = np.load(train_ws_path)[:,1,:]
        self.data1 = train_w_data
        #self.reg = LinearRegression().fit(train_n_data, train_w_data)
        #reg = LinearRegression().fit(train_n_data, train_w_data)
        #self.coef_ = torch.from_numpy(reg.coef_,).transpose(1,0).to(torch.float)
        #self.intercept_ = torch.from_numpy(reg.intercept_).to(torch.float)
        self.coef_,self.intercept_ =  reg_brain(train_n_data, train_w_data)
        
    def predict(self,input):
        #_pred_test_w_data = self.reg.predict(input.cpu().detach().numpy())
        #pred_test_w_data = np.repeat(_pred_test_w_data[None], self.style_gan_xl.mapping.num_ws, axis=0).transpose(1, 0, 2)
        #return torch.from_numpy(pred_test_w_data).to('cuda')
        _pred_test_w_data = torch.mm(input,self.coef_.cuda())+self.intercept_.cuda()
        return _pred_test_w_data.unsqueeze(1).repeat(1, self.style_gan_xl.mapping.num_ws, 1)
        
    def forward(self, input):
        w = self.predict(input)
        out = self.style_gan_xl.synthesis(w, noise_mode='const')
        return F.interpolate(out,(128,128))
        
    def gen_shifted(self, z, shift):
        return self.forward(z + shift)
    
    def data_augment(self,batch):
        all_batch = []
        for n in range(batch):
            all_batch.append(self.data[np.random.randint(0,self.data.shape[0]-1,size=(self.data.shape[0],1))[0]])
        all_batch = np.asarray(all_batch).squeeze()               
        return torch.from_numpy(all_batch).to(torch.float)
    
class ConditionedBigGAN(nn.Module):
    def __init__(self, big_gan, target_classes=(239)):
        super(ConditionedBigGAN, self).__init__()
        self.big_gan = big_gan

        self.target_classes = nn.Parameter(torch.tensor(target_classes, dtype=torch.int64),
            requires_grad=False)

        self.dim_z = self.big_gan.dim_z

    def set_classes(self, cl):
        try:
            cl[0]
        except Exception:
            cl = [cl]
        self.target_classes.data = torch.tensor(cl, dtype=torch.int64)

    def mixed_classes(self, batch_size):
        if len(self.target_classes.data.shape) == 0:
            return self.target_classes.repeat(batch_size).cuda()
        else:
            return torch.from_numpy(
                np.random.choice(self.target_classes.cpu(), [batch_size])).cuda()

    def forward(self, z, classes=None):
        if classes is None:
            classes = self.mixed_classes(z.shape[0]).to(z.device)
        return self.big_gan(z, self.big_gan.shared(classes))


class StyleGAN2Wrapper(nn.Module):
    def __init__(self, g, shift_in_w):
        super(StyleGAN2Wrapper, self).__init__()
        self.style_gan2 = g
        self.shift_in_w = shift_in_w
        self.dim_z = 512
        self.dim_shift = self.style_gan2.style_dim if shift_in_w else self.dim_z

    def forward(self, input, input_is_latent=False):
        return self.style_gan2([input], input_is_latent=input_is_latent)[0]

    def gen_shifted(self, z, shift):
        if self.shift_in_w:
            w = self.style_gan2.get_latent(z)
            return self.forward(w + shift, input_is_latent=True)
        else:
            return self.forward(z + shift, input_is_latent=False)


def make_biggan_config(weights_root):
    with open('models/BigGAN/generator_config.json') as f:
        config = json.load(f)
    config['weights_root'] = weights_root
    return config


@gan_with_shift
def make_big_gan(weights_root, target_class):
    config = make_biggan_config(weights_root)

    config['resolution'] = utils.imsize_dict[config['dataset']]
    config['n_classes'] = utils.nclass_dict[config['dataset']]
    config['G_activation'] = utils.activation_dict[config['G_nl']]
    config['D_activation'] = utils.activation_dict[config['D_nl']]
    config['skip_init'] = True
    config['no_optim'] = True

    G = BigGAN.Generator(**config)
    G.load_state_dict(torch.load(config['weights_root'], map_location='cpu'), strict=True)

    return ConditionedBigGAN(G, target_class).cuda()


@gan_with_shift
def make_proggan(weights_root):
    model = ProgGenerator()
    model.load_state_dict(torch.load(weights_root, map_location='cpu'))
    model.cuda()

    setattr(model, 'dim_z', [512, 1, 1])
    return model


@gan_with_shift
def make_sngan(gan_dir):
    gan = load_model_from_state_dict(gan_dir)
    G = gan.model.eval()
    setattr(G, 'dim_z', gan.distribution.dim)

    return G


def make_style_gan2(size, weights, shift_in_w=True):
    G = StyleGAN2Generator(size, 512, 8)
    G.load_state_dict(torch.load(weights, map_location='cpu')['g_ema'])
    G.cuda().eval()

    return StyleGAN2Wrapper(G, shift_in_w=shift_in_w)


def make_style_gan_xl(weights):
    with dnnlib.util.open_url(weights) as f:
        G = legacy.load_network_pkl(f)['G_ema']
        G = G.requires_grad_(False)
    G =  StyleGanXL(G)
    G.cuda().eval()
    return G


def make_brain(weights,neural_path,train_ws_path,roi):
    with dnnlib.util.open_url(weights) as f:
        G = legacy.load_network_pkl(f)['G_ema']
        G = G.requires_grad_(False).cuda()
    G =  Brain(G,neural_path,train_ws_path,roi)
    G.cuda().eval()
    return G
