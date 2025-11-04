##Code for emulator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import numpy as np
import os
import sys
from datetime import datetime
import h5py as h5
import logging

sys.path.append(os.path.dirname(__file__))
from config import cocoa_config

class Affine(nn.Module):
    def __init__(self):
        super(Affine, self).__init__()

        self.gain = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x * self.gain + self.bias

class Better_ResBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(Better_ResBlock, self).__init__()
        
        if in_size != out_size: 
            self.skip = nn.Linear(in_size, out_size, bias=False) # we don't consider this. remove?
        else:
            self.skip = nn.Identity()

        self.layer1 = nn.Linear(in_size, out_size)
        self.layer2 = nn.Linear(out_size, out_size)

        self.norm1 = Affine()#torch.nn.BatchNorm1d(in_size)
        self.norm3 = Affine()#torch.nn.BatchNorm1d(in_size)

        self.act1 = activation_fcn(in_size) #nn.Tanh()#nn.ReLU()#
        self.act3 = activation_fcn(in_size) #nn.Tanh()#nn.ReLU()#

    def forward(self, x):
        xskip = self.skip(x)

        o1 = self.act1(self.norm1(self.layer1(x)))
        o2 = self.layer2(o1) + xskip             #(self.norm2(self.layer2(o1))) + xskip
        o3 = self.act3(self.norm3(o2))

        return o3

class Better_Attention(nn.Module):
    def __init__(self, in_size ,n_partitions, dropout=False):
        super(Better_Attention, self).__init__()

        self.embed_dim    = in_size//n_partitions
        self.WQ           = nn.Linear(self.embed_dim,self.embed_dim)
        self.WK           = nn.Linear(self.embed_dim,self.embed_dim)
        self.WV           = nn.Linear(self.embed_dim,self.embed_dim)

        self.act          = nn.Softmax(dim=1) #NOT along the batch direction, apply to each vector.
        self.scale        = np.sqrt(self.embed_dim)
        self.n_partitions = n_partitions # n_partions or n_channels are synonyms 
        self.norm         = torch.nn.LayerNorm(in_size) # layer norm has geometric order (https://lessw.medium.com/what-layernorm-really-does-for-attention-in-transformers-4901ea6d890e)

        self.dropout = False
        #self.dropout = dropout
        if self.dropout:
            self.drop = nn.Dropout(p=0.1)
        else:
            self.drop = nn.Identity()

    def forward(self, x):
        x_norm    = self.norm(x)
        batch_size = x.shape[0]
        _x = x_norm.reshape(batch_size,self.n_partitions,self.embed_dim) # put into channels

        Q = self.WQ(_x) # query with q_i as rows
        K = self.WK(_x) # key   with k_i as rows
        V = self.WV(_x) # value with v_i as rows

        dot_product = torch.bmm(Q,K.transpose(1, 2).contiguous())
        normed_mat  = self.act(dot_product/self.scale)
        prod        = torch.bmm(normed_mat,V)

        #out = torch.cat(tuple([prod[:,i] for i in range(self.n_partitions)]),dim=1)+x
        out = self.drop(torch.reshape(prod,(batch_size,-1)))+x # reshape back to vector

        return out

class Better_Transformer(nn.Module):
    def __init__(self, in_size, n_partitions, dropout=False):
        super(Better_Transformer, self).__init__()  
        # get/set up hyperparams
        self.in_size      = in_size
        self.int_dim      = in_size//n_partitions 
        self.n_partitions = n_partitions
        self.act          = activation_fcn(in_size)  #nn.Tanh()   #nn.ReLU()#
        self.norm         = torch.nn.BatchNorm1d(in_size)
        #self.act2         = nn.Tanh()#nn.ReLU()#
        #self.norm2        = torch.nn.BatchNorm1d(in_size)
        self.act3         = activation_fcn(in_size)  #nn.Tanh()
        self.norm3        = torch.nn.BatchNorm1d(in_size)

        # set up weight matrices and bias vectors
        weights1 = torch.zeros((n_partitions,self.int_dim,self.int_dim))
        self.weights1 = nn.Parameter(weights1) # turn the weights tensor into trainable weights
        bias1 = torch.Tensor(in_size)
        self.bias1 = nn.Parameter(bias1) # turn bias tensor into trainable weights

        weights2 = torch.zeros((n_partitions,self.int_dim,self.int_dim))
        self.weights2 = nn.Parameter(weights2) # turn the weights tensor into trainable weights
        bias2 = torch.Tensor(in_size)
        self.bias2 = nn.Parameter(bias2) # turn bias tensor into trainable weights

        # initialize weights and biases
        # this process follows the standard from the nn.Linear module (https://auro-227.medium.com/writing-a-custom-layer-in-pytorch-14ab6ac94b77)
        nn.init.kaiming_uniform_(self.weights1, a=np.sqrt(5)) # matrix weights init 
        fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.weights1) # fan_in in the input size, fan out is the output size but it is not use here
        bound1 = 1 / np.sqrt(fan_in1) 
        nn.init.uniform_(self.bias1, -bound1, bound1) # bias weights init

        nn.init.kaiming_uniform_(self.weights2, a=np.sqrt(5))  
        fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.weights2)
        bound2 = 1 / np.sqrt(fan_in2) 
        nn.init.uniform_(self.bias2, -bound2, bound2)

        self.trained = False

        #self.dropout = dropout
        #if self.dropout:
        #    self.drop = nn.Dropout(p=0.1)
        #else:
        #    self.drop = nn.Identity()

        #Cache for block diagonal matrices (computed once in eval mode)
        self._cached_mat1 = None
        self._cached_mat2 = None
        self._cache_device = None

    def _build_block_matrices(self, device):
        if self._cached_mat1 is None or self._cache_device != device:
            self._cached_mat1 = torch.block_diag(*self.weights1).to(device)
            self._cached_mat2 = torch.block_diag(*self.weights2).to(device)
            self._cache_device = device


    def forward(self,x):
        #Build cached matrices if needed
        if self.trained:
            self._build_block_matrices(x.device)
            mat1 = self._cached_mat1
            mat2 = self._cached_mat2
            self.drop = nn.Identity()
        else:
            #In training mode need to build matrices each time for gradient computation
            mat1 = torch.block_diag(*self.weights1)
            mat2 = torch.block_diag(*self.weights2)
            self.drop = nn.Dropout(p=0.1)

        o1 = self.norm(torch.matmul(x,mat1)+self.bias1)
        o2 = self.act(o1)
        o3 = self.drop(torch.matmul(o1,mat2) + self.bias2) + x
        o4 = self.act3(o3)
        return o4

class activation_fcn(nn.Module):
    def __init__(self, dim):
        super(activation_fcn, self).__init__()

        self.dim = dim
        self.gamma = nn.Parameter(torch.zeros((dim)))
        self.beta = nn.Parameter(torch.zeros((dim)))

    def forward(self,x):
        exp = torch.mul(self.beta,x)
        inv = torch.special.expit(exp)
        fac_2 = 1-self.gamma
        out = torch.mul(self.gamma + torch.mul(inv,fac_2), x)
        return out

class nn_emulator:
    def __init__(self, preset=None, model=None, output_dim=None, input_dim=None,res_size=256,trf_size1=None,trf_size2=None,dropout=False):

        layers = []

        if ( preset is None and model is None ):
            raise Exception('No preset or model was provided.')

        elif (preset is not None and model is not None ):
            raise Exception('Both a preset and a model were provide.\nOnly provide one or the other, not both!')

        elif ( preset is None and model is not None ):
            self.model = model

        elif ( preset is not None and model is None):
            if ( preset == 'xi_restrf' ):
                self.start = 0
                self.stop  = output_dim
                
                # Update network architecture to match input dimension of 15
                layers.append(nn.Linear(input_dim, 256))
                layers.append(Better_ResBlock(256, 256))
                layers.append(Better_ResBlock(256, 256))
                layers.append(Better_ResBlock(256, 256))
                layers.append(nn.Linear(256, 1024))
                layers.append(Better_Attention(1024, 32, dropout))
                layers.append(Better_Transformer(1024, 32, dropout))
                layers.append(Better_Attention(1024, 32))
                layers.append(Better_Transformer(1024, 32))
                layers.append(Better_Attention(1024, 32))
                layers.append(Better_Transformer(1024, 32))
                layers.append(nn.Linear(1024, output_dim))
                layers.append(Affine())

            elif ( preset == '3x2_restrf' ):
                self.start = 0 
                self.stop  = output_dim
                
                layers.append(nn.Linear(input_dim, 512))
                layers.append(Better_ResBlock(512, 512))
                layers.append(Better_ResBlock(512, 512))
                layers.append(Better_ResBlock(512, 512))
                layers.append(nn.Linear(512, 3840))
                layers.append(Better_Attention(3840, 60, dropout))
                layers.append(Better_Transformer(3840, 60, dropout))
                layers.append(Better_Attention(3840, 60))
                layers.append(Better_Transformer(3840, 60))
                layers.append(Better_Attention(3840, 60))
                layers.append(Better_Transformer(3840, 60))
                layers.append(nn.Linear(3840, output_dim))
                layers.append(Affine())
            
            elif ( preset == 'restrf_gen'):
                self.start = 0 
                self.stop  = output_dim
                
                layers.append(nn.Linear(input_dim, res_size))
                layers.append(Better_ResBlock(res_size, res_size))
                layers.append(Better_ResBlock(res_size, res_size))
                layers.append(Better_ResBlock(res_size, res_size))
                layers.append(nn.Linear(res_size, trf_size1))
                layers.append(Better_Attention(trf_size1, trf_size2, dropout))
                layers.append(Better_Transformer(trf_size1, trf_size2, dropout))
                layers.append(Better_Attention(trf_size1, trf_size2))
                layers.append(Better_Transformer(trf_size1, trf_size2))
                layers.append(Better_Attention(trf_size1, trf_size2))
                layers.append(Better_Transformer(trf_size1, trf_size2))
                layers.append(nn.Linear(trf_size1, output_dim))
                layers.append(Affine())

            elif ( preset == 'resnet_gen'):
                self.start = 0
                self.stop  = output_dim

                layers.append(nn.Linear(input_dim, res_size))
                layers.append(Better_ResBlock(res_size, res_size))
                layers.append(Better_ResBlock(res_size, res_size))
                layers.append(Better_ResBlock(res_size, res_size))
                layers.append(nn.Linear(res_size, output_dim))
                layers.append(Affine())

            else:
                raise Exception('Preset is not known!')

        self.model = nn.Sequential(*layers)
        #self.trained = False

    def update_progress(self, train_loss, valid_loss, start_time, epoch, total_epochs, optim):
        elapsed_time = int((datetime.now() - start_time).total_seconds())
        lr = optim.param_groups[0]['lr']
        epoch=epoch+1

        width = 20
        factor = int( width * (epoch/total_epochs) )
        bar = '['
        for i in range(width):
            if i < factor:
                bar += '#'
            else:
                bar += ' '
        bar += ']'

        remaining_time = int((elapsed_time / (epoch)) * (total_epochs - (epoch)))

        print('\r' + bar + ' ' +                                \
              f'Epoch {epoch:3d}/{total_epochs:3d} | ' +        \
              f'loss={train_loss:1.3e}({valid_loss:1.3e}) | ' + \
              f'lr={lr:1.2e} | ' +                              \
              f'time elapsed={elapsed_time:7d} s; time remaining={remaining_time:7d} s',end='')

    def train(self, device, config_file,
            x_train, y_train,
            x_valid, y_valid,
            n_epochs=150, batch_size=100, learning_rate=1e-3, reduce_lr=True, weight_decay=0,
            save_losses=False):

        #summary(self.model)
        print('Batch size = ',batch_size)
        print('N_epochs = ',n_epochs)

        config = cocoa_config(config_file)

        print('Loading and processing the data. May take some time...')
        
        # Get the mask and apply it to get the masked covariance and fiducial datavector
        mask = config.mask
        covmat = torch.as_tensor(config.cov[mask][:, mask], dtype=torch.float64)
        self.dv_fid = torch.as_tensor(config.dv_fid[mask], dtype=torch.float64)

        #self.dv_fid = torch.as_tensor(config.dv_fid[mask], dtype=torch.float64)
        self.dv_evals, self.dv_evecs = torch.linalg.eigh(covmat)

        x_train = torch.as_tensor(x_train)
        y_train = torch.as_tensor(y_train)
        x_valid = torch.as_tensor(x_valid)
        y_valid = torch.as_tensor(y_valid)
    
        self.samples_mean = torch.mean(x_train, dim=0, keepdim=True)
        self.samples_std = torch.std(x_train, dim=0, keepdim=True)
        
        x_train = (x_train - self.samples_mean) / self.samples_std
        x_valid = (x_valid - self.samples_mean) / self.samples_std

        
        #eps = 1e-8
        #safe_evals = torch.clamp(self.dv_evals, min=eps)

        y_train = ((y_train@self.dv_evecs) - (self.dv_fid @ self.dv_evecs)) / torch.sqrt(self.dv_evals)
        y_valid = ((y_valid@self.dv_evecs) - (self.dv_fid @ self.dv_evecs)) / torch.sqrt(self.dv_evals)

        #print(x_train.mean().item(), x_train.std().item())  # should be ~0, ~1
        #print(y_train.mean().item(), y_train.std().item())
        #print(y_valid.mean().item(), y_valid.std().item())

        #y_train = (y_train - self.dv_fid) @ self.dv_evecs / torch.sqrt(safe_evals)
        #y_valid = (y_valid - self.dv_fid) @ self.dv_evecs / torch.sqrt(safe_evals)
        # Transform outputs using eigendecomposition
        #y_train = torch.div((y_train - self.dv_fid) @ self.dv_evecs, torch.sqrt(self.dv_evals))
        #y_valid = torch.div((y_valid - self.dv_fid) @ self.dv_evecs, torch.sqrt(self.dv_evals))


        # initialize arrays
        losses_train = []
        losses_vali = []
        loss = 100.

        # send everything to device
        self.model.to(device)
        
        # convert to float32
        x_train = x_train.to(torch.float32)
        y_train = y_train.to(torch.float32)
        x_valid = x_valid.to(torch.float32)
        y_valid = y_valid.to(torch.float32)


        # setup ADAM optimizer and reduce_lr scheduler
        optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min',patience=15,factor=0.1)

        trainset = torch.utils.data.TensorDataset(x_train, y_train)
        validset = torch.utils.data.TensorDataset(x_valid, y_valid)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
        validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)

    
        print('Datasets loaded!')
        print('Begin training...')
        train_start_time = datetime.now()
        for e in range(n_epochs):
            start_time = datetime.now()
            self.model.train()
            losses = []
            for i, data in enumerate(trainloader):    
                X       = data[0].to(device)
                Y_batch = data[1].to(device)
                Y_pred = self.model(X)

                diff = Y_batch - Y_pred
                chi2 = torch.diag(diff @ torch.t(diff))

                #loss = torch.mean(chi2)                      # ordinary chi2
                loss = torch.mean((1+2*chi2)**(1/2))-1       # hyperbola
                # loss = torch.mean(torch.mean(chi2**(1/2)))   # sqrt(chi2)

                losses.append(loss.cpu().detach().numpy())
                optim.zero_grad()
                loss.backward()
                optim.step()

            losses_train.append(np.mean(losses))

            ###validation loss
            losses = []
            with torch.no_grad():
                self.model.eval()
                for i, data in enumerate(validloader):  
                    X_v       = data[0].to(device)
                    Y_v_batch = data[1].to(device)
                    Y_v_pred = self.model(X_v)

                    diff_v = Y_v_batch - Y_v_pred
                    chi2_v = torch.diag(diff_v @ torch.t(diff_v))

                    #loss_vali = torch.mean(chi2_v)                      # ordinary chi2
                    loss_vali = torch.mean((1+2*chi2_v)**(1/2))-1       # hyperbola
                    # loss_vali = torch.mean(torch.mean(chi2_v**(1/2)))   # sqrt(chi2)

                    losses.append(loss_vali.cpu().detach().numpy())

                losses_vali.append(np.mean(losses))
                scheduler.step(losses_vali[e])

            self.update_progress(losses_train[-1],losses_vali[-1],train_start_time, e, n_epochs, optim)
        
        if ( save_losses ):
            np.savetxt("losses.txt", np.array([losses_train,losses_vali],dtype=np.float64))

        self.trained = True
        print('\nDone!')

    def predict(self, X):
        #assert self.trained, "The emulator needs to be trained first before predicting"

        with torch.no_grad():
            y_pred = self.model((torch.Tensor(X) - self.samples_mean) / self.samples_std)

        #y_pred = (y_pred * torch.sqrt(self.dv_evals)) @ torch.linalg.inv(self.dv_evecs) + self.dv_fid
        y_pred = (y_pred * torch.sqrt(self.dv_evals)) @ self.dv_evecs_T + self.dv_fid
        return y_pred.cpu().detach().numpy()
    
    def predict_full(self, X):
        """
        Predict and expand back to full datavector using the mask
        """
        # Get masked prediction
        y_pred_masked = self.predict(X)
        
        
        # Create full datavector with zeros for masked elements
        y_pred_full = np.zeros((y_pred_masked.shape[0], len(self.mask)))
        y_pred_full[:, self.mask] = y_pred_masked
        
        return y_pred_full

    def save(self, filename):
        #root = './external_modules/data/lsst_y1_cosmic_shear_emulator/'
        #model_cpu = self.model.cpu()
        torch.save(self.model.state_dict(), filename)
        with h5.File(filename + '.h5', 'w') as f:
            f['sample_mean']   = self.samples_mean.cpu()
            f['sample_std']    = self.samples_std.cpu()
            f['dv_fid']        = self.dv_fid.cpu()
            f['dv_evals']      = self.dv_evals.cpu()
            f['dv_evecs']      = self.dv_evecs.cpu()
            # Store the output dimension for reference
            f['output_dim']    = self.dv_fid.shape[0]
        
    def load(self, filename, config_file, device=torch.device('cpu'),state_dict=True):
        self.trained = True
        #if device!=torch.device('cpu'):
        #    print("hey, you are using a GPU!")
        #    torch.set_default_dtype('torch.cuda.FloatTensor')
        #else:
        #    torch.set_default_dtype('torch.FloatTensor')

        for layer in self.model:
            if isinstance(layer, Better_Transformer):
                layer.trained = True

        if state_dict==False:
            self.model = torch.load(filename,map_location=device)
        else:
            state_dict = torch.load(filename, map_location=device)
            self.model.load_state_dict(state_dict)

        self.model.eval()
        self.model.to(device)

        
        with h5.File(filename + '.h5', 'r') as f:
            self.samples_mean  = torch.Tensor(f['sample_mean'][:]).to(device)
            self.samples_std   = torch.Tensor(f['sample_std'][:]).to(device)
            self.dv_fid        = torch.Tensor(f['dv_fid'][:]).to(device)
            self.dv_evals      = torch.Tensor(f['dv_evals'][:]).to(device)
            self.dv_evecs      = torch.Tensor(f['dv_evecs'][:]).to(device)
            # Load output dimension for reference
            self.output_dim    = f['output_dim'][()]

        self.dv_evecs_T = self.dv_evecs.T.contiguous()

        self.config = cocoa_config(config_file)
        self.mask = self.config.mask

        print('Loaded emulator')
