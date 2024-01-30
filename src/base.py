import torch 
import torch.nn as nn
import numpy as np
from transformers import GPT2Model, GPT2Config
from torch.utils.data import Dataset
import pdb as pdb
from torch.utils.data import DataLoader


class GaussianSampler():
    def __init__(self, n_dims, bias=None, scale=None):
        self.bias = bias
        self.scale = scale
        self.n_dims = n_dims

    def sample_xs(self, n_points, num_tasks, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.randn(num_tasks, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(num_tasks, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == num_tasks
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b
    

class LinearRegression():
    def __init__(self, n_dims, num_tasks, pool_dict=None, seeds=None, scale=1):
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.w_b = torch.randn(num_tasks, n_dims, 1)
        elif seeds is not None:
            self.w_b = torch.zeros(num_tasks, n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == num_tasks
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(n_dims, 1, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:num_tasks]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):  
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return lambda ys_pred, ys : (ys - ys_pred).square()

    @staticmethod
    def get_training_metric():
        return lambda ys_pred, ys : (ys - ys_pred).square().mean()
        


class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        batch , points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(batch, points, 1),
                torch.zeros(batch, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(batch, 2 * points, dim)
        return zs

    def forward(self, xs, ys):
        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        return prediction[:, ::2, 0]  # predict only on xs
    

class Curriculum:
    def __init__(self, n_points_curriculum,n_dims_curriculum):
        self.n_points_curriculum = n_points_curriculum
        self.n_dims_curriculum = n_dims_curriculum
        self.n_points = n_points_curriculum['start']
        self.n_dims_truncated = n_dims_curriculum['start']
        self.step_count = 0

    def update(self):
        self.step_count += 1
        self.n_dims_truncated = self.update_var(
            self.n_dims_truncated, self.n_dims_curriculum
            )
        self.n_points = self.update_var(
            self.n_points, self.n_points_curriculum
            )

    def update_var(self, var, schedule):
        if self.step_count % schedule['schedule'] == 0:
            var += schedule['inc']
        return min(var, schedule['end'])
    
class TrainDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x_sample = self.X[index]
        y_sample = self.Y[index]
        return x_sample, y_sample
    
class CopulaDensityModel(nn.Module): # copula density composed with exp function not copula 
    def __init__(self):
        super(CopulaDensityModel, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(), # output in [0,1]
        ) # think about forcing other structure
    def forward(self,u,v): # u,v are 3D tensors : batch x (points-1) x dim
        return self.backbone(torch.cat([u.reshape(-1,1),v.reshape(-1,1)],dim=-1)).reshape(u.shape)
    

class CidTransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(CidTransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions= 2*n_positions, # maybe n_positions (reduce gpu memory)
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(1, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 2)
        self.const = - 0.5*np.log(2*torch.pi)

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence of 1D tensors."""
        batch, points, dim = xs_b.shape
        zs = torch.cat((xs_b, ys_b[:,:,None]), dim=2)
        zs = zs.view(batch, points*(dim+1),1)
        return zs

    def forward(self, xs, ys):
        _, points, dim = xs.shape
        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        mean  = self._read_out(output)[:,:,0]
        #std = torch.exp(self._read_out(output)[:,:,1])
        log_std = self._read_out(output)[:,:,1]
        log_ppd = -0.5*((zs[:,:,0]-mean)/torch.exp(log_std))**2 - log_std + self.const
        x2y_prediction = torch.arange(points*(dim+1))%(dim+1)==(dim-1)
        x2x_prediction = torch.arange(points*(dim+1))%(dim+1)!=dim
        return mean[:,x2y_prediction], log_std[:,x2y_prediction], log_ppd[:,x2x_prediction]
    
class CidModel(nn.Module):
    def __init__(self, n_dims, n_positions, device):
        super(CidModel, self).__init__()
        self.copulaYmodel = CopulaDensityModel().to(device)
        self.copulaXmodel = CopulaDensityModel().to(device)
        self.cidTransformerModel = CidTransformerModel(n_dims=n_dims,n_positions=(n_dims+1)*n_positions).to(device)
        self.alphas = 1/(torch.arange(end=n_positions-1,device=device)+1)
    def forward(self,xs,ys): 
        batch , points, dim = xs.shape
        log_ppd = self.cidTransformerModel(xs,ys)[-1] # batch x (points*dims)
        # test if log_ppd conatins nan
        #if torch.isnan(log_ppd).any():
        #pdb.set_trace()
        log_ppd = log_ppd.view(batch,points,dim) # batch x points x dim 
        log_ppd_f = log_ppd[:,1:,:] #.transpose(1,2)  # batch x (points-1) x dim 
        log_ppd = log_ppd[:,:-1,:] #.transpose(1,2) # batch x (points-1) x dim
        copulaXfactor = torch.exp(torch.sum(torch.log(self.copulaXmodel(log_ppd_f,log_ppd)),dim=-1)) # batch x (points-1)
        copulaYfactor = self.copulaYmodel(log_ppd_f[:,:,-1],log_ppd[:,:,-1]) # batch x (points-1)
        alphas = self.alphas[:points-1]
        result = torch.log(1-alphas+alphas*copulaYfactor*copulaXfactor)-torch.log(1-alphas+alphas*copulaXfactor) # batch x (points-1)
        return torch.cumsum(result,dim=-1),log_ppd[:,:,-1]    
    

class CidLoss(nn.Module):
    def __init__(self):
        super(CidLoss, self).__init__()
    def forward(self, log_ppd_2match,log_ppd):
        #loss_value = (log_ppds-log_predictions).square().mean()
        loss_value = ((log_ppd-log_ppd_2match)*torch.exp(log_ppd)).mean()
        return loss_value

def get_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def get_train_dataloader(curriculum, batch_size, n_dims, num_tasks, start):
    seeds_ws = list(range(start,num_tasks+start))
    seeds_xs = list(range(start+num_tasks,start+2*num_tasks))
    data_sampler = GaussianSampler(n_dims)
    task = LinearRegression(n_dims, num_tasks, seeds=seeds_ws) 
    X = data_sampler.sample_xs(curriculum.n_points,num_tasks,curriculum.n_dims_truncated,seeds=seeds_xs)
    Y = task.evaluate(X)
    dataset = TrainDataset(X,Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def get_val_data(n_points,n_dims,num_tasks_val,device):
    seeds_ws = list(range(1,1+num_tasks_val))
    seeds_xs = list(range(1+num_tasks_val,1+2*num_tasks_val))
    data_sampler = GaussianSampler(n_dims)
    task = LinearRegression(n_dims, num_tasks_val, seeds=seeds_ws)
    val_xs = data_sampler.sample_xs(n_points,num_tasks_val,n_dims,seeds=seeds_xs).to(device)
    val_ys = task.evaluate(val_xs).to(device)
    return val_xs,val_ys