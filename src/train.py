import torch 
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import sys
from base import GaussianSampler, LinearRegression,TransformerModel,Curriculum,TrainDataset,CidModel,CidLoss,get_model_size

def main(objective,method,gpu):

    device = torch.device(gpu if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    if objective == 'scale':
        n_dims = 10
        num_tasks = 512
        num_tasks_val = 32
        n_points = 500
        n_positions = 500
        batch_size = 32 
        epochs = 10000
        print_inc = 100
        n_points_curriculum = {
            'start':100,
            'end':n_points,
            'inc':50,
            'schedule':1000
        }
        n_dims_curriculum = {   
            'start':2,
            'end':n_dims,
            'inc':1,
            'schedule':1000
        }
    else:
        n_dims = 10
        num_tasks = 64
        num_tasks_val = 16
        n_points = 20
        n_positions = 20
        assert n_positions >= n_points
        epochs = 100
        batch_size = 4
        print_inc = 10
        n_points_curriculum = {
            'start':10,
            'end':n_points,
            'inc':10,
            'schedule':10
        }
        n_dims_curriculum = {   
            'start':2,
            'end':n_dims,
            'inc':1,
            'schedule':10
        }
    assert n_positions >= n_points
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    curriculum = Curriculum(n_points_curriculum,n_dims_curriculum)

    seeds_ws = list(range(1,1+num_tasks_val))
    seeds_xs = list(range(1+num_tasks_val,1+2*num_tasks_val))
    data_sampler = GaussianSampler(n_dims)
    #pool_dict = LinearRegression.generate_pool_dict(n_dims, num_tasks)
    task = LinearRegression(n_dims, num_tasks_val, seeds=seeds_ws)
    val_xs = data_sampler.sample_xs(n_points,num_tasks_val,n_dims,seeds=seeds_xs).to(device)
    val_ys = task.evaluate(val_xs).to(device)

    seeds_ws = list(range(2*num_tasks_val+1,num_tasks+2*num_tasks_val+1))
    seeds_xs = list(range(2*num_tasks_val+1+num_tasks,2*num_tasks_val+1+2*num_tasks))
    data_sampler = GaussianSampler(n_dims)
    #pool_dict = LinearRegression.generate_pool_dict(n_dims, num_tasks)
    task = LinearRegression(n_dims, num_tasks, seeds=seeds_ws) 
    X = data_sampler.sample_xs(curriculum.n_points,num_tasks,curriculum.n_dims_truncated,seeds=seeds_xs)
    Y = task.evaluate(X)
    dataset = TrainDataset(X,Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    if method == 'vanilla':
        vanillaTransformerModel = TransformerModel(n_dims=n_dims,n_positions=n_positions).to(device)
        print(f"Model size: {get_model_size(vanillaTransformerModel)} parameters")
        optimizer = torch.optim.Adam(vanillaTransformerModel.parameters(), lr=1e-3)
        loss_func = nn.CrossEntropyLoss() #nn.MSELoss()
        writer = SummaryWriter(comment=f'{objective}_{method}_transformer')
        for epoch in range(epochs):
            train_loss = 0
            for xs, ys in dataloader:
                vanillaTransformerModel.train()
                optimizer.zero_grad()
                output = vanillaTransformerModel(xs.to(device), ys[:,:,None].to(device))
                loss = loss_func(output, ys.to(device))
                loss.backward()
                optimizer.step()
                train_loss += loss.detach().item()

            vanillaTransformerModel.eval()
            with torch.no_grad():
                output = vanillaTransformerModel(val_xs, val_ys[:,:,None])
                val_loss = loss_func(output, val_ys).detach().item()

            if epoch % print_inc == 0:
                print(f'Epoch {epoch}: Train Loss: {train_loss}, Val Loss: {val_loss}')

            writer.add_scalar('Train/Loss', train_loss, global_step=epoch)
            writer.add_scalar('Val/Loss', val_loss, global_step=epoch)
            curriculum.update()

    elif method == 'cid':
        cidModel = CidModel(n_dims=n_dims,n_positions=n_positions,device=device).to(device) 
        print(f"Model size: {get_model_size(cidModel)} parameters")
        optimizer = torch.optim.Adam(cidModel.parameters(), lr=1e-3)
        loss_func = CidLoss() 
        writer = SummaryWriter(comment=f'{objective}_{method}_transformer')
        for epoch in range(epochs):
            train_loss = 0
            for xs, ys in dataloader:
                cidModel.train()
                optimizer.zero_grad()
                log_ppds,log_predictions = cidModel(xs.to(device), ys.to(device))
                loss = loss_func(log_ppds,log_predictions)
                loss.backward()
                optimizer.step()
                train_loss += loss.detach().item()

            cidModel.eval()
            with torch.no_grad():
                mean,std,_ = cidModel.cidTransformerModel(val_xs, val_ys)
                output = mean + std*torch.randn(num_tasks_val,n_points).to(device)
                val_loss = nn.CrossEntropyLoss()(output, val_ys).detach().item()

            if epoch % print_inc == 0:
                print(f'Epoch {epoch}: Train Loss: {train_loss}, Val Loss: {val_loss}')
            writer.add_scalar('Train/Loss', train_loss, global_step=epoch)
            writer.add_scalar('Val/Loss', val_loss, global_step=epoch)
            curriculum.update()


if __name__ == "__main__":
    objective = sys.argv[1] if len(sys.argv) > 1 else 'experimental'
    method = sys.argv[2] if len(sys.argv) > 2 else 'vanilla'
    gpu = sys.argv[3] if len(sys.argv) > 3 else 'cpu'
    main(objective='scale',method='cid',gpu=gpu)
    