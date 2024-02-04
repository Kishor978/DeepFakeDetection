import sys,os
import numpy as np 
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import time
from time import perf_counter
import pickle

from model.config import load_config
from dataloader import load_data,load_checkpoint
from model.autoencoder import ConvAutoEncoder
from model.varient_autoencoder import ConvVAE

config=load_config()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(dir_path,mod,num_epochs,pretrained_model_filename,
                test_model, batch_size):
    print("Loading data....")
    dataloaders,dataset_size=load_data(dir_path,batch_size)
    print("Loadibg data is completed...")
    
    if mod=="ed":
        from train.train_autoencoder import train,valid
        model=ConvAutoEncoder(config)
    
    else:
        from train.train_VAE import train,valid
        model=ConvVAE(config)
    
    optimizer=optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    
    criterion=nn.CrossEntropyLoss()
    criterion.to(device)
    mse=nn.MSELoss()
    min_val_loss=int(config["min_val_loss"])
    scheduler=lr_scheduler.StepLR(optimizer,step_size=15,gamma=0.1)
    
    if pretrained_model_filename:
        model,optimizer,start_epoch,min_loss=load_pretrained(
            pretrained_model_filename
        )
        
    model.to(device)
    torch.manual_seed(1)
    train_loss,train_acc,valid_loss,valid_acc=[],[],[],[]
    since=time.time()
    
    for epoch in range(0,num_epochs):
        train_loss,train_acc,epoch_loss=train(model,device,dataloaders["train"],
                                              criterion,optimizer,epoch,
                                              train_loss,train_acc,mse)
        
        valid_loss,valid_acc=valid(model,device,dataloaders["validation"],
                                   criterion,epoch,valid_acc,valid_loss,mse)
        
        scheduler.step()
    
    time_elapsed=time.time() -since
    
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    print("\nSaving model....\n")
    
    file_path=os.path.join("weight",
        f'convSwinT_{mod}_{time.strftime("%b_%d_%Y_%H_%M_%S",time.localtime())}')
    
    with open(f"{file_path}.pkl", "wb") as f:
        pickle.dump([train_loss, train_acc, valid_loss, valid_acc], f)

    state={
        "epoch": num_epochs+1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "min_loss": epoch_loss,       
    }
    
    weight=f"{file_path}.pth"
    torch.save(state,weight)
    
    print("Done...")
    
    if test_model:
        test(model,dataloaders,dataset_size,mod,weight )