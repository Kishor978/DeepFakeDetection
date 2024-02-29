import sys,os
import numpy as np 
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import time
from time import perf_counter
import pickle
import optparse

from model.config import load_config
from dataloader import load_data,load_checkpoint
from model.autoencoder import ConvAutoEncoder
from model.varient_autoencoder import ConvVAE

config=load_config()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(dir_path,mod,num_epochs,pretrained_model_filename,
                test_model, batch_size):
    """This function is designed to train a CNN model, either an Autoencoder (AE)
    or a Variational Autoencoder (VAE), based on the specified mode. It uses the 
    PyTorch framework for training and validation, saving checkpoints, and optionally 
    testing the model. The function supports training from scratch or continuing training from a pretrained model.

    Parameters:
        dir_path: The directory path containing the training, validation, and test datasets.
        mod: A string indicating the mode of the model, either "ed" (Autoencoder) or another value (VAE).
        num_epochs: The number of training epochs.
        pretrained_model_filename: The filename/path of the pretrained model checkpoint file (optional).
        test_model: A boolean indicating whether to perform testing after training.
        batch_size: The batch size for training.
    """
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
                                   criterion,epoch,valid_loss,valid_acc,mse)
        
        scheduler.step()
    
    time_elapsed=time.time() -since
    
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    print("\nSaving model....\n")
    os.makedirs("weight", exist_ok=True)
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
        
def load_pretrained(pretrained_model_filename):
    """function is loads a pretrained model and its optimizer from a saved checkpoint file.
    It checks the existence of the specified checkpoint file, attempts to load the model, 
    optimizer, starting epoch, and minimum loss using a load_checkpoint function 
    and then moves the optimizer's state to the specified device.

"""
    assert os.path.isfile(
        pretrained_model_filename), "Saved model file does not exist...\nExiting!!!!"
    model,optimizer,start_epoch,min_loss=load_checkpoint(
        model, optimizer, filename=pretrained_model_filename
    )
    
    for state in optimizer.state.values():
        for k,v in state.item():
            if isinstance(v,torch.Tensor):
                state[k]=v.to(device)
    return model,optimizer,start_epoch,min_loss


def test(model,dataloaders,dataset_size,mod,weight):
    """It loads a pretrained model from a given checkpoint file, sets the model to 
    evaluation mode, and iterates over batches of the test dataset to make predictions.
    It compares the predicted labels with the ground truth labels, counts the number of 
    correct predictions, and prints the overall accuracy.

    Parameters:
        model: The PyTorch model to be evaluated.
        dataloaders: A dictionary containing PyTorch data loaders for the test dataset.
        dataset_sizes: A dictionary containing the sizes of different datasets (e.g., "test").
        mod: A string indicating the mode ("ed" or other) used for prediction.
        weight: The filename/path of the checkpoint file containing pretrained weights.
    """
    print("\nRunning test.....\n")
    model.eval()
    check_point=torch.load(weight,map_location="cpu")
    model.load_state_dict(check_point["state_dict"])
    _=model.eval()
    
    Sum=0
    counter=0
    for inputs,labels in dataloaders["test"]:
        inputs=inputs.to(device)
        labels=labels.to(device)
        if mod== "ed":
            output=model(inputs).to(device).float()
        else:
            output=model(inputs)[0].to(device).float()
        
        _,prediction=torch.max(output,1)
        
        pred_label=labels[prediction]
        pred_label=pred_label.detach().cpu().numpy()
        main_label=labels.detach().cpu().numpy()
        bool_list = list(map(lambda x, y: x == y, pred_label, main_label))
        Sum+=sum(np.array(bool_list)*1)
        counter+=1
        print(f"Prediction:{Sum}/{len(inputs)*counter}")
    print(f'Prediction:{Sum}/{dataset_size["test"]}{(Sum/dataset_size["test"])*100:.2f}%')
    
def gen_parser():
    """This function creates and configures an option parser for command-line arguments.
    It parses the provided command-line arguments and returns a tuple containing the parsed values.
    """
    
    parser=optparse.OptionParser("Train ConSwinT model")
    parser.add_option("-e","--epoch",type=int,dest="epoch",
                      help="Number of epochs used for training the ConSwinT.")
    parser.add_option("-v","--version",dest="version",help="version 0.1.1")
    parser.add_option("-d","--dir",dest="dir",help="Training data path")
    parser.add_option("-m","--model",dest="model",help="model ed or vae")
    parser.add_option("-p","--pretrained",dest="pretrained",
                      help="Saved model file name. If you want to continue from the previous trained model")
    parser.add_option("-t","--test",dest="test",help="run test on test dataset")
    parser.add_option("-b","--batch_size",dest="batch_size",help="batch size.")
    
    (options,_)=parser.parse_args()
    dir_path=options.dir 
    epoch=options.epoch
    mod="ed" if options.model=="ed" else "vae"
    test_model="y" if options.test else None
    pretrained_model_filename=options.pretrained if options.pretrained else None
    batch_size=options.batch_size if options.batch_size else config["batch_size"]
    
    return dir_path,mod,epoch,pretrained_model_filename,test_model,int(batch_size)

def main():
    start_time = perf_counter()
    path, mod, epoch, pretrained_model_filename, test_model, batch_size = gen_parser()
    train_model(path, mod, epoch, pretrained_model_filename, test_model, batch_size)
    end_time = perf_counter()
    print("\n\n--- %s seconds ---" % (end_time - start_time))


if __name__ == "__main__":
    main()
