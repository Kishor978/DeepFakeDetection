import torch 

def train(model,device,train_loader,criterion,
          optimizer,epoch,train_loss,train_acc,mse=None):
    """This function performs the training loop for NN model on a specified training dataset.
    It iterates over batches, computes the loss, performs backpropagation, updates the model parameters,
    and tracks training metrics such as loss and accuracy.
    Parameters:
            model: The neural network model to be trained.
            device: The device on which to perform training (CPU or GPU).
            train_loader: The PyTorch data loader providing training data.
            criterion: The loss criterion used to calculate the loss.
            optimizer: The optimization algorithm used for updating the model parameters.
            epoch: The current training epoch.
            train_loss: A list to store training losses during the training process.
            train_acc: A list to store training accuracies during the training process.
            mse: An optional parameter for Mean Squared Error loss.
            
        Returns:

            train_loss: A list containing training losses for each batch and the overall epoch.
            train_acc: A list containing training accuracies for each batch and the overall epoch.
            epoch_loss: The average loss over the entire training dataset for the current epoch.
            """
    model.train()
    curr_loss=0
    t_pred=0
    for batch_idx,(images,targets) in enumerate(train_loader):
        images,targets=images.to(device),targets.to(device)
        optimizer.zero_grad()
        output=model(images).squeeze()
        loss=criterion(output,targets)