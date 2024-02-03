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
        
        loss.backward()
        optimizer.step()
        curr_loss+=loss.sum().item()
        _,preds=torch.max(output,1)
        t_pred+=torch.sum(preds==targets.data).item()
        
        if batch_idx%10==0:
            print(
                "Train Epoch:{} [{}/{} ({:.0f}%)]\t loss: {:.6f}".format(
                    epoch,batch_idx*len(images),
                    len(train_loader.dataset),
                    100.0*batch_idx/len(train_loader),
                    loss.item(),
                )
            )
            train_loss.append(loss.sum().item()/len(images))
            train_acc.append(loss.sum().item()/len(images))
    epoch_loss=curr_loss/len(train_loader.dataset)
    epoch_acc=curr_loss/len(train_loader.dataset)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    
    print(
        "\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            epoch_loss,
            t_pred,
            len(train_loader.dataset),
            100.0 * t_pred / len(train_loader.dataset),
        )
    )

    return train_loss, train_acc, epoch_loss

def valid(model,device,test_loader,criterion,epoch,
          valid_loss,valid_acc,mse=None):
    """function is designed to perform the validation loop for a NN model.
    it evaluates the model on a specified validation dataset, computes the loss,
    and tracks validation metrics such as loss and accuracy
    """
    model.eval()
    test_loss=0
    correct=0
    
    with torch.no_grad():
        for batch_idx,(images,targets) in enumerate(test_loader):
            images,targets=images.to(device),targets.to(device)
            output=model(images).squeeze()
            loss=criterion(output,targets)
            test_loss+=loss.sum().item()
            _,preds=torch.max(output,1)
            correct+=torch.sum(preds=targets.data)
            if batch_idx % 10 == 0:
                print(
                    "Valid Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(images),
                        len(test_loader.dataset),
                        100.0 * batch_idx / len(test_loader),
                        loss.item(),
                    )
                )

                valid_loss.append(loss.sum().item() / len(images))
                valid_acc.append(preds.sum().item() / len(images))

    epoch_loss = test_loss / len(test_loader.dataset)
    epoch_acc = correct / len(test_loader.dataset)

    valid_loss.append(epoch_loss)
    valid_acc.append(epoch_acc.item())

    print(
        "Valid Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            epoch_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    return valid_loss, valid_acc
