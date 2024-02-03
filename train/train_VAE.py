import torch

def train(model,device,train_loader,criterion,optimizer,
          epoch,train_loss,train_acc,mse):
    model.train()
    curr_loss=0
    t_pred=0
    
    for batch_idx, (images,targets) in enumerate(train_loader):
        images,targets=images.to(device), targets.to(device)
        optimizer.zero_grad()
        output,recons=model(images)
        loss_m=criterion(output,targets)
        vae=mse(recons,images)
        loss=loss_m+vae
        
        loss.backward()
        optimizer.step()
        
        curr_loss+=loss.sum().item()
        _,preds=torch.max(output,1)
        t_pred+= torch.sum(preds==targets.data).item()
        
        if batch_idx % 10 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} vae_Loss {:.6f}".format(
                    epoch,
                    batch_idx * len(images),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss_m.item(),
                    vae.item(),
                )
            )
            train_loss.append(loss.sum().item()/len(images))
            train_acc.append(loss.sum().item()/len(images))
    epoch_loss= curr_loss/len(train_loader.dataset)
    epoch_acc=t_pred/len(train_loader.dataset)
    
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
    return train_loss,train_acc,epoch_loss
