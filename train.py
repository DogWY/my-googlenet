import torch

from tqdm import tqdm

def train_epoch(model, data_loader, optimizer, criterion, epoch, num_epochs, device='cpu'):
    total_correct = 0
    total_loss = 0
    total_item = 0
    
    model.train()

    pbar = tqdm(data_loader)
    for image, label in pbar:
        image, label = image.to(device), label.to(device)
        pred, aux1, aux2 = model(image)
        
        loss_1 = criterion(pred, label)
        loss_2 = criterion(aux1, label)
        loss_3 = criterion(aux2, label)
        loss = loss_1 + 0.3 * loss_2 + 0.3 * loss_3
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * image.shape[0]
        
        pred_label = torch.argmax(pred, dim=1)
        correct = (pred_label == label).sum()
        total_correct += correct
        
        total_item += image.shape[0]
        
        pbar.set_description(f"[{epoch}/{num_epochs}] Train Batch Loss: {loss.item():.4f}, Acc: {correct/image.shape[0]:.4f}")

    pbar.write(f"[{epoch}/{num_epochs}] Train Total Loss: {total_loss/total_item:.4f}, Acc: {total_correct/total_item:.4f}")
    
    return total_loss, total_correct
    
    