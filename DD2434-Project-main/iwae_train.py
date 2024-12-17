import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from iwae_models import IWAE
import os
from typing import Callable

def train_one_epoch(model: IWAE,
                    loss_fn: Callable,
                    optimizer: optim.Optimizer,
                    training_loader: DataLoader,
                    device: any,
                    epoch_index: int = 0,
                    tb_writer: SummaryWriter | None = None) -> None:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(pbar := tqdm(training_loader)):
        pbar.set_description(f'Epoch {epoch_index+1} | avg loss {running_loss / (i+1)}')

        # Every data instance is an input + label pair
        x, label = data
        x = x.to(device)

        # Zero gradients for every batch.
        optimizer.zero_grad()

        # Make predictions for this batch.
        y = model(x)

        # Compute the loss and its gradients.
        loss = loss_fn(x, label, *y)
        loss.backward()

        # Adjust learning weights.
        optimizer.step()

        # Gather data and report.
        running_loss += loss.item()
    last_loss = running_loss / len(training_loader) # loss per batch
    if tb_writer is not None:
        tb_x = epoch_index * len(training_loader) + i + 1
        tb_writer.add_scalar('Loss/train', last_loss, tb_x)
    return last_loss

def train(model: nn.Module,
          loss_fn: nn.Module,
          optimizer: optim.Optimizer,
          training_loader: DataLoader,
          test_loader: DataLoader,
          num_epochs: int = 20,
          save_model: bool = True,
          write_tb: bool = True,
          device: any = None) -> None:
  if device is None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  if write_tb:
    tb_writer = SummaryWriter('runs/trainer_{}'.format(timestamp))
  else:
    tb_writer = None
  # Initializing in a separate cell so we can easily add more epochs to the same run
  epoch_number = 0

  best_tloss = np.inf
  for epoch in range(num_epochs):
      # Make sure gradient tracking is on, and do a pass over the data
      model.train(True)
      avg_loss = train_one_epoch(model, loss_fn, optimizer, training_loader, device, epoch_number, tb_writer)

      running_tloss = 0.0
      # Set the model to evaluation mode, disabling dropout and using population
      # statistics for batch normalization.
      model.eval()

      # Disable gradient computation and reduce memory consumption.
      with torch.inference_mode():
          for i, tdata in enumerate(test_loader):
              xt, tlabel = tdata
              xt = xt.to(device)
              yt = model(xt)
              tloss = loss_fn(xt, tlabel, *yt)
              running_tloss += tloss

      avg_tloss = running_tloss / (i + 1)
      print('Training loss = {} | Test loss = {}'.format(avg_loss, avg_tloss))

      # Log the running loss averaged per batch
      # for both training and validatiomodel_dir
      if write_tb:
        tb_writer.add_scalars('Training vs. Test Loss',
                        { 'Training' : avg_loss, 'Test' : avg_tloss },
                        epoch_number + 1)
        tb_writer.flush()

      # Track best performance, and save the model's state
      if save_model and avg_tloss < best_tloss:
          best_tloss = avg_tloss
          model_dir = "./models"
          if not os.path.exists(model_dir):
            os.makedirs(model_dir)
          model_path = model_dir + '/model_{}_{}'.format(timestamp, epoch_number)
          torch.save(model.state_dict(), model_path)

      epoch_number += 1