import os

import torch.utils.data

import dflows as nf
import dflows.transforms as nft
from config import Config
from loops import test_model, train_model

if __name__ == '__main__':

  # set up config
  config = Config("MNIST")
  print(f"Using device: {config.device}")

  # data loaders
  data_loader_config = {
    'batch_size_train': config.batch_size,
    'batch_size_test': config.batch_size,
    'data_root': config.data_root,
    'ood_mode': config.ood_mode,
  }
  train_loader, test_loader = config.data_loader(**data_loader_config)

  # create flow (pending)
  base_dist = torch.distributions.MultivariateNormal(torch.zeros(config.manifold_dims), torch.eye(config.manifold_dims))

  ambient_model = nf.SquareNormalizingFlow(transforms=[
    nft.AffineCoupling(nft.CouplingNetwork(**config.coupling_network)),
    nft.Conv1x1(**config.conv1x1),
    nft.AffineCoupling(nft.CouplingNetwork(**config.coupling_network)),
    nft.Conv1x1(**config.conv1x1),
    nft.AffineCoupling(nft.CouplingNetwork(**config.coupling_network)),
    nft.Conv1x1(**config.conv1x1),
    nft.AffineCoupling(nft.CouplingNetwork(**config.coupling_network)),
    nft.Conv1x1(**config.conv1x1),
  ])

  # set up optimizer
  optim_config = {
    'params': ambient_model.parameters(),
    'lr': config.learning_rate
  }
  optim = torch.optim.Adam(**optim_config)

  # list to store train / test losses
  train_losses = []
  test_losses = []

  # load saved model and optimizer, if present
  if config.load_saved_params:
    model_state_path = f"{config.model_path}/model.pth"
    if os.path.exists(model_state_path):
      ambient_model.load_state_dict(torch.load(model_state_path, map_location=config.device))
      print('Loaded saved model state from:', model_state_path)
    optim_state_path = f"{config.model_path}/optim.pth"
    if os.path.exists(optim_state_path):
      optim.load_state_dict(torch.load(optim_state_path, map_location=config.device))
      print('Loaded saved optim state from:', optim_state_path)

  # run train / test loops
  print('\nStarted Train/Test')
  test_model(nn=ambient_model, epoch=0, loader=test_loader, config=config, test_losses=test_losses)
  for current_epoch in range(1, config.n_epochs + 1):
    train_model(nn=ambient_model, epoch=current_epoch, loader=train_loader, config=config, optim=optim, train_losses=train_losses)
    test_model(nn=ambient_model, epoch=current_epoch, loader=test_loader, config=config, test_losses=test_losses)
