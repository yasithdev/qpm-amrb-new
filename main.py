import os

import torch.nn.functional
import torch.optim
import torch.utils.data
from matplotlib import pyplot as plt
from tqdm import tqdm

import config
import flows as nf
import flows.transforms as nft
from data_loaders import load_mnist
from flows.util import proj, pad
from util import gen_patches_from_img, gen_img_from_patches, set_requires_grad, get_best_device


def train_encoder(nn: nft.FlowTransform,
                  epoch: int,
                  loader: torch.utils.data.DataLoader,
                  save_path: str,
                  dry_run: bool = False,
                  ) -> None:
  # initialize loop
  nn = nn.to(device)
  nn.train()
  size = len(loader.dataset)
  sum_loss = 0

  # training
  set_requires_grad(nn, True)
  with torch.enable_grad():
    iterable = tqdm(loader, desc=f'[TRN] Epoch {epoch}', **config.tqdm_args)
    x: torch.Tensor
    y: torch.Tensor
    for batch_idx, (x, y) in enumerate(iterable):
      # reshape
      x = gen_patches_from_img(x.to(device), patch_h=config.patch_H, patch_w=config.patch_W)

      # forward pass
      z, fwd_log_det = nn.forward(x)
      zu = proj(z, manifold_dims=config.manifold_dims)
      z_pad = pad(zu, off_manifold_dims=(config.total_dims - config.manifold_dims), chw=(config.model_C, config.model_H, config.model_W))
      x_r, inv_log_det = nn.inverse(z_pad)

      # calculate loss
      minibatch_loss = torch.nn.functional.mse_loss(x_r, x)

      # backward pass
      optim.zero_grad()
      minibatch_loss.backward()
      optim.step()

      # accumulate sum loss
      sum_loss += minibatch_loss.item() * config.batch_size

      # logging
      stats = {'Loss(mb)': f'{minibatch_loss.item():.4f}'}
      iterable.set_postfix(stats)

  # post-training
  avg_loss = sum_loss / size
  tqdm.write(f'[TRN] Epoch {epoch}: Loss(avg): {avg_loss:.4f}')
  train_losses.append(avg_loss)

  # save model/optimizer states
  if not dry_run:
    torch.save(nn.state_dict(), f'{save_path}/model.pth')
    torch.save(optim.state_dict(), f'{save_path}/optim.pth')
  print()


def test_encoder(nn: nft.FlowTransform,
                 epoch: int,
                 loader: torch.utils.data.DataLoader,
                 vis_path: str,
                 dry_run: bool = False,
                 ) -> None:
  # initialize loop
  nn = nn.to(device)
  nn.eval()
  size = len(loader.dataset)
  sum_loss = 0

  # initialize plot(s)
  img_count = 5
  fig, ax = plt.subplots(img_count, 2)
  current_plot = 0

  # testing
  set_requires_grad(nn, False)
  with torch.no_grad():
    iterable = tqdm(loader, desc=f'[TST] Epoch {epoch}', **config.tqdm_args)
    x: torch.Tensor
    y: torch.Tensor
    for x, y in iterable:
      # reshape
      x = gen_patches_from_img(x.to(device), patch_h=config.patch_H, patch_w=config.patch_W)

      # forward pass
      z, fwd_log_det = nn.forward(x)
      zu = proj(z, manifold_dims=config.manifold_dims)
      z_pad = pad(zu, off_manifold_dims=(config.total_dims - config.manifold_dims), chw=(config.model_C, config.model_H, config.model_W))
      x_r, inv_log_det = nn.inverse(z_pad)

      # calculate loss
      minibatch_loss = torch.nn.functional.mse_loss(x_r, x)

      # accumulate sum loss
      sum_loss += minibatch_loss.item() * config.batch_size

      # logging
      stats = {'Loss(mb)': f'{minibatch_loss.item():.4f}'}
      iterable.set_postfix(stats)

      # accumulate plots
      if current_plot < img_count:
        ax[current_plot, 0].imshow(gen_img_from_patches(x, patch_h=config.patch_H, patch_w=config.patch_W)[0, 0])
        ax[current_plot, 1].imshow(gen_img_from_patches(x_r, patch_h=config.patch_H, patch_w=config.patch_W)[0, 0])
        current_plot += 1

  # post-testing
  avg_loss = sum_loss / size
  test_losses.append(avg_loss)
  tqdm.write(f'[TST] Epoch {epoch}: Loss(avg): {avg_loss:.4f}')

  # save generated plot
  if not dry_run:
    plt.savefig(f"{vis_path}/epoch_{epoch}.png")
  plt.close()
  print()


if __name__ == '__main__':

  # set up device
  device = get_best_device()
  print(f"Using device: {device}")

  # data loaders
  train_loader, test_loader = load_mnist(batch_size_train=config.batch_size, batch_size_test=config.batch_size, data_root=config.data_root)

  # create flow (pending)
  base_dist = torch.distributions.MultivariateNormal(torch.zeros(config.manifold_dims), torch.eye(config.manifold_dims))

  # set up model
  coupling_network_config = {
    'num_channels': config.model_C // 2,
    'num_hidden_channels': config.model_C,
    'num_hidden_layers': 1
  }
  conv1x1_config = {
    'num_channels': config.model_C
  }
  model = nf.SquareNormalizingFlow(transforms=[
    nft.AffineCoupling(nft.CouplingNetwork(**coupling_network_config)),
    nft.Conv1x1(**conv1x1_config),
    nft.AffineCoupling(nft.CouplingNetwork(**coupling_network_config)),
    nft.Conv1x1(**conv1x1_config),
    nft.AffineCoupling(nft.CouplingNetwork(**coupling_network_config)),
    nft.Conv1x1(**conv1x1_config),
  ])

  # set up optimizer
  optim_config = {
    'params': model.parameters(),
    'lr': config.learning_rate
  }
  optim = torch.optim.Adam(**optim_config)

  # list to store train / test losses
  train_losses = []
  test_losses = []

  # load saved model and optimizer, if present
  model_state_path = f"{config.saved_models_path}/model.pth"
  if os.path.exists(model_state_path):
    model.load_state_dict(torch.load(model_state_path))
    print('Loaded saved model state from:', model_state_path)
  optim_state_path = f"{config.saved_models_path}/optim.pth"
  if os.path.exists(optim_state_path):
    optim.load_state_dict(torch.load(optim_state_path))
    print('Loaded saved optim state from:', optim_state_path)

  # run train / test loops
  print('\nStarted Train/Test')
  test_encoder(model, 0, test_loader, config.vis_path, dry_run=config.dry_run)
  for current_epoch in range(1, config.n_epochs + 1):
    train_encoder(model, current_epoch, train_loader, config.saved_models_path, dry_run=config.dry_run)
    test_encoder(model, current_epoch, test_loader, config.vis_path, dry_run=config.dry_run)
