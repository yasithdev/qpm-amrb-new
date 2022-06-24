import os

import torch.nn.functional
import torch.optim
import torch.utils.data
from matplotlib import pyplot as plt
from tqdm import tqdm

import normalizing_flows as nf
from config import *
from datasets import load_mnist
from normalizing_flows.transforms.flow_transform import FlowTransform
from normalizing_flows.util import proj, pad
from util import gen_patches_from_img, gen_img_from_patches, set_requires_grad, get_best_device


def train_encoder(nn: FlowTransform,
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
    iterable = tqdm(loader, desc=f'[TRN] Epoch {epoch}', **tqdm_args)
    x: torch.Tensor
    y: torch.Tensor
    for batch_idx, (x, y) in enumerate(iterable):
      # reshape
      x = gen_patches_from_img(x.to(device), patch_h=patch_H, patch_w=patch_W)

      # forward pass
      z, fwd_log_det = nn.forward(x)
      zu = proj(z, manifold_dims=manifold_dims)
      z_pad = pad(zu, off_manifold_dims=(total_dims - manifold_dims), chw=(model_C, model_H, model_W))
      x_r, inv_log_det = nn.inverse(z_pad)

      # calculate loss
      minibatch_loss = torch.nn.functional.mse_loss(x_r, x)

      # backward pass
      optim.zero_grad()
      minibatch_loss.backward()
      optim.step()

      # accumulate sum loss
      sum_loss += minibatch_loss.item() * batch_size

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


def test_encoder(nn: FlowTransform,
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
    iterable = tqdm(loader, desc=f'[TST] Epoch {epoch}', **tqdm_args)
    x: torch.Tensor
    y: torch.Tensor
    for x, y in iterable:
      # reshape
      x = gen_patches_from_img(x.to(device), patch_h=patch_H, patch_w=patch_W)

      # forward pass
      z, fwd_log_det = nn.forward(x)
      zu = proj(z, manifold_dims=manifold_dims)
      z_pad = pad(zu, off_manifold_dims=(total_dims - manifold_dims), chw=(model_C, model_H, model_W))
      x_r, inv_log_det = nn.inverse(z_pad)

      # calculate loss
      minibatch_loss = torch.nn.functional.mse_loss(x_r, x)

      # accumulate sum loss
      sum_loss += minibatch_loss.item() * batch_size

      # logging
      stats = {'Loss(mb)': f'{minibatch_loss.item():.4f}'}
      iterable.set_postfix(stats)

      # accumulate plots
      if current_plot < img_count:
        ax[current_plot, 0].imshow(gen_img_from_patches(x, patch_h=patch_H, patch_w=patch_W)[0, 0])
        ax[current_plot, 1].imshow(gen_img_from_patches(x_r, patch_h=patch_H, patch_w=patch_W)[0, 0])
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
  train_loader, test_loader = load_mnist(batch_size_train=batch_size, batch_size_test=batch_size)

  # create flow (pending)
  base_dist = torch.distributions.MultivariateNormal(torch.zeros(manifold_dims), torch.eye(manifold_dims))

  # set up model and optimizer
  model = nf.SquareNormalizingFlow(transforms=[
    nf.AffineCoupling(nf.CouplingNetwork(in_channels=model_C // 2, hidden_channels=model_C // 2, num_hidden=1, out_channels=model_C // 2)),
    nf.Conv1x1(num_channels=model_C),
    nf.AffineCoupling(nf.CouplingNetwork(in_channels=model_C // 2, hidden_channels=model_C // 2, num_hidden=1, out_channels=model_C // 2)),
    nf.Conv1x1(num_channels=model_C),
    nf.AffineCoupling(nf.CouplingNetwork(in_channels=model_C // 2, hidden_channels=model_C // 2, num_hidden=1, out_channels=model_C // 2)),
    nf.Conv1x1(num_channels=model_C),
    nf.AffineCoupling(nf.CouplingNetwork(in_channels=model_C // 2, hidden_channels=model_C // 2, num_hidden=1, out_channels=model_C // 2)),
    nf.Conv1x1(num_channels=model_C),
    nf.AffineCoupling(nf.CouplingNetwork(in_channels=model_C // 2, hidden_channels=model_C // 2, num_hidden=1, out_channels=model_C // 2)),
    nf.Conv1x1(num_channels=model_C),
    nf.AffineCoupling(nf.CouplingNetwork(in_channels=model_C // 2, hidden_channels=model_C // 2, num_hidden=1, out_channels=model_C // 2)),
    nf.Conv1x1(num_channels=model_C)
  ])
  optim = torch.optim.Adam(
    params=model.parameters(),
    lr=learning_rate
  )

  # list to store train / test losses
  train_losses = []
  test_losses = []

  # load saved model and optimizer, if present
  model_state_path = f"{saved_models_path}/model.pth"
  if os.path.exists(model_state_path):
    model.load_state_dict(torch.load(model_state_path))
  optim_state_path = f"{saved_models_path}/optim.pth"
  if os.path.exists(optim_state_path):
    optim.load_state_dict(torch.load(optim_state_path))

  # run train / test loops
  is_emulation = True
  test_encoder(model, 0, test_loader, vis_path, dry_run=is_emulation)
  for current_epoch in range(1, n_epochs + 1):
    train_encoder(model, current_epoch, train_loader, saved_models_path, dry_run=is_emulation)
    test_encoder(model, current_epoch, test_loader, vis_path, dry_run=is_emulation)
