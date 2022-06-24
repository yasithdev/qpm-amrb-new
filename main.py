import os
import sys

import torch.backends.cuda
# import torch.backends.mps
import torch.nn.functional
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
from matplotlib import pyplot as plt

import normalizing_flows as nf
from datasets import load_mnist
from normalizing_flows.transforms.flow_transform import FlowTransform
from normalizing_flows.util import set_requires_grad

tqdm_args = {'bar_format': '{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}', 'file': sys.stdout}


def proj(
  z: torch.Tensor
) -> torch.Tensor:
  (zu, zv) = torch.split(z, (K, D - K), dim=-1)
  return zu


def pad(
  zu: torch.Tensor
) -> torch.Tensor:
  z_pad = torch.constant_pad_nd(zu, (0, D - K))
  return z_pad


def train(nn: FlowTransform,
          epoch: int,
          loader: torch.utils.data.DataLoader,
          out_path: str
          ) -> None:
  nn = nn.to(device)
  nn.train()
  set_requires_grad(model, True)

  # noinspection PyTypeChecker
  size = len(loader.dataset)

  sum_loss = 0

  # training
  iterable = tqdm(loader, desc=f'[TRN] Epoch {epoch}', **tqdm_args)
  x: torch.Tensor
  y: torch.Tensor
  for batch_idx, (x, y) in enumerate(iterable):
    # flatten
    x = torch.reshape(x, (x.size(0), -1)).to(device)

    # forward pass
    z, fwd_log_det = nn.to(device).forward(x)
    zu = proj(z)
    z_pad = pad(zu)
    x_r, inv_log_det = nn.to(device).inverse(z_pad)

    # calculate loss
    minibatch_loss = torch.nn.functional.mse_loss(x_r, x)

    # accumulate sum loss
    sum_loss += torch.nn.functional.mse_loss(x_r, x, reduction='sum').item()

    # backward pass
    optimizer.zero_grad()
    minibatch_loss.backward()
    optimizer.step()

    # logging
    stats = {'Loss(mb)': f'{minibatch_loss.item():.4f}'}
    iterable.set_postfix(stats)
    train_losses.append(minibatch_loss.item())

  # post-training
  torch.save(nn.state_dict(), f'{out_path}/model.pth')
  torch.save(optimizer.state_dict(), f'{out_path}/optimizer.pth')
  avg_loss = sum_loss / size
  tqdm.write(f'[TRN] Epoch {epoch}: Loss(avg): {avg_loss:.4f}')


def test(nn: FlowTransform,
         epoch: int,
         loader: torch.utils.data.DataLoader
         ) -> None:
  nn = nn.to(device)
  nn.eval()
  set_requires_grad(model, False)

  # noinspection PyTypeChecker
  size = len(loader.dataset)

  sum_loss = 0
  correct = 0

  img_count = 5
  fig, ax = plt.subplots(img_count, 2)
  current_plot = 0

  with torch.no_grad():
    iterable = tqdm(loader, desc=f'[TST] Epoch {epoch}', **tqdm_args)
    x: torch.Tensor
    y: torch.Tensor
    for x, y in iterable:
      # flatten
      x = torch.reshape(x, (x.size(0), -1)).to(device)

      # forward pass
      z, fwd_log_det = nn.forward(x)
      zu = proj(z)
      z_pad = pad(zu)
      x_r, inv_log_det = nn.inverse(z_pad)

      # accumulate sum loss
      sum_loss += torch.nn.functional.mse_loss(x_r, x, reduction='sum').item()
      if current_plot < img_count:
        ax[current_plot, 0].imshow(x[0].reshape(W, H, -1))
        ax[current_plot, 1].imshow(x_r[0].reshape(W, H, -1))
        current_plot += 1
      # pred = output.data.max(1, keepdim=True)[1]
      # correct += pred.eq(target.data.view_as(pred)).sum()

  # post-testing
  avg_loss = sum_loss / size
  avg_acc = 100. * correct / size
  test_losses.append(avg_loss)
  tqdm.write(f'[TST] Epoch {epoch}: Loss(avg): {avg_loss:.4f}, Accuracy: {correct}/{size} ({avg_acc:.0f}%)')

  plt.savefig(f"results/epoch_{epoch}.png")
  plt.close()


if __name__ == '__main__':

  # model config
  W = 28
  H = 28
  D = W * H
  K = 10
  B = 32

  # training config
  learning_rate = 1e-2
  momentum = 0.2
  n_epochs = 100

  if torch.cuda.is_available():
    device = "cuda"
  # elif torch.backends.mps.is_available():
  #   device = "mps"
  else:
    device = "cpu"
  print(f"Using device: {device}")

  # data loaders
  train_loader, test_loader = load_mnist(B, B)

  # create flow (pending)
  base_dist = torch.distributions.MultivariateNormal(torch.zeros(D), torch.eye(D))
  model = nf.SquareNormalizingFlow([
    nf.AffineCouplingTransform(
      nf.CouplingNetwork(in_dim=D // 2, hidden_dim=D, num_hidden=2, out_dim=D // 2),
      nf.CouplingNetwork(in_dim=D // 2, hidden_dim=D, num_hidden=2, out_dim=D // 2)
    ),
    nf.AffineCouplingTransform(
      nf.CouplingNetwork(in_dim=D // 2, hidden_dim=D, num_hidden=2, out_dim=D // 2),
      nf.CouplingNetwork(in_dim=D // 2, hidden_dim=D, num_hidden=2, out_dim=D // 2)
    ),
    nf.AffineCouplingTransform(
      nf.CouplingNetwork(in_dim=D // 2, hidden_dim=D, num_hidden=2, out_dim=D // 2),
      nf.CouplingNetwork(in_dim=D // 2, hidden_dim=D, num_hidden=2, out_dim=D // 2)
    )
  ])
  optimizer = optim.SGD(
    params=model.parameters(),
    lr=learning_rate,
    momentum=momentum
  )

  train_losses = []
  test_losses = []

  test(model, 0, test_loader)

  for current_epoch in range(1, n_epochs + 1):
    print()
    train(model, current_epoch, train_loader, 'results')
    print()
    test(model, current_epoch, test_loader)
