import sys

import torch.backends.cuda
# import torch.backends.mps
import torch.nn.functional
import torch.optim as optim
import torch.utils.data
from einops import rearrange
from matplotlib import pyplot as plt
from tqdm import tqdm

import normalizing_flows as nf
from datasets import load_mnist
from normalizing_flows.transforms.flow_transform import FlowTransform
from normalizing_flows.util import set_requires_grad

tqdm_args = {'bar_format': '{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}', 'file': sys.stdout}


def proj(
  z: torch.Tensor
) -> torch.Tensor:
  z = rearrange(z, 'b c h w -> b (c h w)')
  zu, zv = z[:, :manifold_dims], z[:, manifold_dims:]
  return zu


def pad(
  zu: torch.Tensor
) -> torch.Tensor:
  z_pad = torch.constant_pad_nd(zu, (0, total_dims - manifold_dims))
  z = rearrange(z_pad, 'b (c h w) -> b c h w', c=model_C, h=model_H, w=model_W)
  return z


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
    # reshape
    x = patched_img(x.to(device))

    # forward pass
    z, fwd_log_det = nn.forward(x)
    zu = proj(z)
    z_pad = pad(zu)
    x_r, inv_log_det = nn.inverse(z_pad)

    # calculate loss
    minibatch_loss = torch.nn.functional.mse_loss(x_r, x)

    # backward pass
    optimizer.zero_grad()
    minibatch_loss.backward()
    optimizer.step()

    # accumulate sum loss
    sum_loss += minibatch_loss.item() * batch_size

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
      # reshape
      x = patched_img(x.to(device))

      # forward pass
      z, fwd_log_det = nn.forward(x)
      zu = proj(z)
      z_pad = pad(zu)
      x_r, inv_log_det = nn.inverse(z_pad)

      # calculate loss
      minibatch_loss = torch.nn.functional.mse_loss(x_r, x)

      # accumulate sum loss
      sum_loss += minibatch_loss.item() * batch_size

      # plot first item of img_count batches
      if current_plot < img_count:
        ax[current_plot, 0].imshow(unpatch_img(x)[0, 0])
        ax[current_plot, 1].imshow(unpatch_img(x_r)[0, 0])
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


def patched_img(x: torch.Tensor) -> torch.Tensor:
  patched = rearrange(x, 'b c (h pH) (w pW) -> b (pH pW c) h w', pH=patch_H, pW=patch_W)
  return patched


def unpatch_img(x: torch.Tensor) -> torch.Tensor:
  unpatched = rearrange(x, 'b (pH pW c) h w -> b c (h pH) (w pW)', pH=patch_H, pW=patch_W)
  return unpatched


if __name__ == '__main__':

  # data config
  img_C, img_H, img_W = (1, 28, 28)
  total_dims = img_C * img_H * img_W

  # model config
  patch_H, patch_W = (4, 4)
  model_C, model_H, model_W = (img_C * patch_H * patch_W), (img_H // patch_H), (img_W // patch_W)
  manifold_dims = 1 * model_H * model_W

  # training config
  batch_size = 32
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
  train_loader, test_loader = load_mnist(batch_size_train=batch_size, batch_size_test=batch_size)

  # create flow (pending)
  base_dist = torch.distributions.MultivariateNormal(torch.zeros(manifold_dims), torch.eye(manifold_dims))

  mdl_C_half = model_C // 2
  model = nf.SquareNormalizingFlow([
    nf.AffineCouplingTransform(
      nf.CouplingNetwork(in_channels=mdl_C_half, hidden_channels=mdl_C_half, num_hidden=1, out_channels=mdl_C_half),
      nf.CouplingNetwork(in_channels=mdl_C_half, hidden_channels=mdl_C_half, num_hidden=1, out_channels=mdl_C_half)
    ),
    nf.AffineCouplingTransform(
      nf.CouplingNetwork(in_channels=mdl_C_half, hidden_channels=mdl_C_half, num_hidden=1, out_channels=mdl_C_half),
      nf.CouplingNetwork(in_channels=mdl_C_half, hidden_channels=mdl_C_half, num_hidden=1, out_channels=mdl_C_half)
    ),
    nf.AffineCouplingTransform(
      nf.CouplingNetwork(in_channels=mdl_C_half, hidden_channels=mdl_C_half, num_hidden=1, out_channels=mdl_C_half),
      nf.CouplingNetwork(in_channels=mdl_C_half, hidden_channels=mdl_C_half, num_hidden=1, out_channels=mdl_C_half)
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
