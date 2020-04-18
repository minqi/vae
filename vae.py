import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.distributions import Normal, MultivariateNormal
from torchvision import datasets, transforms
from torchvision.utils import save_image


class ReshapeTransform:
	def __init__(self, new_size):
		self.new_size = new_size

	def __call__(self, img):
		return torch.reshape(img, self.new_size).flatten()


class MLPEncoder(nn.Module):
	def __init__(self, input_size, output_size, hidden_size=128):
		super().__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc_mu = nn.Linear(hidden_size, output_size)
		self.fc_logvar = nn.Linear(hidden_size, output_size)

	def forward(self, x):
		h = torch.tanh(self.fc1(x))
		return self.fc_mu(h), self.fc_logvar(h)


class MLPDecoder(nn.Module):
	def __init__(self, input_size, output_size, hidden_size=128, decoder_type='bernoulli'):
		super().__init__()

		self.decoder_type = decoder_type

		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc21 = nn.Linear(hidden_size, output_size)
		self.fc22 = nn.Linear(hidden_size, output_size)

	def forward(self, z):
		h = torch.tanh(self.fc1(z))

		if self.decoder_type == 'bernoulli':
			p = torch.sigmoid(self.fc21(h))
			return p
		elif self.decoder_type == 'gaussian':
			mu = self.fc21(h)
			logvar = torch.diag_embed(self.fc22(h).exp())
			dist = MultivariateNormal(mu, logvar)
			return dist.sample()
		else:
			raise ValueError(f'Unsupported decoder type, {self.decoder_type}')


class ConvEncoder(nn.Module):
	def __init__(self, hidden_size):
		pass

	def forward(self, x):
		pass


class ConvDecoder(nn.Module):
	def __init__(self, hidden_size):
		pass

	def forward(self, z):
		pass


class VAE(nn.Module):
	def __init__(self, encoder, decoder):
		super().__init__()

		self.encoder = encoder
		self.decoder = decoder

	def encode(self, x):
		return self.encoder(x)

	def reparameterize(self, eps, mu, logvar):
		return eps*logvar + mu

	def decode(self, z):
		return self.decoder(z)

	def sample(self):
		z = torch.randn(self.decoder.hidden_size)
		return self.decode(z)

	def forward(self, x):
		mu, logvar = self.encoder(x)
		eps = torch.randn(mu.shape) # B x |z|
		z = self.reparameterize(eps, mu, logvar)
		x_pred = self.decode(z)

		return x_pred, mu, logvar


def ELBO(x, x_pred, mu, logvar):
	KL = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	BCE = F.binary_cross_entropy(x_pred, x, reduction='sum')

	return KL + BCE


def train(model, optimizer, train_loader):
	model.train()
	device = next(model.parameters()).device
	mini_batch_ratio = len(train_loader)
	for b, (data, _) in enumerate(train_loader):
		data = data.to(device)
		optimizer.zero_grad()
		x_pred, mu, logvar = model(data)
		elbo = len(train_loader)*ELBO(data.view(-1, 784), x_pred, mu, logvar)
		elbo.backward()
		optimizer.step()

		log_str = ' '.join([f"train batch:{b}\ttrain loss: {elbo.item()/train_loader.batch_size:4.4f}"])
		print(log_str)


def test(model, test_loader):
	model.eval()
	test_loss = 0
	device = next(model.parameters()).device
	mini_batch_ratio = len(train_loader)
	for b, (data, _) in enumerate(test_loader):
		data = data.to(device)
		x_pred, mu, logvar = model(data)
		test_loss += mini_batch_ratio*ELBO(data.view(-1, 784), x_pred, mu, logvar).item()
		log_str = ' '.join([f"test batch:{b}\ttest loss: {test_loss.item()/test_loader.batch_size:4.4f}"])


def run_trial(args, model, optimizer, n_epochs, train_loader, test_loader, save=False):
	for epoch in range(n_epochs):
		print(f'Beginning epoch {epoch}...')
		train(model, optimizer, train_loader)
		# test(model, test_loader)
		if save:
			with torch.no_grad():
				n = args.test_output_size
				if args.test_output == 'uniform':
					assert args.latent_size == 2, 'Sampling on uniform manifold only supported for latent_size=2.'
					filename = f'results/uniform_{n}x{n}_' + str(epoch) + '.png'
					save_uniform_manifold_2d(model, n, filename)
				elif args.test_output == 'random':
					filename = f'results/random_sample_{n}x{n}_' + str(epoch) + '.png'
					save_random_decodings(model, n, filename)


def save_random_decodings(model, n, filename):
	total = n**2
	data_iter = iter(train_loader)
	sample = data_iter.next()[0][:total]
	remaining = total - sample.shape[0]
	while remaining > 0:
		sample = torch.cat((sample, data_iter.next()[0][:remaining]), 0)
		remaining = total - sample.shape[0]
	sample, _, _ = model(sample)
	save_image(sample.view(total, 1, 28, 28).cpu(), filename, nrow=n)


def save_uniform_manifold_2d(model, n, filename):
	u_manifold = uniform_manifold_2d(n)
	sample = model.decode(u_manifold)
	save_image(sample.view(n**2, 1, 28, 28).cpu(), filename, nrow=n)


def uniform_manifold_2d(n):
	p_z = Normal(0, 1)
	coord = torch.linspace(0.1, 0.99, n)
	x, y = torch.meshgrid([coord, coord])
	z1 = p_z.icdf(x)
	z2 = p_z.icdf(y)
	manifold = torch.stack((z1, z2), -1)
	return manifold


def set_seed(seed):
	np.random.seed(seed)
	torch.manual_seed(seed)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='VAE test')
	parser.add_argument('-b', '--batch_size', type=int, default=128)
	parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
	parser.add_argument('-s', '--seed', type=int, default=88)
	parser.add_argument('--hidden_size', type=int, default=400)
	parser.add_argument('--latent_size', type=int, default=2)
	parser.add_argument('--n_epochs', type=int, default=10)
	parser.add_argument('--no_cuda', type=bool, default=False)
	parser.add_argument('--test_output', type=str, default='random')
	parser.add_argument('-tn', '--test_output_size', type=int, default=20)

	args = parser.parse_args()
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device('cuda' if args.cuda else 'cpu')
	dataloader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

	set_seed(args.seed)

	# Load MNIST digits
	train_loader = torch.utils.data.DataLoader(
		datasets.MNIST('data/', train=True, download=True,
		transform=transforms.Compose([
			transforms.ToTensor(),
			ReshapeTransform((-1, 784)),
		])),
		batch_size=args.batch_size,
		shuffle=True,
		**dataloader_kwargs)

	test_loader = torch.utils.data.DataLoader(
		datasets.MNIST('data/', train=True, download=True,
		transform=transforms.Compose([
			transforms.ToTensor(),
			ReshapeTransform((-1, 784)),
		])),
		batch_size=args.batch_size,
		shuffle=True,
		**dataloader_kwargs)

	x_shape = iter(train_loader).next()[0][0].shape
	input_size = torch.prod(torch.tensor(x_shape)).item()

	encoder = MLPEncoder(input_size=input_size, hidden_size=args.hidden_size, output_size=args.latent_size)
	decoder = MLPDecoder(input_size=args.latent_size, hidden_size=args.hidden_size, output_size=input_size, decoder_type='bernoulli')
	vae = VAE(encoder, decoder).to(device)

	optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

	run_trial(args, vae, optimizer, args.n_epochs, train_loader, test_loader, save=True)



