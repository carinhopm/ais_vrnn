from torch.distributions import Distribution
from torch.distributions import Bernoulli
from torch import nn, Tensor
import torch

class ReparameterizedDiagonalGaussian(Distribution):
    """
    A distribution `N(y | mu, sigma I)` compatible with the reparameterization trick given `epsilon ~ N(0, 1)`.
    """

    def __init__(self, mu: Tensor, sigma: Tensor):
        assert mu.shape == sigma.shape, f"Tensors `mu` : {mu.shape} and ` log_sigma` : {sigma.shape} must be of the same shape"
        self.mu = mu
        self.sigma = sigma

    def sample_epsilon(self) -> Tensor:
        """`\eps ~ N(0, I)`"""
        return torch.empty_like(self.mu).normal_()

    def sample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (without gradients)"""
        with torch.no_grad():
            return self.rsample()

    def rsample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (with the reparameterization trick) """
        return self.mu + self.sigma * self.sample_epsilon()

    def log_prob(self, z: Tensor) -> Tensor:
        """return the log probability: log `p(z)`"""
        dist = torch.distributions.Normal(self.mu, self.sigma)
        return dist.log_prob(z)

    
class VRNN(nn.Module):

    def __init__(self, input_shape, latent_shape, generative_bias,device):
        super(VRNN, self).__init__()
        self.input_shape = input_shape
        self.latent_shape = latent_shape
        self.device = device
        self.generative_bias = generative_bias.to(self.device)

        self.phi_x = nn.Sequential(nn.Linear(self.input_shape, self.latent_shape),
                                   nn.ReLU(),
                                   nn.Linear(self.latent_shape, self.latent_shape))

        self.phi_z = nn.Sequential(nn.Linear(self.latent_shape, self.latent_shape),
                                   nn.ReLU(),
                                   nn.Linear(self.latent_shape, self.latent_shape))

        self.prior = nn.Sequential(nn.Linear(self.latent_shape, self.latent_shape),
                                   nn.ReLU(),
                                   nn.Linear(self.latent_shape, 2*self.latent_shape))

        self.encoder = nn.Sequential(nn.Linear(2*self.latent_shape, self.latent_shape),
                                     nn.ReLU(),
                                     nn.Linear(self.latent_shape, 2*self.latent_shape))

        self.decoder = nn.Sequential(nn.Linear(2*self.latent_shape, self.latent_shape),
                                     nn.ReLU(),
                                     nn.Linear(self.latent_shape, self.input_shape))

        self.rnn = nn.LSTM(input_size = 2*self.latent_shape, hidden_size = self.latent_shape)
        torch.nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        torch.nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
        torch.nn.init.zeros_(self.rnn.bias_ih_l0)
        torch.nn.init.zeros_(self.rnn.bias_hh_l0)

    def _prior(self, h, sigma_min=0.0, raw_sigma_bias=0.5):
        out = self.prior(h)
        mu, sigma = out.chunk(2, dim=-1) #mu and sigma batch X latent
        
        sigma_min = torch.full_like(sigma, sigma_min)
        sigma = torch.maximum(torch.nn.functional.softplus(sigma + raw_sigma_bias), sigma_min)

        return ReparameterizedDiagonalGaussian(mu, sigma)

    def posterior(self, hidden, x, prior_mu,  sigma_min=0.0, raw_sigma_bias=0.5):
        encoder_input = torch.cat([hidden, x], dim=1)
        hidden = self.encoder(encoder_input)
        mu, sigma = hidden.chunk(2, dim=-1)
        
        sigma_min = torch.full_like(sigma, sigma_min)
        sigma = torch.maximum(torch.nn.functional.softplus(sigma + raw_sigma_bias), sigma_min)

        mu = mu + prior_mu

        return ReparameterizedDiagonalGaussian(mu, sigma)

    def generative(self, z_enc, h):
        px_logits = self.decoder(torch.cat([z_enc, h], dim=1))
        px_logits = px_logits + self.generative_bias
        return Bernoulli(logits=px_logits)

    def forward(self, inputs, targets, logits=None):
        
        batch_size, seq_len, datadim = inputs.shape
        
        inputs = inputs.permute(1,0,2) #seq_len X batchsize X data_dim
        targets = targets.permute(1,0,2) #seq_len X batchsize X data_dim
        
        hs = torch.zeros(seq_len, batch_size, self.latent_shape, device = self.device) # Init LSTM hidden state output. seq_len X batch X latent
        out = torch.zeros(batch_size, self.latent_shape, device = self.device) # Init LSTM hidden state
        log_px = []
        log_pz = []
        log_qz = []
        for t in range(seq_len):
            x = inputs[t, :, :] #x_hat is batch X input
            y = targets[t, :, :]

            #Embed input
            x_hat = self.phi_x(x) #x_hat is batch X latent

            #Create prior distribution
            pz = self._prior(out) #out is batch X latent

            #Create approximate posterior
            qz = self.posterior(out, x_hat, prior_mu=pz.mu)

            #Sample and embed z from posterior
            z = qz.rsample()
            z_hat = self.phi_z(z) #z_hat is batch X latent

            #Decode z_hat
            px = self.generative(z_hat, out)

            #Update h from LSTM
            rnn_input = torch.cat([x_hat, z_hat], dim=1)
            rnn_input = rnn_input.unsqueeze(0) #rnn_input is 1 X batch X 2*latent
            out, _ = self.rnn(rnn_input) #out is 1 X batch X latent
            out = out.squeeze(axis=0)  #out is batch X latent
            
            hs[t,:] = out
            
            #loss
            log_px.append(px.log_prob(y).sum(dim=1)) #Sum over dimension
            log_pz.append(pz.log_prob(z).sum(dim=1)) #Sum over dimension
            log_qz.append(qz.log_prob(z).sum(dim=1)) #Sum over dimension
            
            if logits is not None:
                logits[t, :, :] = px.logits
        
        return log_px, log_pz, log_qz, logits, hs
