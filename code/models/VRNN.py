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

        ## Feature extractor. It extracts necessary input data
        ## Input : sequence of four 
        self.phi_x = nn.Sequential(nn.Linear(self.input_shape, self.latent_shape),
                                   nn.ReLU(),
                                   nn.Linear(self.latent_shape, self.latent_shape))

        self.phi_z = nn.Sequential(nn.Linear(self.latent_shape, self.latent_shape),
                                   nn.ReLU(),
                                   nn.Linear(self.latent_shape, self.latent_shape))

        ## Input would be recurrentShape and output would be 2 times stochastic shape
        self.prior = nn.Sequential(nn.Linear(self.latent_shape, self.latent_shape),
                                   nn.ReLU(),
                                   nn.Linear(self.latent_shape, 2*self.latent_shape))

        ##Input would be sum of stochastic and recurrent Shape
        self.encoder = nn.Sequential(nn.Linear(2*self.latent_shape, self.latent_shape),
                                     nn.ReLU(),
                                     nn.Linear(self.latent_shape, 2*self.latent_shape))

        ####Input would be sum of stochastic and recurrent Shape
        self.decoder = nn.Sequential(nn.Linear(2*self.latent_shape, self.latent_shape),
                                     nn.ReLU(),
                                     nn.Linear(self.latent_shape, self.input_shape))

        self.dropoutAfterRNN = nn.Dropout(0.1)
        ##Input would be 2 times stochastic and hiddenSize should be recurrence hidden space. 
        self.rnn = nn.LSTM(input_size = 2*self.latent_shape, hidden_size = self.latent_shape)
        torch.nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        torch.nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
        torch.nn.init.zeros_(self.rnn.bias_ih_l0)
        torch.nn.init.zeros_(self.rnn.bias_hh_l0)

    def _prior(self, h, sigma_min=0.0, raw_sigma_bias=0.5):
        out = self.prior(h)
        mu, sigma = out.chunk(2, dim=-1) #mu and sigma batch X latent
        
        sigma_min = torch.full_like(sigma, sigma_min)
        sigma = torch.max(torch.nn.functional.softplus(sigma + raw_sigma_bias), sigma_min)

        return ReparameterizedDiagonalGaussian(mu, sigma)

    def posterior(self, hidden, x, prior_mu,  sigma_min=0.0, raw_sigma_bias=0.5):
        ##Concatenates last hidden state and x (So if A and B are of shape (3, 4), torch.cat([A, B], dim=0) will be of shape (6, 4) )
        encoder_input = torch.cat([hidden, x], dim=1)
        hidden = self.encoder(encoder_input)
        mu, sigma = hidden.chunk(2, dim=-1)
        
        sigma_min = torch.full_like(sigma, sigma_min)
        sigma = torch.max(torch.nn.functional.softplus(sigma + raw_sigma_bias), sigma_min)

        mu = mu + prior_mu

        return ReparameterizedDiagonalGaussian(mu, sigma)

    ## This should return a guassion instead if we are using continous input. 
    def generative(self, z_enc, h):
        px_logits = self.decoder(torch.cat([z_enc, h], dim=1))
        px_logits = px_logits + self.generative_bias
        return Bernoulli(logits=px_logits)

    def forward(self, inputs, targets, labels, logits=None):
    #def forward(self, inputs, targets, logits=None):
        
        batch_size, seq_len, datadim = inputs.shape
        
        inputs = inputs.permute(1,0,2) #seq_len X batchsize X data_dim
        targets = targets.permute(1,0,2) #seq_len X batchsize X data_dim
        
        ## Hidden state at each time step
        hs = torch.zeros(seq_len, batch_size, self.latent_shape, device = self.device) # Init LSTM hidden state output. seq_len X batch X latent

        ## Last hidden state from LSTM
        out = torch.zeros(batch_size, self.latent_shape, device = self.device) # Init LSTM hidden state
        log_px = []
        log_pz = []
        log_qz = []

        h = torch.zeros(1,batch_size, self.latent_shape, device = self.device) # Init LSTM hidden state
        c = torch.zeros(1,batch_size, self.latent_shape, device = self.device) # Init LSTM cell
        
        ## Initializing z_mus to store the latent mean vectors 
        z_mus = torch.zeros(seq_len, len(labels), self.latent_shape, device = self.device)
        
        logits = torch.zeros(seq_len, len(labels), datadim , device = self.device)
        
        
        #print('z_mus len {}'.format(z_mus.shape))

        #print('seq_len {}'.format(seq_len))
        
        ## seq_len is the time length
        for t in range(seq_len):
            
            x = inputs[t, :, :] #x_hat is batch X input
            y = targets[t, :, :]

            #Embed input
            x_hat = self.phi_x(x) #x_hat is batch X latent
            
            #Create prior distribution
            ## Prior distribution knows everything that happenned in the past but has no 
            ## knowledge about present
            pz = self._prior(out) #out is batch X latent

            #Create approximate posterior
            ## out is everything in the past
            ## x_hat is everything in the present
            ## We try to create a distribution that can explain both the past using prior and present 
            ## using the x_hat. But these distribution can be different.
            ## posterior is the diagonal guassion. This dist is used to restrict the freedom. 
            ## 
            qz = self.posterior(out, x_hat, prior_mu=pz.mu)
            
            #Sample and embed z from posterior
            z = qz.rsample()
            z_hat = self.phi_z(z) #z_hat is batch X latent

            #Decode z_hat
            ## px is the prob distribution that should be able to reconstruct the input.
            ## px is the multivariate burnouilli distribution.
            px = self.generative(z_hat, out)

            #Update h from LSTM
            rnn_input = torch.cat([x_hat, z_hat], dim=1)
            rnn_input = rnn_input.unsqueeze(0) #rnn_input is 1 X batch X 2*latent
            #out, _ = self.rnn(rnn_input) #out is 1 X batch X latent
            out, (h, c) = self.rnn(rnn_input, (h, c)) #out is 1 X batch X latent
            
            out = self.dropoutAfterRNN(out)
                        
            out = out.squeeze(axis=0)  #out is batch X latent
            
            hs[t,:] = out
            
            ## qz.mu is of dimension batchSize * latentSize
            #print('qz mean: {}'.format(qz.mu))
            
            z_mus[t, :, :] = qz.mu
            
            #loss
            ## log_px is the reconstruction loss that is reconstructed using burnoulli distribution. 
            log_px.append(px.log_prob(y).sum(dim=1)) #Sum over dimension
            ## log_pz is the probably og sampling that we drew from the prior distribution. 
            log_pz.append(pz.log_prob(z).sum(dim=1)) #Sum over dimension
            ## log_qz is the probability of sampling if we had drawn of posterior.  
            log_qz.append(qz.log_prob(z).sum(dim=1)) #Sum over dimension
            
            ## qz - pz is the KL divergence
            ##Dimension of px.logits: seqLength * BatchSize * InputShape
            if logits is not None:
                logits[t, :, :] = px.logits
                
        return log_px, log_pz, log_qz, logits, hs, z_mus
