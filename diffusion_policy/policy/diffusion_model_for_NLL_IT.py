"""
Code for Diffusion model class
"""
import os
import math
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from diffusion_policy.common import logger


class DiffusionModel_IT(nn.Module):
    """Base class diffusion model for x, with optional conditional info, y.
       *logsnr integration: we do integrals in terms of logsnr instead of snr.
       Therefore we have to include a factor of 1/snr in integrands."""

    def __init__(self, DP_model, device):
        super().__init__()
        self.logs = {"val loss": [],
                     "train loss": []}  # store stuff for plotting, could use tensorboard for larger models
        self.loc_logsnr, self.scale_logsnr = 0., 2.  # initial location and scale for integration. Reset in data-driven way by dataset_info
        #self.dataset_info()
        
        self.clip = 4  # initial quantile for integration.
        self.device = device
        self.dtype, self.d, self.shape, self.left = None, None, None, None  # dtype, dimensionality, and shape for data, set when we see it in "fit"
        self.model = DP_model                         # A model that takes in model(x, y (optional), snr, is_simple) and outputs the noise estimate
        
    def update_model (self, new_model):
        self.model = new_model

    def set_noise_scheduler(self, noise_scheduler):
        self.noise_scheduler = noise_scheduler

    # def forward(self, batch, logsnr):
    #     """Batch is either [x,y] or [x,] depending on whether it is a conditional model.
    #     """
    #     return self.model(batch, torch.exp(logsnr))

    def noisy_channel(self, x, logsnr):
        '''
        The noise channel is working as the forward process of the DDPM
        here, we need to computet the corresponding diffusion timestep to the logsnr
        '''
        # compare each sigmoid(logsnr) (or sigmoid(alpha)) with alpha_cumprod of DDIM to find the closest value of alpha_cumprod to each sigmoid(logsnr)
        # Then find the correspoing t of logsnr
        diff = torch.abs(torch.sigmoid(logsnr).unsqueeze(1) - self.noise_scheduler.alphas_cumprod.unsqueeze(0).to(self.device))
        timestep = torch.argmin(diff, dim=1)   # shape (N,)

        eps = torch.randn((len(logsnr),) + self.shape, dtype=x.dtype, device=x.device)
        noisy_data = self.noise_scheduler.add_noise(x, eps, timestep)
        return noisy_data, timestep, eps

    def mse(self, batch, logsnr, mse_type='epsilon', xinterval=None):
        """Return MSE curve either conditioned or not on y.
        x_hat = z/sqrt(snr) - eps_hat(z, snr)/sqrt(snr),
        so x_hat - x = (eps - eps_hat(z, snr))/sqrt(snr).
        And we actually reparametrize eps_hat to depend on eps(z/sqrt(1+snr), snr)
        Options are:
        mse_type: {"epsilon" or "x"), for error in predicting noise, or predicting data.
        xinterval: Whether predictions should be clamped to some interval.
        delta: If provided, hard round to the nearest discrete value
        soft: If provided, soft round to the nearest discrete value
        """
        x = batch[0].to(self.device)  # assume iterator gives other things besides x in list (e.g. y)
        z, timesteps, eps = self.noisy_channel(x, logsnr)


        local_cond = batch[1]
        global_cond = batch[2]
        cond_data = batch[3]
        cond_mask = batch[4]

        z[cond_mask] = cond_data[cond_mask]

        eps_hat = self.model(z, timesteps, local_cond=local_cond, global_cond=global_cond)
        
        err_eps = (eps - eps_hat).flatten(start_dim=1) 
        mse_eps = torch.einsum('ij,ij->i', err_eps, err_eps)  # MSE for epsilon
        # if mse_type == 'epsilon':
        #     return mse_x * torch.exp(logsnr)  # Special form of x_hat, eps_hat leads to this relation
        # elif mse_type == 'x':
        #     return mse_x
        
        # x_hat = torch.sqrt(1 + torch.exp(-logsnr.view(self.left))) * z - eps_hat * torch.exp(-logsnr.view(self.left) / 2)
        # if delta:
        #     if soft:
        #         x_hat = soft_round(x_hat, torch.exp(logsnr).view(self.left), xinterval,
        #                            delta)  # soft round to nearest discrete value
        #     else:
        #         x_hat = delta * torch.round((x_hat - xinterval[0]) / delta) + xinterval[
        #             0]  # hard round to the nearest discrete value
        # if xinterval:
        #     x_hat = torch.clamp(x_hat, xinterval[0], xinterval[1])  # clamp predictions to not fall outside of range
        # err = (x - x_hat).flatten(start_dim=1)  # Flatten for, e.g., image data
        # mse_x = torch.einsum('ij,ij->i', err, err)  # MSE for epsilon
        # if mse_type == 'epsilon':
        #     return mse_x * torch.exp(logsnr)  # Special form of x_hat, eps_hat leads to this relation
        # elif mse_type == 'x':
        #     return mse_x
        return mse_eps

    def nll(self, batch, logsnr_samples_per_x=1, xinterval=None):
        """-log p(x) (or -log p(x|y) if y is provided) estimated for a batch."""
        nll = 0
        for _ in range(logsnr_samples_per_x):
            logsnr, w = self.logistic_integrate(len(batch[0]), *self.loc_scale, device=self.device)
            mses = self.mse(batch, logsnr, mse_type='epsilon', xinterval=xinterval)
            nll += self.loss(mses, logsnr, w) / logsnr_samples_per_x
        return nll

    def loss(self, mses, logsnr, w):
        """
        Returns the (per-sample) losses from MSEs, for convenience adding constants to match
        with NLL expression.
        :param mses:  Mean square error for *epsilon*
        :param logsnr:  Log signal to noise ratio
        :param w:  Integration weights
        :return: loss, -log p(x) estimate
        """
        mmse_gap = mses - self.mmse_g(
            logsnr)  # The "scale" does not change the integral, but may be more numerically stable.
        loss = self.h_g + 0.5 * (w * mmse_gap).mean()
        return loss  # *logsnr integration, see paper

    @torch.no_grad()
    def test_nll(self, dataloader, npoints=100, xinterval=None):
        """Calculate expected NLL on data at test time.  Main difference is that we can clamp the integration
        range, because negative values of MMSE gap we can switch to Gaussian decoder to get zero.
        npoints - number of points to use in integration
        delta - if the data is discrete, delta is the gap between discrete values.
        E.g. delta = 1/127.5 for CIFAR data (0, 255) scaled to -1, 1 range
        xinterval - a tuple of the range of the discrete values, e.g. (-1, 1) for CIFAR10 normalized
        soft -  using soft discretization if True.
        """
        if self.model.training:
            print("Warning - estimating test NLL but model is in train mode")
        results = {}  # Return multiple forms of results in a dictionary
        clip = self.clip
        loc, scale = self.loc_scale
        logsnr, w = logistic_integrate(npoints, loc=loc, scale=scale, clip=clip, device=self.device, deterministic=True)
        left_logsnr, right_logsnr = loc - clip * scale, loc + clip * scale

        # sort logsnrs along with weights
        logsnr, idx = logsnr.sort()
        w = w[idx].to('cpu')

        results['logsnr'] = logsnr.to('cpu')
        results['w'] = w
        mses = []  # Store all MSEs, per sample, logsnr, in an array
        total_samples = 0
        val_loss = 0
        for batch in tqdm(dataloader):

            nbatch = self.normalizer.normalize(batch)
            nobs = nbatch['obs']
            naction = nbatch['action']

            B, _, Do = nobs.shape
            To = self.n_obs_steps
            assert Do == self.obs_dim
            T = self.horizon
            Da = self.action_dim

            # build input
            device = self.device
            dtype = self.dtype

            # handle different ways of passing observation
            local_cond = None
            global_cond = None
            trajectory = naction

            if self.obs_as_local_cond:
                # condition through local feature
                # all zero except first To timesteps
                local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
                local_cond[:,:To] = nobs[:,:To]
                shape = (B, T, Da)
                cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
                cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

            elif self.obs_as_global_cond:
                # condition throught global feature
                global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
                shape = (B, T, Da)
                if self.pred_action_steps_only:
                    shape = (B, self.n_action_steps, Da)
                    start = To
                    if self.oa_step_convention:
                        start = To - 1
                    end = start + self.n_action_steps
                    trajectory = naction[:,start:end]
                cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
                cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            else:
                # condition through impainting
                shape = (B, T, Da+Do)
                cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
                cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
                cond_data[:,:To,Da:] = nobs[:,:To]
                cond_mask[:,:To,Da:] = True
                trajectory = torch.cat([naction, nobs], dim=-1)

            # construct a batch of data in the form that diffusion model of IT expects
            batch_data = [trajectory, local_cond, global_cond, cond_data, cond_mask]
            data = batch[0].to(self.device)  # assume iterator gives other things besides x in list
            n_samples = len(data)
            total_samples += n_samples

            val_loss += self.nll([data, ] + batch[1:], xinterval=xinterval).cpu() * n_samples

            mses.append(torch.zeros(n_samples, len(logsnr)))
            for j, this_logsnr in enumerate(logsnr):
                this_logsnr_broadcast = this_logsnr * torch.ones(len(data), device=self.device)

                # Regular MSE, clamps predictions, but does not discretize
                this_mse = self.mse([data, ] + batch[1:], this_logsnr_broadcast, mse_type='epsilon',
                                    xinterval=xinterval).cpu()
                mses[-1][:, j] = this_mse

        val_loss /= total_samples

        mses = torch.cat(mses, dim=0)  # Concatenate the batches together across axis 0
        results['mses-all'] = mses  # Store array of mses for each sample, logsnr
        mses = mses.mean(dim=0)  # Average across samples, giving MMSE(logsnr)

        results['mses'] = mses
        results['mmse_g'] = self.mmse_g(logsnr.to(self.device)).to('cpu')

        # here the results of the integral is clamped to 0 if the MMSE is negative
        results['nll (nats)'] = torch.mean(self.h_g - 0.5 * w * torch.clamp(results['mmse_g'] - mses, 0.))
        results['nll (bpd)'] = results['nll (nats)'] / math.log(2) / self.d
        
        ## Variance (of the mean) calculation - via CLT, it's the variance of the samples (over epsilon, x, logsnr) / n samples.
        ## n_samples is number of x samples * number of logsnr samples per x
        inds = (results['mmse_g'] - results[
            'mses']) > 0  # we only give nonzero estimates in this region (for continuous estimators)
        n_samples = results['mses-all'].numel()
        wp = w[inds]
        results['nll (nats) - var'] = torch.var(
            0.5 * wp * (results['mmse_g'][inds] - results['mses-all'][:, inds])) / n_samples
        
        results['nll (bpd) - std'] = torch.sqrt(results['nll (nats) - var']) / math.log(2) / self.d
        
        return results, val_loss

    @property
    def loc_scale(self):
        """Return the parameters defining a normal distribution over logsnr, inferred from data statistics."""
        return self.loc_logsnr, self.scale_logsnr

    def dataset_info(self, data, covariance_spectrum=None, diagonal=False):
        """covariance_spectrum can provide precomputed spectrum to speed up frequent experiments.
           diagonal: {False, True}  approximates covariance as diagonal, useful for very high-d data.
        """
        # logger.log("Getting dataset statistics, including eigenvalues,")
        # data = nbatch['action'].to("cpu")  # assume iterator gives other things besides x in list
        # logger.log('using # samples given:', len(data))
        
        self.d = len(data[0].flatten())
        if not diagonal:
            assert len(data) > self.d, f"Use a batch with more samples {len(data[0])} than dimensions {self.d}"
        self.shape = data[0].shape
        self.dtype = data[0].dtype
        self.left = (-1,) + (1,) * (len(self.shape))  # View for left multiplying a batch of samples
        x = data.flatten(start_dim=1)
        if covariance_spectrum:  # May save in cache to avoid processing for every experiment
            self.mu, self.U, self.log_eigs = covariance_spectrum
        else:
            var, self.mu = torch.var_mean(x, 0)
            x = x - self.mu
            if diagonal:
                self.log_eigs = torch.log(var)
                self.U = None  # "U" in this diagonal approximation should be identity, but would need to be handled specially to avoid storing large matrix.
            else:
                _, eigs, self.U = torch.linalg.svd(x, full_matrices=False)  # U.T diag(eigs^2/(n-1)) U = covariance
                self.log_eigs = 2 * torch.log(eigs) - math.log(len(x) - 1)  # Eigs of covariance are eigs**2/(n-1)  of SVD
            # t.save((self.mu, self.U, self.log_eigs), './covariance/cifar_covariance.pt')  # Save like this

        self.log_eigs = self.log_eigs.to(self.device)
        self.mu = self.mu.to(self.device)
        if self.U is not None:
            self.U = self.U.to(self.device)

        # Used to estimate good range for integration
        self.loc_logsnr = -self.log_eigs.mean().item()
        if diagonal:
            # A heuristic, since we won't get good variance estimate from diagonal - use loc/scale from CIFAR.
            self.loc_logsnr, self.scale_logsnr = 6.261363983154297, 3.0976245403289795
        else:
            self.scale_logsnr = torch.sqrt(1 + 3. / math.pi * self.log_eigs.var()).item()

    def fit(self, dataloader_train, epochs=10, use_optimizer='adam', lr=1e-4, verbose=False):
        """Given dataset, train the MMSE model for predicting the noise (or score).
           See image_datasets.py for example of torch dataset, that can be used with dataloader
           Shape needs to be compatible with model inputs.
        """
        if use_optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        # Return to standard fitting paradigm
        for i in range(1, epochs + 1):  # Main training loop
            print("training ... ")
            train_loss = 0.
            t0 = time.time()
            total_samples = 0
            self.train()
            for batch in tqdm(dataloader_train):
                num_samples = len(batch[0])
                total_samples += num_samples
                optimizer.zero_grad()
                loss = self.nll(batch)
                loss.backward()
                optimizer.step()
                # Track running statistics
                train_loss += loss.detach().cpu().item() * num_samples
            train_loss /= total_samples
            iter_per_sec = len(dataloader_train) / (time.time() - t0)
            out_path = os.path.join(logger.get_dir(), f"model_epoch{i}.pt")
            torch.save(self.model.state_dict(), out_path)  # save model
            self.logs['train loss'].append(train_loss)

            if verbose:
                logger.log('epoch: {:3d}\t train loss: {:0.4f}\t iter/sec: {:0.2f}'.
                           format(i, train_loss, iter_per_sec))

    @property
    def h_g(self):
        """Differential entropy for a N(mu, Sigma), where Sigma matches data, with same dimension as data."""
        return 0.5 * self.d * math.log(2 * math.pi * math.e) + 0.5 * self.log_eigs.sum().item()

    def mmse_g(self, logsnr):
        """The analytic MMSE for a Gaussian with the same eigenvalues as the data in a Gaussian noise channel."""
        return torch.sigmoid(logsnr + self.log_eigs.view((-1, 1))).sum(axis=0)  # *logsnr integration, see note
    
    def logistic_integrate(self, npoints, loc, scale, clip=4., device='cpu', deterministic=False):
        """Return sample point and weights for integration, using
        a truncated logistic distribution as the base, and importance weights.
        These logsnr values are the integration points.
        """
        loc, scale, clip = torch.tensor(loc, device=device), torch.tensor(scale, device=device), torch.tensor(clip, device=device)

        # IID samples from uniform, use inverse CDF to transform to target distribution
        if deterministic:
            torch.manual_seed(0)
        ps = torch.rand(npoints, dtype=loc.dtype, device=device)
        ps = torch.sigmoid(-clip) + (torch.sigmoid(clip) - torch.sigmoid(-clip)) * ps  # Scale quantiles to clip
        logsnr = loc + scale * torch.logit(ps)  # Using quantile function for logistic distribution

        # importance weights
        weights = scale * torch.tanh(clip / 2) / (torch.sigmoid((logsnr - loc)/scale) * torch.sigmoid(-(logsnr - loc)/scale))
        return logsnr, weights
