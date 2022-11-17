import torch
import torch.utils.data
from modules.scatter import Scatter
from pytorch_lightning.core.module import LightningModule
from torch import nn
from torch.nn import functional as F


class Scattering(object):

    def __init__(self, device, in_channel: int = 15):
        model = Scatter(
            in_channels=in_channel, trainable_laziness=False, device=device)
        model.eval()
        self.model = model

    def __call__(self, sample):
        with torch.no_grad():
            to_return, _ = self.model(sample)

        return to_return

    def out_shape(self):
        return self.model.out_shape()


class GSAE(LightningModule):
    def __init__(self, hparams):
        super(GSAE, self).__init__()

        self.input_dim = hparams.input_dim
        self.bottle_dim = hparams.bottle_dim
        self.hidden_dim = hparams.hidden_dim
        self.learning_rate = hparams.learning_rate
        self.alpha = hparams.alpha
        self.beta = hparams.beta
        self.n_epochs = hparams.n_epochs
        self.len_epoch = hparams.len_epoch
        self.batch_size = hparams.batch_size

        self.loss_list = []

        if len(hparams.gpu_ids) > 0:
            self.dev_type = 'cuda'

        if len(hparams.gpu_ids) == 0:
            self.dev_type = 'cpu'

        self.scattering_network = Scattering(
            device=self.dev_type, in_channel=self.input_dim)

        self.fc11 = nn.Linear(
            self.scattering_network.out_shape(), self.hidden_dim)
        self.bn11 = nn.BatchNorm1d(self.hidden_dim)

        self.fc12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bn12 = nn.BatchNorm1d(self.hidden_dim)

        self.fc21 = nn.Linear(self.hidden_dim, self.bottle_dim)
        self.fc22 = nn.Linear(self.hidden_dim, self.bottle_dim)

        self.fc3 = nn.Linear(self.bottle_dim, self.hidden_dim)
        self.fc4 = nn.Linear(
            self.hidden_dim, self.scattering_network.out_shape())
        # energy prediction
        self.regfc1 = nn.Linear(self.bottle_dim, 20)
        self.regfc2 = nn.Linear(20, 1)

        self.eps = 1e-5

    # def kl_div(self, mu, logvar):
    #     KLD_element = mu.pow(2).add(
    #         logvar.exp()).mul(-1).add(1).add(logvar)
    #     KLD = torch.sum(KLD_element).mul(-0.5)

    #     return KLD

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def encode(self, batch):
        # get batch of coeffs
        coeffs = self.scattering_network(batch)

        h = self.bn11(F.relu(self.fc11(coeffs)))
        h = self.bn12(F.relu(self.fc12(h)))
        mu = self.fc21(h)
        logvar = self.fc22(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar, coeffs

    def predict(self, z):
        h = F.relu(self.regfc1(z))
        y_pred = self.regfc2(h)
        return y_pred

    def forward(self, batch):
        # encoding
        z, mu, logvar, coeffs = self.encode(batch)
        # predict
        y_pred = self.predict(z)
        # recon
        coeffs_recon = self.decode(z)

        return coeffs_recon, y_pred, mu, logvar, coeffs

    def loss_multi_GSAE(self,
                        recon_x, x,
                        mu, logvar,
                        y_pred, y,
                        alpha, beta, batch_idx):

        # reconstruction loss
        recon_loss = nn.MSELoss()(recon_x.flatten(), x.flatten())

        # regression loss
        reg_loss = nn.MSELoss()(y_pred, y)

        # kl divergence
        # KLD = self.kl_div(mu, logvar)

        # loss annealing
        # num_epochs = self.n_epochs - 5
        # total_batches = self.len_epoch * num_epochs
        # weight = min(1, float(self.trainer.global_step) / float(total_batches))
        # kl_loss = weight * KLD

        reg_loss = alpha * reg_loss.mean()

        # kl_loss = beta * kl_loss

        # total_loss = recon_loss + reg_loss + kl_loss
        total_loss = recon_loss + reg_loss

        self.loss_list.append(total_loss.item())

        log_losses = {'loss': total_loss.detach(),
                      'recon_loss': recon_loss.detach(),
                      'pred_loss': reg_loss.detach(),
                      #   'kl_loss': kl_loss.detach(),
                      }

        return total_loss, log_losses

    def get_loss_list(self):
        return self.loss_list

    def relabel(self, loss_dict, label):
        loss_dict = {label + str(key): val for key, val in loss_dict.items()}
        return loss_dict

    def shared_step(self, batch):
        y = batch.time
        y = y.float()[:, None]

        coeffs_recon, y_hat, mu, logvar, coeffs = self(batch)

        return coeffs, coeffs_recon, y_hat, mu, logvar, y

    def training_step(self, batch, batch_idx):
        x, x_hat, y_hat, mu, logvar, y = self.shared_step(batch)

        loss, log_losses = self.loss_multi_GSAE(recon_x=x_hat, x=x,
                                                mu=mu, logvar=logvar,
                                                y_pred=y_hat, y=y,
                                                alpha=self.alpha, beta=self.beta,
                                                batch_idx=batch_idx)

        log_losses = self.relabel(log_losses, 'train_')
        self.log_dict(log_losses, on_step=True, on_epoch=True,
                      batch_size=self.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        x, x_hat, y_hat, mu, logvar, y = self.shared_step(batch)

        loss, log_losses = self.loss_multi_GSAE(recon_x=x_hat, x=x,
                                                mu=mu, logvar=logvar,
                                                y_pred=y_hat, y=y,
                                                alpha=self.alpha, beta=self.beta,
                                                batch_idx=batch_idx)

        log_losses = self.relabel(log_losses, 'val_')

        self.log_dict(log_losses, on_step=True, on_epoch=True,
                      batch_size=self.batch_size)

        return loss

    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     avg_reconloss = torch.stack([x['val_recon_loss']
    #                                 for x in outputs]).mean()
    #     avg_regloss = torch.stack([x['val_pred_loss'] for x in outputs]).mean()
    #     avg_klloss = torch.stack([x['val_kl_loss'] for x in outputs]).mean()

    #     tensorboard_logs = {'val_loss': avg_loss,
    #                         'val_avg_recon_loss': avg_reconloss,
    #                         'val_avg_pred_loss': avg_regloss,
    #                         'val_avg_kl_loss': avg_klloss,
    #                         }

    #     return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
