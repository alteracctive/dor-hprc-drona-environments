import os
import builtins
from .helpers import (
    _py_str,
    _load_template,
    _LIGHTNING_IMPORT_BLOCK,
    _get_additional_imports,
    _get_data_helpers,
)

def _drona_msg(msg, level="warning"):
    if hasattr(builtins, "drona_add_message"):
        builtins.drona_add_message(msg, level)
    else:
        print(f"[{level.upper()}] {msg}")

def _gen_generative_script(
    exp_name, gen_dataset_type, gen_custom_path, latent_dim,
    ep, bs, lr, nw, seed_val, acc, dev, prec, log_n, log_dir,
    logger_lines, callback_block,
    gen_model_type="vae",
    gen_dataset_format="tensors",
):
    """Returns (train_script, prefetch_script_or_None)."""
    cache_dir = "./data"
    use_builtin = gen_dataset_type != "custom"

    if use_builtin:
        if gen_dataset_type == "llava-onevision":
            in_ch, img_sz = 3, 224
        else:
            raise ValueError(f"Prepared generative dataset '{gen_dataset_type}' is not available via cluster modules. Direct internet downloads are disabled. Please provide a custom dataset directory or select LLaVA-OneVision.")
        dm = f'''class LitDataModule(L.LightningDataModule):
    """DataModule managing downloads, setup, and data loading for VAE builtin datasets."""
    def __init__(self, data_dir=DATA_DIR, batch_size={bs}, num_workers={nw}):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_name = "{_py_str(gen_dataset_type)}"

    def setup(self, stage=None):
        # Load train and validation dataset splits
        self.train_ds = load_builtin_dataset(self.dataset_name, self.data_dir, train=True)
        self.val_ds = load_builtin_dataset(self.dataset_name, self.data_dir, train=False)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)
'''
    else:
        if not gen_custom_path:
            _drona_msg("Custom image folder path is required for Generative model.", "error")
            return None, None
        in_ch, img_sz = 3, 64   # sensible default for custom folders
        if gen_dataset_format == "images":
            dm = f'''class LitDataModule(L.LightningDataModule):
    """DataModule managing custom folder-based raw image datasets for VAE."""
    def __init__(self, data_root="{_py_str(gen_custom_path)}", batch_size={bs}, num_workers={nw}):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.Resize(({img_sz}, {img_sz})),
            transforms.ToTensor(),
        ])

    def setup(self, stage=None):
        self.train_ds = ImageFolder(os.path.join(self.data_root, "train"), transform=self.transform)
        self.val_ds = ImageFolder(os.path.join(self.data_root, "val"), transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)
'''
        else:
            dm = f'''class LitDataModule(L.LightningDataModule):
    """DataModule managing custom folder-based tensor datasets for VAE."""
    def __init__(self, data_root="{_py_str(gen_custom_path)}", batch_size={bs}, num_workers={nw}):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Loads class-sorted subdirectories of .pt tensors for VAE train and validation
        self.train_ds = TensorFolderDataset(os.path.join(self.data_root, "train"), image_size={img_sz})
        self.val_ds = TensorFolderDataset(os.path.join(self.data_root, "val"), image_size={img_sz})

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)
'''
        prefetch_script = None

    flat_size = in_ch * img_sz * img_sz

    # Build model block dynamically based on gen_model_type
    if gen_model_type == "gan":
        model_block = f'''class LitModel(L.LightningModule):
    """
    Generative Adversarial Network (GAN) with Generator and Discriminator.
    Uses manual optimization in PyTorch Lightning.
    """

    def __init__(self, flat_size={flat_size}, latent_dim={latent_dim}, lr={lr}):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # Generator: maps noise to reconstructed image space
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, flat_size),
            nn.Sigmoid(),   # Output pixels normalized to [0, 1]
        )
        
        # Discriminator: maps image to real/fake probability
        self.discriminator = nn.Sequential(
            nn.Linear(flat_size, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        b = x.size(0)

        opt_g, opt_d = self.optimizers()

        # Sample noise
        z = torch.randn(b, self.hparams.latent_dim, device=self.device)

        # Train Discriminator
        self.toggle_optimizer(opt_d)
        real_labels = torch.ones(b, 1, device=self.device)
        fake_labels = torch.zeros(b, 1, device=self.device)

        real_pred = self.discriminator(x)
        d_loss_real = F.binary_cross_entropy(real_pred, real_labels)

        fake_imgs = self(z)
        fake_pred = self.discriminator(fake_imgs.detach())
        d_loss_fake = F.binary_cross_entropy(fake_pred, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        self.manual_backward(d_loss)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

        # Train Generator
        self.toggle_optimizer(opt_g)
        fake_pred_g = self.discriminator(fake_imgs)
        g_loss = F.binary_cross_entropy(fake_pred_g, real_labels)

        self.manual_backward(g_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        self.log_dict({{"train_loss": d_loss + g_loss, "d_loss": d_loss, "g_loss": g_loss}}, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        b = x.size(0)
        
        # Compute validation losses
        z = torch.randn(b, self.hparams.latent_dim, device=self.device)
        fake_imgs = self(z)
        
        real_labels = torch.ones(b, 1, device=self.device)
        fake_labels = torch.zeros(b, 1, device=self.device)
        
        real_pred = self.discriminator(x)
        d_loss_real = F.binary_cross_entropy(real_pred, real_labels)
        fake_pred = self.discriminator(fake_imgs)
        d_loss_fake = F.binary_cross_entropy(fake_pred, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        g_loss = F.binary_cross_entropy(fake_pred, real_labels)
        
        self.log_dict({{"val_loss": d_loss + g_loss, "val_d_loss": d_loss, "val_g_loss": g_loss}})

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        return [opt_g, opt_d]
'''
    elif gen_model_type == "diffusion":
        model_block = f'''class LitModel(L.LightningModule):
    """
    Denoising Diffusion Probabilistic Model (DDPM).
    Learns to predict added noise at random timesteps of image corruption.
    """

    def __init__(self, flat_size={flat_size}, latent_dim={latent_dim}, lr={lr}, num_timesteps=1000):
        super().__init__()
        self.save_hyperparameters()

        # Simple MLP noise prediction model (can be extended to ConvNets)
        self.model = nn.Sequential(
            nn.Linear(flat_size + 1, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, flat_size),
        )

        # Dynamic Beta schedule
        beta = torch.linspace(1e-4, 0.02, num_timesteps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)

    def forward(self, x_t, t):
        # Concatenate time step t (normalized) with corrupted image x_t
        t_normalized = t.float().unsqueeze(-1) / self.hparams.num_timesteps
        inputs = torch.cat([x_t, t_normalized], dim=-1)
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        x_0, _ = batch
        x_0 = x_0.view(x_0.size(0), -1)
        b = x_0.size(0)

        # Sample random timestep
        t = torch.randint(0, self.hparams.num_timesteps, (b,), device=self.device)
        noise = torch.randn_like(x_0)

        # Extract coefficients
        alpha_bar_t = self.alpha_bar[t].unsqueeze(-1)

        # Compute corrupted image x_t
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1.0 - alpha_bar_t) * noise

        # Predict noise and optimize MSE
        noise_pred = self(x_t, t)
        loss = F.mse_loss(noise_pred, noise)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_0, _ = batch
        x_0 = x_0.view(x_0.size(0), -1)
        b = x_0.size(0)
        t = torch.randint(0, self.hparams.num_timesteps, (b,), device=self.device)
        noise = torch.randn_like(x_0)
        alpha_bar_t = self.alpha_bar[t].unsqueeze(-1)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1.0 - alpha_bar_t) * noise
        noise_pred = self(x_t, t)
        loss = F.mse_loss(noise_pred, noise)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
'''
    else: # vae
        model_block = f'''class LitModel(L.LightningModule):
    """
    Variational Autoencoder with fully-connected encoder/decoder.
    Optimizes the Evidence Lower Bound (ELBO): reconstruction loss + KL divergence.
    """

    def __init__(self, flat_size={flat_size}, latent_dim={latent_dim}, lr={lr}):
        super().__init__()
        self.save_hyperparameters()

        # Encoder: Projects input image to the latent distribution parameters (mu and log_var)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 512), 
            nn.ReLU(),
            nn.Linear(512, 256), 
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_log_var = nn.Linear(256, latent_dim)

        # Decoder: Maps latent sample z back into reconstruction space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), 
            nn.ReLU(),
            nn.Linear(256, 512), 
            nn.ReLU(),
            nn.Linear(512, flat_size),
            nn.Sigmoid(),   # Constrain output pixels to [0, 1] range
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def _elbo_loss(self, x, x_hat, mu, log_var):
        x_flat = x.view(x.size(0), -1)
        recon = F.binary_cross_entropy(x_hat, x_flat, reduction="sum") / x.size(0)
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.size(0)
        return recon + kl, recon, kl

    def _shared_step(self, batch, stage):
        x, _ = batch
        x_hat, mu, log_var = self(x)
        loss, recon, kl = self._elbo_loss(x, x_hat, mu, log_var)
        
        self.log(f"{{stage}}_loss", loss, prog_bar=True)
        self.log(f"{{stage}}_recon", recon)
        self.log(f"{{stage}}_kl", kl)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
'''

    template = _load_template("generative")
    train_script = template.replace("# __LIGHTNING_IMPORT_BLOCK__", _LIGHTNING_IMPORT_BLOCK.strip())
    train_script = train_script.replace("# __ADDITIONAL_IMPORTS__", _get_additional_imports("builtin" if use_builtin else "custom", gen_dataset_type if use_builtin else "", dataset_format=gen_dataset_format).strip())
    train_script = train_script.replace("__DATA_DIR__", _py_str(cache_dir))
    train_script = train_script.replace("__LOG_DIR__", _py_str(log_dir))
    train_script = train_script.replace("__EXPERIMENT_NAME__", _py_str(exp_name))
    train_script = train_script.replace("__LR__", str(lr))
    train_script = train_script.replace("__BATCH_SIZE__", str(bs))
    train_script = train_script.replace("__NUM_WORKERS__", str(nw))
    train_script = train_script.replace("__MAX_EPOCHS__", str(ep))
    train_script = train_script.replace("__ACCELERATOR__", _py_str(acc))
    train_script = train_script.replace("__DEVICES__", str(dev))
    train_script = train_script.replace("__PRECISION__", str(prec) if str(prec).isdigit() else f'"{_py_str(prec)}"')
    train_script = train_script.replace("__LOG_EVERY_N_STEPS__", str(log_n))
    train_script = train_script.replace("__SEED__", str(seed_val))
    train_script = train_script.replace("__IN_CHANNELS__", str(in_ch))
    train_script = train_script.replace("__IMAGE_SIZE__", str(img_sz))
    train_script = train_script.replace("__LATENT_DIM__", str(latent_dim))
    train_script = train_script.replace("__FLAT_SIZE__", str(flat_size))
    
    # Generate dynamic helpers tailored only to the chosen dataset
    data_helpers = _get_data_helpers("builtin" if use_builtin else "custom", gen_dataset_type if use_builtin else "", cv_model_arch="cnn", dataset_format=gen_dataset_format)
    train_script = train_script.replace("# __DATA_HELPERS_BLOCK__", data_helpers.strip())
    
    train_script = train_script.replace("# __DATAMODULE_BLOCK__", dm.strip())
    train_script = train_script.replace("# __MODEL_BLOCK__", model_block.strip())
    train_script = train_script.replace("        # __LOGGER_LINES__", "\n".join(logger_lines))
    train_script = train_script.replace("    # __CALLBACK_BLOCK__", "    " + callback_block.replace("\n", "\n    "))

    return train_script, None

