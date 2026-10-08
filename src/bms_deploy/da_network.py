"""Inference-only I2SB wrapper using the original ADM and diffusion code."""
import json
from pathlib import Path
import numpy as np
import torch
from ._vendor.guided_diffusion.script_util import create_model
from ._vendor.i2sb_diffusion import Diffusion
from ._vendor.sampling_util import space_indices


class I2SBNet(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.diffusion_model = create_model(**config['architecture'])
        levels = torch.linspace(config['t0'], config['T'], config['interval'], device='cpu') * config['interval']
        self.register_buffer('noise_levels', levels, persistent=False)

    def forward(self, x, steps, cond):
        return self.diffusion_model(torch.cat([x, cond], dim=1), self.noise_levels[steps].detach())


class I2SBInference:
    def __init__(self, weight_dir, device='cuda'):
        self.directory = Path(weight_dir)
        self.config = json.loads((self.directory / 'config.json').read_text())
        c = self.config
        if c['schema'] != 'bms-i2sb-ema-v1' or c['interval'] != 1000 or c['context_slices'] != 1:
            raise ValueError('Unsupported bundled I2SB configuration')
        if c['architecture']['in_channels'] != 4 or c['architecture']['out_channels'] != 1:
            raise ValueError('Expected center endpoint + three condition channels')
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        state = torch.load(self.directory / 'ema.pt', map_location='cpu', weights_only=True, mmap=True)
        with torch.device('meta'):
            self.net = I2SBNet(c)
        self.net.load_state_dict(state, strict=True, assign=True)
        self.net.to(self.device).eval()
        b = (torch.linspace(1e-4 ** .5, (c['beta_max'] / c['interval']) ** .5,
                            c['interval'], dtype=torch.float64) ** 2).numpy()
        b = np.concatenate([b[:c['interval']//2], np.flip(b[:c['interval']//2])])
        self.diffusion = Diffusion(b, self.device)

    @torch.no_grad()
    def sample(self, x1, cond, nfe=50):
        if not 0 < nfe < self.config['interval']:
            raise ValueError('nfe must be between 1 and 999')
        x1, cond = x1.to(self.device), cond.to(self.device)
        def predict(xt, step):
            steps = torch.full((len(xt),), step, device=self.device, dtype=torch.long)
            out = self.net(xt, steps, cond)
            return xt - self.diffusion.get_std_fwd(steps, xdim=xt.shape[1:]) * out
        xs, _ = self.diffusion.ddpm_sampling(
            space_indices(self.config['interval'], nfe + 1), predict, x1,
            ot_ode=self.config['ot_ode'], log_steps=[0], verbose=False)
        return xs[:, 0, 0]

    def release_gpu(self):
        self.net.to('cpu')
        self.diffusion = None
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
