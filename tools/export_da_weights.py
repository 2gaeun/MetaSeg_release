"""One-time trusted-checkpoint EMA extraction; not a deployment dependency."""
import argparse
import gc
import hashlib
import json
import pickle
from pathlib import Path
import torch
from bms_deploy.da_network import I2SBNet


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--architecture-pkl', type=Path, required=True)
    p.add_argument('--t1ce2bb-checkpoint', type=Path, required=True)
    p.add_argument('--bb2t1ce-checkpoint', type=Path, required=True)
    a = p.parse_args()
    architecture = pickle.load(a.architecture_pkl.open('rb'))
    architecture.update(in_channels=4, out_channels=1, use_fp16=False)
    report = {}
    for direction, job, step, path in [('t1ce2bb',112788,175000,a.t1ce2bb_checkpoint),
                                     ('bb2t1ce',114599,150000,a.bb2t1ce_checkpoint)]:
        destination = a.bundle / 'weights' / f'{direction}_{job}_step{step:07d}'
        destination.mkdir(parents=True, exist_ok=False)
        opt = pickle.load((path.parent / 'options.pkl').open('rb'))
        required = dict(interval=1000, beta_max=.3, t0=.0001, T=1., context_slices=1,
                        cond_x1=True, endpoint_channels=1, condition_channels=3, output_channels=1,
                        ot_ode=False, use_fp16=False)
        for key, value in required.items():
            if getattr(opt, key) != value: raise ValueError(f'Unexpected option {key}')
        config = dict(schema='bms-i2sb-ema-v1', training_job=job, step=step, direction=direction,
                      architecture=architecture, **required)
        with torch.device('meta'): model = I2SBNet(config)
        print('EXPORT_LOADING', direction, flush=True)
        ckpt = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        model.load_state_dict(ckpt['net'], strict=True, assign=True)
        parameters = list(model.named_parameters())
        shadows = ckpt['ema']['shadow_params']
        if len(shadows) != len(parameters): raise ValueError('EMA parameter count mismatch')
        state = dict(ckpt['net'])
        for (name, parameter), value in zip(parameters, shadows):
            if value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise ValueError(f'EMA shape/dtype mismatch: {name}')
            state[name] = value
        model.load_state_dict(state, strict=True, assign=True)
        outfile = destination / 'ema.pt'
        torch.save(state, outfile)
        # Re-open the portable artifact and compare every tensor with the training EMA.
        saved = torch.load(outfile, map_location='cpu', weights_only=True, mmap=True)
        for key, value in state.items():
            if not torch.equal(saved[key], value): raise ValueError(f'Export mismatch: {key}')
        (destination / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        report[direction] = dict(training_job=job, step=step, source_checkpoint=str(path),
            source_bytes=path.stat().st_size, source_mtime_ns=path.stat().st_mtime_ns,
            ema_tensors=len(shadows), all_exported_tensors_equal=True,
            file=str(outfile.relative_to(a.bundle)), bytes=outfile.stat().st_size, sha256=digest(outfile))
        print('EXPORT_COMPLETE', json.dumps(report[direction]), flush=True)
        del ckpt, model, parameters, shadows, state, saved, parameter, value
        gc.collect()
    (a.bundle / 'weights/da_export.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__': main()
