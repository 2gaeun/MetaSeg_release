# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Extracted unchanged from I2SB i2sb/util.py. See LICENSE_I2SB.
def space_indices(num_steps, count):
    assert count <= num_steps
    frac_stride = 1 if count <= 1 else (num_steps - 1) / (count - 1)
    cur_idx = 0.0
    taken_steps = []
    for _ in range(count):
        taken_steps.append(round(cur_idx))
        cur_idx += frac_stride
    return taken_steps


def unsqueeze_xdim(z, xdim):
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]
