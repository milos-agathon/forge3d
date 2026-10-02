"""Second-order term: sun single-scattered in the medium, arriving at the first terrain hit
(surface irradiance from the lit medium). Reuses nephele_first_order.py definitions without
recomputing its expensive passes."""
import sys, numpy as np
from pathlib import Path
src = open(Path(__file__).with_name('nephele_first_order.py'), encoding='utf-8').read()
head = src.split('print("terrain hits ..."')[0]
sys.argv = [None, sys.argv[1], str(Path(__file__).with_name('first-order'))]
exec(head)
geo = np.load(OUT / 'geometry.npz'); fo = np.load(OUT / 'first-order.npz')
hit, t_hit = geo['hit'], geo['t_hit']
p_hit = eye + dirs * np.where(hit, t_hit, 0.0)[..., None]
# Sun transmittance (medium) x terrain visibility on a 1-unit lattice over the medium box.
gx = np.arange(bmin[0], bmax[0] + 1e-9, 1.0); gy = np.arange(bmin[1], bmax[1] + 1e-9, 1.0); gz = np.arange(bmin[2], bmax[2] + 1e-9, 1.0)
P = np.stack(np.meshgrid(gx, gy, gz, indexing='ij'), -1)
sd = np.broadcast_to(to_sun, P.shape)
Tsun = np.exp(-optical_depth(P, sd, medium_exit(P, sd), 0.25))
vis = ~occluded(P, sd, np.full(P.shape[:-1], 400.0)) & ~(P[..., 1] < np.nan_to_num(height(P[..., 0], P[..., 2]), nan=-1e9))
Lsun_grid = Tsun * vis[..., None]
def lsun(p):
    u = np.clip((p - bmin) / 1.0, 0, np.asarray(Lsun_grid.shape[:3]) - 1 - 1e-9)
    i = np.floor(u).astype(int); f = u - i; out = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[..., 0] if dx else 1 - f[..., 0]) * (f[..., 1] if dy else 1 - f[..., 1]) * (f[..., 2] if dz else 1 - f[..., 2])
                out = out + Lsun_grid[np.minimum(i[..., 0] + dx, len(gx) - 1), np.minimum(i[..., 1] + dy, len(gy) - 1), np.minimum(i[..., 2] + dz, len(gz) - 1)] * w[..., None]
    return out
n_hit = normal_at(p_hit[..., 0], p_hit[..., 2])
exec(src[src.index('def fibonacci'):src.index('SPHERE = ')])
SPH = fibonacci(192); dw = 4 * np.pi / len(SPH)
idx = np.argwhere(hit); ph = p_hit[hit]; nh = n_hit[hit]
E = np.zeros((len(ph), 3)); STEP = 0.5
for w in SPH:
    cw = nh @ w
    use = cw > 0
    if not use.any():
        continue
    o = ph[use] + nh[use] * 1e-3; L = np.zeros((use.sum(), 3)); tau = np.zeros((use.sum(), 3)); alive = np.ones(use.sum(), bool)
    reach = medium_exit(o, np.broadcast_to(w, o.shape)); n = int(np.ceil(reach.max() / STEP))
    for i in range(n):
        t = (i + 0.5) * STEP; p = o + w * t
        alive &= (t < reach)
        d = p[:, 1] - height(p[:, 0], p[:, 2]); alive &= ~(np.isfinite(d) & (d < 0))
        if not alive.any(): break
        rho = np.where(alive, density(p), 0.0)
        tm = tau + (rho * STEP * 0.5)[:, None] * sigma_t
        L += np.exp(-tm) * (rho * STEP)[:, None] * sigma_s * hg(w @ to_sun) * sun_L * lsun(p)
        tau += (rho * STEP)[:, None] * sigma_t
    E[use] += (albedo / np.pi * dw) * cw[use, None] * L
S2 = np.zeros(dirs.shape); S2[hit] = E
np.save(OUT / 'surface-cloudlight.npy', S2)
LUMA = np.asarray([0.2126, 0.7152, 0.0722])
terr = np.load(FIX / 'terrain-mask.npy').astype(bool)
print('terrain surface sun-in-scatter (2nd order) lum', float((S2[terr] @ LUMA).mean()))
print('T_cam * that', float(((fo['T_cam'] * S2)[terr] @ LUMA).mean()))
