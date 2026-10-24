"""Sanity checks for mujoco/leg_assembly.xml: DOF after closure, constraint drift while
both servos track sine targets, kinematic range of the knee servo, and knee/servo coupling."""
import os, numpy as np, mujoco
HERE = os.path.dirname(os.path.abspath(__file__))
m = mujoco.MjModel.from_xml_path(os.path.join(HERE, '..', 'mujoco', 'leg_assembly.xml'))
d = mujoco.MjData(m)
PAIRS = [(m.site(a).id, m.site(b).id) for a, b in
         [('loop1_coupler_pin', 'loop1_bellcrank_pin'), ('loop2_pushrod_pin', 'loop2_shin_pin')]]
q = lambda j: d.qpos[m.joint(j).qposadr[0]]
def gaps(): return [np.linalg.norm(d.site_xpos[a] - d.site_xpos[b]) for a, b in PAIRS]
def Jc():
    out = []
    for a, b in PAIRS:
        j1 = np.zeros((3, m.nv)); j2 = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, j1, None, a); mujoco.mj_jacSite(m, d, j2, None, b); out.append(j1 - j2)
    return np.vstack(out)

mujoco.mj_resetDataKeyframe(m, d, m.key('home').id); mujoco.mj_forward(m, d)
s = np.linalg.svd(Jc(), compute_uv=False)
print(f'nv={m.nv}, constraint rows=6, rank={int((s > 1e-6 * s[0]).sum())} -> mobility {m.nv - int((s > 1e-6 * s[0]).sum())} (expect 2)')

# dynamic test: both servos follow sines
A = np.radians([25, 20]); f = [0.7, 1.1]; worst = [0, 0]; T = 6.0
while d.time < T:
    d.ctrl[:] = A * np.sin(2 * np.pi * np.array(f) * d.time)
    mujoco.mj_step(m, d)
    g = gaps(); worst = [max(w, x) for w, x in zip(worst, g)]
ok = np.all(np.isfinite(d.qpos))
print(f'{T:.0f}s sine sweep (+-25 deg thigh, +-20 deg knee servo): finite={ok}, '
      f'max pin gap loop1={worst[0]*1e6:.2f} um, loop2={worst[1]*1e6:.2f} um')

# kinematic range of the knee servo (thigh at 0): continuation outward from home in 0.5 deg steps,
# stopping at a toggle/lockup (passive-joint Jacobian goes singular) or when the pins can't meet.
act = [m.joint(j).dofadr[0] for j in ('dof_thigh_motor', 'dof_shin_motor')]
pas = [i for i in range(m.nv) if i not in act]
def project(iters=40):
    for _ in range(iters):
        mujoco.mj_forward(m, d); r = np.concatenate([d.site_xpos[a] - d.site_xpos[b] for a, b in PAIRS])
        J = Jc(); J[:, act] = 0
        mujoco.mj_integratePos(m, d.qpos, -np.linalg.pinv(J, rcond=1e-6) @ r, 1.0)
    mujoco.mj_forward(m, d)
    Jp = Jc()[:, pas]; Jp = Jp[[0, 1, 3, 4]]          # in-plane rows
    return max(gaps()), np.linalg.svd(Jp, compute_uv=False)[-1]
def sweep(direction, thigh=0.0):
    mujoco.mj_resetDataKeyframe(m, d, m.key('home').id)
    d.qpos[m.joint('dof_thigh_motor').qposadr[0]] = thigh
    _, s0 = project(); out = []
    for k in range(1, 361):
        deg = direction * 0.5 * k
        d.qpos[m.joint('dof_shin_motor').qposadr[0]] = np.radians(deg)
        err, smin = project()
        if err > 1e-7 or smin < 0.05 * s0: return out, deg, ('no assembly' if err > 1e-7 else 'toggle/singular')
        out.append((deg, np.degrees(q('dof_knee')), smin / s0))
    return out, None, 'no limit within 180'
up, lim_up, why_up = sweep(+1); dn, lim_dn, why_dn = sweep(-1)
rows = sorted(dn + [(0.0, np.degrees(m.key('home').qpos[m.joint('dof_knee').qposadr[0]]), 1.0)] + up)
print(f'knee servo range (thigh=0): {rows[0][0]:.1f} .. {rows[-1][0]:.1f} deg '
      f'[lower stops: {why_dn}, upper stops: {why_up}]')
print('  servo_deg  knee_deg  conditioning(1=home)')
for r in rows[::max(1, len(rows) // 14)] + [rows[-1]]:
    print(f'   {r[0]:6.1f}   {r[1]:8.2f}   {r[2]:.2f}')
k = np.array(rows); g = np.gradient(k[:, 1], k[:, 0])
print(f'transmission d(knee)/d(servo): {g[np.argmin(abs(k[:,0]))]:.3f} at home, range {g.min():.3f} .. {g.max():.3f}')
def solve(servo_b, thigh):
    mujoco.mj_resetDataKeyframe(m, d, m.key('home').id)
    d.qpos[m.joint('dof_shin_motor').qposadr[0]] = servo_b; d.qpos[m.joint('dof_thigh_motor').qposadr[0]] = thigh
    project()
# coupling: thigh servo moves, knee servo held -> knee relative angle changes?
solve(0.0, 0.0); k0 = q('dof_knee'); solve(0.0, np.radians(10)); k1 = q('dof_knee')
print(f'coupling: +10 deg thigh servo with knee servo held -> knee joint changes {np.degrees(k1-k0):.2f} deg')
