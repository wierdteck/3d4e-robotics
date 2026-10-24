"""Open the closed-loop leg in the MuJoCo viewer and sweep both servos so the linkage moves.
    python scripts/demo_viewer.py            # animated
    python scripts/demo_viewer.py --manual   # drive the servos yourself (Control panel sliders)
Orange dots = the two loop-closure pins; they should stay together.
"""
import os, sys, time, numpy as np, mujoco, mujoco.viewer
HERE = os.path.dirname(os.path.abspath(__file__))
m = mujoco.MjModel.from_xml_path(os.path.join(HERE, '..', 'mujoco', 'leg_assembly.xml'))
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key('home').id)
m.vis.global_.fovy = 30
manual = '--manual' in sys.argv
# Knee servo stays inside its usable range (-81.5..+49.5 deg); see mujoco/README.md
AMP, FREQ = np.radians([30.0, 35.0]), np.array([0.25, 0.4])
if manual:
    m.actuator_ctrlrange[:] = np.radians([[-90, 90], [-75, 45]]); m.actuator_ctrllimited[:] = 1
with mujoco.viewer.launch_passive(m, d) as v:
    v.cam.lookat[:] = [0.03, -0.02, 0.01]; v.cam.distance = 0.35
    v.cam.azimuth, v.cam.elevation = 90, -89
    v.opt.sitegroup[:] = 1
    while v.is_running():
        t0 = time.time()
        if not manual:
            d.ctrl[:] = AMP * np.sin(2 * np.pi * FREQ * d.time)
        for _ in range(int(1 / 60 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        v.sync()
        time.sleep(max(0, 1 / 60 - (time.time() - t0)))
