# Leg with closed loops in MuJoCo

    python scripts/make_cut_urdf.py   # urdf/leg_assembly.urdf -> urdf/leg_assembly_loopcut.urdf
    python scripts/make_mjcf.py       # -> mujoco/leg_assembly.xml (+ connects, actuators, `home` key)
    python scripts/check_loops.py     # DOF, pin drift, servo range
    python -m mujoco.viewer --mjcf="mujoco/leg_assembly.xml"   # load key "home" before playing

## Cuts (one joint per loop)
| loop | path | cut | closed by |
|---|---|---|---|
| 1 | knee servo -> horn -> hip_link_2 (coupler) -> servo_horn_v35 (bellcrank) -> bearing on thigh pivot | `closing_left_horn` (coupler-bellcrank pin) | `connect loop1_close` |
| 2 | bellcrank -> thigh_link_2 (pushrod) -> shin crank -> knee -> thigh | pushrod-to-shin pin (never mated in Onshape) | `connect loop2_close` |

Onshape's own cut (weld at bearing body10 via `...__loop_closure` dummy) is undone: the bellcrank now
hangs off the thigh bearing through `dof_thigh`. The mechanism is planar, so a single point
`connect` per loop is enough. Tree joints: 6 hinges, 2 actuated (`dof_thigh_motor`, `dof_shin_motor`).

## Things to know
- **Export pose isn't closed for loop 2.** The pushrod hole is 0.55 mm from the shin hole (both found by
  circle-fitting the STLs). The sites sit on the true hole centres; keyframe `home` is the solved closed
  pose (knee -0.56 deg, pushrod -0.24 deg from export). Always reset to `home`, not qpos0.
- **Rounded right angles snapped.** Onshape writes pi/2 as 1.5708, which tilts the hinge axes ~4 urad and
  makes the loops fight out-of-plane. The cut URDF snaps these values (moves parts < 0.3 um).
- **Knee servo is a crank-rocker.** Usable `dof_shin_motor` range is about -81.5 to +49.5 deg (any thigh angle).
  At -81.5 deg the pushrod/shin loop goes singular; past +49.5 deg the knee reverses direction (dead point).
  Set ctrlrange/joint range inside that, with margin. Knee/servo ratio is -0.87 at home and varies with angle.
- **The thigh servo moves the knee too:** with the knee servo held, +10 deg of thigh changes `dof_knee` by 9.66 deg.
- **Masses are about 1000x too small** in the Onshape export (thigh 0.02 g). `KP` in make_mjcf.py is a
  placeholder sized for these masses: fix masses, then set kp/forcerange from the servo datasheet.
- Meshes are visual only (contype=0); the URDF has no collision geometry. Add a foot sphere before ground contact.
- Pin drift in a 6 s +-25/+-20 deg sine test: about 20 um. Tighten `solref` on the connects if needed (keep it at least 2x timestep).
