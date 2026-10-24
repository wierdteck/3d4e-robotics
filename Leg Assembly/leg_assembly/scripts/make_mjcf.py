"""Convert leg_assembly_loopcut.urdf -> mujoco/leg_assembly.xml and close both loops with
<equality><connect site1 site2/></equality>.

Pin locations (CAD hole centres, not the export pose, so a mis-mated pose can't get baked in):
  loop1  coupler (hip_link_2) <-> bellcrank (servo_horn_v35): the cut joint `closing_left_horn` origin
  loop2  pushrod (thigh_link_2) far hole  <-> shin crank hole (Shin_Ankle_Link Body6)
         pushrod hole centre in its link frame: (113.010, 7.102) mm; shin hole centre (world, export
         pose): (89.719, -40.474) mm, found by circle-fitting the STL rings (r = 1.6 / 1.7 mm).
The export pose leaves loop 2 open by ~0.55 mm, so a closed `home` keyframe is solved and written.
"""
import os, sys, numpy as np, mujoco, xml.etree.ElementTree as ET
HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, '..', 'urdf', 'leg_assembly_loopcut.urdf')
OUT = os.path.join(HERE, '..', 'mujoco', 'leg_assembly.xml')

COUPLER, BELL = 'hip_link_2_v8_1__body1', 'servo_horn_v35_1__body1'
PUSHROD, SHIN = 'thigh_link_2_v10_1__body1', 'shin_ankle_link_v33_1__body5'
LOOP1_PIN_IN_COUPLER = np.array([-0.0283319, -0.0139777, 0.0048])   # closing_left_horn origin
PUSHROD_PIN_LOCAL = np.array([0.113010, 0.007102, -0.0022])
SHIN_PIN_WORLD_XY = np.array([0.089719, -0.040474])
ACTUATED = ['dof_thigh_motor', 'dof_shin_motor']
KP = 1e-3          # placeholder: scale with real masses / servo specs (see README)

def fmt(v): return ' '.join('%.9g' % x for x in v)

def main():
    # 1. URDF -> raw MJCF (inject compiler options; keep meshes as visual geoms)
    txt = open(URDF).read()
    txt = txt.replace('package://leg_assembly/meshes/', '')
    txt = txt.replace('<robot name="leg_assembly">',
        '<robot name="leg_assembly"><mujoco><compiler meshdir="../meshes/" discardvisual="false" '
        'fusestatic="true" strippath="true" balanceinertia="true"/></mujoco>', 1)
    cwd = os.getcwd(); os.chdir(os.path.dirname(URDF))   # meshdir is resolved from here
    try:
        m = mujoco.MjModel.from_xml_string(txt)
    finally:
        os.chdir(cwd)
    mujoco.mj_saveLastXML(OUT, m)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)

    def body(n): return m.body(n).id
    def to_local(b, pw):  # world point -> body-local
        return d.xmat[b].reshape(3, 3).T @ (pw - d.xpos[b])
    def to_world(b, pl): return d.xpos[b] + d.xmat[b].reshape(3, 3) @ pl

    # 2. pin sites in each body's own frame
    p1w = to_world(body(COUPLER), LOOP1_PIN_IN_COUPLER)
    p2w = to_world(body(PUSHROD), PUSHROD_PIN_LOCAL)
    shin_pin_w = np.array([*SHIN_PIN_WORLD_XY, p2w[2]])   # same plane as the pushrod hole
    sites = {
        (COUPLER, 'loop1_coupler_pin'): to_local(body(COUPLER), p1w),
        (BELL, 'loop1_bellcrank_pin'): to_local(body(BELL), p1w),
        (PUSHROD, 'loop2_pushrod_pin'): PUSHROD_PIN_LOCAL,
        (SHIN, 'loop2_shin_pin'): to_local(body(SHIN), shin_pin_w),
    }
    print('export-pose gap  loop1 %.3f mm   loop2 %.3f mm' %
          (0.0, 1e3 * np.linalg.norm(p2w - shin_pin_w)))

    # 3. edit the MJCF
    tree = ET.parse(OUT); root = tree.getroot()
    root.set('model', 'leg_assembly_closed')
    opt = root.find('option')
    if opt is None:
        opt = ET.Element('option'); root.insert(1, opt)
    opt.set('timestep', '0.001'); opt.set('integrator', 'implicitfast')
    bodies = {b.get('name'): b for b in root.iter('body')}
    for (bn, sn), pos in sites.items():
        ET.SubElement(bodies[bn], 'site', name=sn, pos=fmt(pos), size='0.0015', rgba='1 0.5 0 1')
    for g in root.iter('geom'):            # meshes are visual only (URDF has no <collision>)
        g.set('contype', '0'); g.set('conaffinity', '0')

    eq = ET.SubElement(root, 'equality')
    ET.SubElement(eq, 'connect', name='loop1_close', site1='loop1_coupler_pin', site2='loop1_bellcrank_pin',
                  solref='0.002 1', solimp='0.95 0.99 0.001')
    ET.SubElement(eq, 'connect', name='loop2_close', site1='loop2_pushrod_pin', site2='loop2_shin_pin',
                  solref='0.002 1', solimp='0.95 0.99 0.001')

    ex = ET.SubElement(root, 'contact')
    for a, b in [(COUPLER, BELL), (PUSHROD, SHIN)]:
        ET.SubElement(ex, 'exclude', body1=a, body2=b)

    act = ET.SubElement(root, 'actuator')
    for jn in ACTUATED:
        ET.SubElement(act, 'position', name=jn, joint=jn, kp=str(KP), dampratio='1')

    ET.indent(tree, '  '); tree.write(OUT, encoding='unicode')

    # 4. solve a closed `home` pose (servos at 0) by projecting onto the constraints
    m = mujoco.MjModel.from_xml_path(OUT); d = mujoco.MjData(m)
    act_dofs = [m.joint(j).dofadr[0] for j in ACTUATED]
    pairs = [(m.site(a).id, m.site(b).id) for a, b in
             [('loop1_coupler_pin', 'loop1_bellcrank_pin'), ('loop2_pushrod_pin', 'loop2_shin_pin')]]
    def residual():
        mujoco.mj_forward(m, d)
        r, J = [], []
        for s1, s2 in pairs:
            j1 = np.zeros((3, m.nv)); j2 = np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, j1, None, s1); mujoco.mj_jacSite(m, d, j2, None, s2)
            r.append(d.site_xpos[s1] - d.site_xpos[s2]); J.append(j1 - j2)
        return np.concatenate(r), np.vstack(J)
    for _ in range(50):
        r, Jc = residual()
        if np.linalg.norm(r) < 1e-12: break
        Jc[:, act_dofs] = 0                                  # hold both servos at 0
        mujoco.mj_integratePos(m, d.qpos, -np.linalg.pinv(Jc, rcond=1e-8) @ r, 1.0)
    r, _ = residual()
    print('home pose residual %.2e m; passive joint shifts (deg):' % np.linalg.norm(r),
          {m.joint(i).name: round(float(np.degrees(d.qpos[m.joint(i).qposadr[0]])), 3)
           for i in range(m.njnt) if abs(d.qpos[m.joint(i).qposadr[0]]) > 1e-6})
    tree = ET.parse(OUT); root = tree.getroot()
    kf = ET.SubElement(root, 'keyframe')
    ET.SubElement(kf, 'key', name='home', qpos=fmt(d.qpos), ctrl=fmt(np.zeros(m.nu)))
    ET.indent(tree, '  '); tree.write(OUT, encoding='unicode')
    print('wrote', os.path.normpath(OUT))

if __name__ == '__main__':
    main()
