"""Build leg_assembly_loopcut.urdf from the Onshape export: a spanning tree with exactly
one joint removed per kinematic loop. The loops are re-closed in MuJoCo (see make_mjcf.py).

Loop 1 (knee servo -> horn -> hip_link_2 coupler -> servo_horn_v35 bellcrank -> thigh bearing)
    CUT at `closing_left_horn` (coupler <-> bellcrank pin).
    Onshape had cut this loop at the bearing weld instead (body10__1__loop_closure dummy); that is
    undone here: the bellcrank now hangs off the thigh-side bearing through `dof_thigh`, which is
    a real revolute, so the tree is fully articulated and only a pin (connect) is needed to close it.
Loop 2 (bellcrank -> thigh_link_2 pushrod -> shin crank -> knee -> thigh)
    Not present in the Onshape export at all (pushrod far end was never mated). Left open in the
    tree; the pushrod->shin pin is closed in MuJoCo.
"""
import copy, sys, os, xml.etree.ElementTree as ET
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from urdf_fk import world, parse_origin

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'urdf', 'leg_assembly.urdf')
DST = os.path.join(HERE, '..', 'urdf', 'leg_assembly_loopcut.urdf')

B8, B9, B10 = ('4668k144_permanently_lubricated_stainless_steel_ball_bearing_v1_1__body%d' % i for i in (8, 9, 10))
DUMMY = B10 + '__1__loop_closure'
BELL = 'servo_horn_v35_1__body1'

def rpy_from_R(R):
    # URDF rpy: R = Rz(y) Ry(p) Rx(r)
    p = np.arcsin(-np.clip(R[2, 0], -1, 1))
    if abs(np.cos(p)) > 1e-9:
        r = np.arctan2(R[2, 1], R[2, 2]); y = np.arctan2(R[1, 0], R[0, 0])
    else:
        r = np.arctan2(-R[1, 2], R[1, 1]); y = 0.0
    return r, p, y

def set_origin(elem, M):
    o = elem.find('origin')
    if o is None:
        o = ET.SubElement(elem, 'origin')
    o.set('xyz', ' '.join('%.9g' % v for v in M[:3, 3]))
    o.set('rpy', ' '.join('%.9g' % v for v in rpy_from_R(M[:3, :3])))

def main():
    tree = ET.parse(SRC); robot = tree.getroot()
    _, W = world(SRC)
    joints = {j.get('name'): j for j in robot.findall('joint')}
    links = {l.get('name'): l for l in robot.findall('link')}

    # --- sanity: Onshape's weld really is closed in the export pose
    err = np.linalg.norm(W[DUMMY][:3, 3] - W[B10][:3, 3])
    assert err < 1e-6, err

    # --- cut loop 1 at closing_left_horn
    robot.remove(joints['closing_left_horn'])
    # --- drop Onshape's loop-closure dummy (body9 -> dummy weld)
    robot.remove(joints['fastened_2_loop_closure']); robot.remove(links[DUMMY])

    # --- re-root: body10 (thigh side) --fixed--> body9 --dof_thigh--> bellcrank
    j = ET.SubElement(robot, 'joint', name='fastened_bearing9', type='fixed')
    set_origin(j, np.linalg.inv(W[B10]) @ W[B9])
    ET.SubElement(j, 'parent', link=B10); ET.SubElement(j, 'child', link=B9)

    # New bellcrank frame N := body9 frame (lies on the bellcrank pivot axis, z = axis).
    N = W[B9]; Ninv = np.linalg.inv(N); Wb = W[BELL]
    bell = links[BELL]
    for tag in ('inertial', 'visual', 'collision'):
        for e in bell.findall(tag):
            set_origin(e, Ninv @ Wb @ parse_origin(e.find('origin')))
    jr = joints['dof_right_horn']           # bellcrank -> pushrod, re-expressed in N
    set_origin(jr, Ninv @ Wb @ parse_origin(jr.find('origin')))

    jt = joints['dof_thigh']                # reverse: parent body9, child bellcrank
    jt.find('parent').set('link', B9); jt.find('child').set('link', BELL)
    set_origin(jt, np.eye(4)); jt.find('axis').set('xyz', '0 0 1')

    # Onshape prints right angles as 1.5708 / 3.14159 (~4e-6 rad off), which tilts the hinge axes
    # by microradians and makes the planar linkage very slightly non-planar (the closure
    # constraints then fight in z). Snap near-multiples of pi/2 in joint origins to exact values.
    for jn in robot.findall('joint'):
        o = jn.find('origin')
        if o is None: continue
        rpy = [float(v) for v in o.get('rpy', '0 0 0').split()]
        snapped = [round(v / (np.pi / 2)) * (np.pi / 2) if abs(v - round(v / (np.pi / 2)) * (np.pi / 2)) < 1e-4 else v
                   for v in rpy]
        o.set('rpy', ' '.join('%.12g' % v for v in snapped))

    # keep file tidy: joints after links
    for jn in [x for x in robot.findall('joint')]:
        robot.remove(jn); robot.append(jn)
    robot.insert(0, ET.Comment(' Derived from leg_assembly.urdf by scripts/make_cut_urdf.py: loop 1 cut at '
                               'closing_left_horn, loop 2 (pushrod->shin) open. Close both in the simulator. '))
    ET.indent(tree, '    ')
    tree.write(DST, xml_declaration=False, encoding='unicode')

    # --- verify: every link that existed keeps its world pose
    _, W2 = world(DST)
    worst = max(np.abs(W2[l][:3, 3] - W[l][:3, 3]).max() for l in W2 if l != BELL)
    # bellcrank frame moved by design; check its visual instead
    vb_old = Wb @ parse_origin(ET.parse(SRC).getroot().find(f"link[@name='{BELL}']/visual/origin"))
    vb_new = W2[BELL] @ parse_origin(bell.find('visual/origin'))
    print('wrote', os.path.normpath(DST))
    print('max position change of other links (from rpy snapping): %.2e m   bellcrank visual change: %.2e' % (worst, np.abs(vb_new - vb_old).max()))

if __name__ == '__main__':
    main()
