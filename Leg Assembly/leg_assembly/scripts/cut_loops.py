"""Rebuild the Onshape-exported leg URDF as a spanning tree with ONE revolute
cut per kinematic loop, plus a closure frame on each side of every cut.
The simulator then closes each loop with a point (connect / point-to-point)
constraint between the two frames of a pair.

Loop 1 (knee-servo four-bar): servo B horn -> hip_link_2 (coupler) -> servo_horn_v35 (bellcrank)
        Cut at the coupler<->bellcrank pin (the old `closing_left_horn` joint).
        The bellcrank now hangs off thigh_link_1 on its bearing (joint `dof_thigh`, coaxial with the thigh servo).
Loop 2 (pushrod): bellcrank -> thigh_link_2 (pushrod) -> shin rear crank.
        Cut at the pushrod<->shin pin (never mated in Onshape, so it was missing from the export).

Usage:  python cut_loops.py ../urdf/leg_assembly.urdf ../urdf/leg_assembly_cut.urdf
Writes the URDF plus <dst>.closures.json describing the constraint pairs.
"""
import sys, json, xml.etree.ElementTree as ET
import numpy as np
from math import cos, sin, atan2, asin

THIGH   = "thigh_link_1_v59_1__body2"
BELL    = "servo_horn_v35_1__body1"
COUPLER = "hip_link_2_v8_1__body1"
PUSHROD = "thigh_link_2_v10_1__body1"
SHIN    = "shin_ankle_link_v33_1__body5"
BEAR9   = "4668k144_permanently_lubricated_stainless_steel_ball_bearing_v1_1__body9"
OLD_LC  = "4668k144_permanently_lubricated_stainless_steel_ball_bearing_v1_1__body10__1__loop_closure"

# Pin centres in WORLD coords at q=0 (metres), measured by circle-fitting the
# M3 holes (r = 1.6-1.7 mm) in the STL meshes. Both frames of a pair share one
# world z so the point constraint's z row is redundant, not fighting the hinges.
# Loop 1 pin comes straight from the old closing_left_horn joint (exact).
PUSHROD_FAR_PIN = np.array([0.090266, -0.040422])   # hole in thigh_link_2 far end
SHIN_CRANK_PIN  = np.array([0.089719, -0.040474])   # hole in shin_ankle_link body6 (rigid to shin)
LOOP2_Z         = 0.014407                          # pushrod mid-plane

def T(xyz, rpy):
    r, p, y = rpy
    Rx = np.array([[1,0,0],[0,cos(r),-sin(r)],[0,sin(r),cos(r)]])
    Ry = np.array([[cos(p),0,sin(p)],[0,1,0],[-sin(p),0,cos(p)]])
    Rz = np.array([[cos(y),-sin(y),0],[sin(y),cos(y),0],[0,0,1]])
    M = np.eye(4); M[:3,:3] = Rz@Ry@Rx; M[:3,3] = xyz; return M

def rpy(R):
    p = asin(max(-1.0, min(1.0, -R[2,0])))
    return atan2(R[2,1], R[2,2]), p, atan2(R[1,0], R[0,0])

f3 = lambda s: [float(v) for v in s.split()]
fmt = lambda v: " ".join(f"{x:.9g}" for x in v)

def read_origin(e):
    o = e.find("origin")
    return np.eye(4) if o is None else T(f3(o.get("xyz", "0 0 0")), f3(o.get("rpy", "0 0 0")))

def write_origin(e, M):
    o = e.find("origin")
    if o is None:
        o = ET.Element("origin"); e.insert(0, o)
    o.set("xyz", fmt(M[:3,3])); o.set("rpy", fmt(rpy(M[:3,:3])))

def main(src, dst):
    tree = ET.parse(src); robot = tree.getroot()
    links  = {l.get("name"): l for l in robot.findall("link")}
    joints = {j.get("name"): j for j in robot.findall("joint")}
    by_child = {j.find("child").get("link"): j for j in joints.values()}

    W = {}
    def fk(l):
        if l not in W:
            j = by_child.get(l)
            W[l] = np.eye(4) if j is None else fk(j.find("parent").get("link")) @ read_origin(j)
        return W[l]
    for l in links: fk(l)

    def drop_joint(name):
        robot.remove(joints.pop(name))

    # 1) remove Onshape's weld-style loop closure (bearing9 <-> bearing10 copy)
    drop_joint("fastened_2_loop_closure"); robot.remove(links.pop(OLD_LC))

    # 2) cut loop 1: coupler -> bellcrank revolute becomes a constraint
    loop1_pin_w = W[BELL][:3,3].copy()          # child frame of closing_left_horn = the pin
    drop_joint("closing_left_horn")

    # 3) re-root the bellcrank on the thigh via its bearing (old dof_thigh, reversed)
    j = joints["dof_thigh"]
    a_b9 = np.array(f3(j.find("axis").get("xyz")))   # old axis: bearing rel. bellcrank, in bearing frame
    F = W[BEAR9].copy()                               # new bellcrank frame: on the bearing axis
    lk = links[BELL]
    for tag in ("inertial", "visual", "collision"):
        for e in lk.findall(tag):
            write_origin(e, np.linalg.inv(F) @ W[BELL] @ read_origin(e))
    W[BELL] = F
    j.find("parent").set("link", THIGH); j.find("child").set("link", BELL)
    write_origin(j, np.linalg.inv(W[THIGH]) @ F)
    j.find("axis").set("xyz", fmt(-a_b9))             # reversed parent/child -> same sign convention

    # bearing9 is physically on the thigh side: bolt it to the thigh
    jb = ET.SubElement(robot, "joint", name="fastened_bearing9_to_thigh", type="fixed")
    ET.SubElement(jb, "origin"); ET.SubElement(jb, "parent", link=THIGH); ET.SubElement(jb, "child", link=BEAR9)
    write_origin(jb, np.linalg.inv(W[THIGH]) @ W[BEAR9])

    # 4) pushrod now hangs off the relocated bellcrank frame
    write_origin(joints["dof_right_horn"], np.linalg.inv(W[BELL]) @ W[PUSHROD])

    # 5) closure frames (tiny-mass links, world-aligned, z = hinge axis)
    def frame(name, parent, p_world):
        Mw = np.eye(4); Mw[:3,3] = p_world
        l = ET.SubElement(robot, "link", name=name)
        ine = ET.SubElement(l, "inertial")
        ET.SubElement(ine, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(ine, "mass", value="1e-09")
        ET.SubElement(ine, "inertia", ixx="1e-14", ixy="0", ixz="0", iyy="1e-14", iyz="0", izz="1e-14")
        jj = ET.SubElement(robot, "joint", name=name + "_joint", type="fixed")
        ET.SubElement(jj, "origin"); ET.SubElement(jj, "parent", link=parent); ET.SubElement(jj, "child", link=name)
        Mloc = np.linalg.inv(W[parent]) @ Mw
        write_origin(jj, Mloc)
        return Mloc[:3,3]

    loops = []
    pA = frame("closing_left_horn_coupler", COUPLER, loop1_pin_w)
    pB = frame("closing_left_horn_bellcrank", BELL, loop1_pin_w)
    loops.append(dict(name="closing_left_horn", type="revolute_z_as_point",
                      a=dict(link=COUPLER, frame="closing_left_horn_coupler", pos=pA.tolist()),
                      b=dict(link=BELL, frame="closing_left_horn_bellcrank", pos=pB.tolist()),
                      gap_at_q0_mm=0.0))
    pw_a = np.r_[PUSHROD_FAR_PIN, LOOP2_Z]; pw_b = np.r_[SHIN_CRANK_PIN, LOOP2_Z]
    pA = frame("closing_pushrod_rod", PUSHROD, pw_a)
    pB = frame("closing_pushrod_shin", SHIN, pw_b)
    loops.append(dict(name="closing_pushrod", type="revolute_z_as_point",
                      a=dict(link=PUSHROD, frame="closing_pushrod_rod", pos=pA.tolist()),
                      b=dict(link=SHIN, frame="closing_pushrod_shin", pos=pB.tolist()),
                      gap_at_q0_mm=float(np.linalg.norm(pw_a - pw_b) * 1000)))

    ET.indent(tree, "    ")
    robot.insert(0, ET.Comment(" Loop-cut version generated by scripts/cut_loops.py from leg_assembly.urdf. "
                                "Close loops in the simulator using <this file>.closures.json "))
    tree.write(dst, xml_declaration=False, encoding="unicode")
    json.dump(dict(actuated=["dof_thigh_motor", "dof_shin_motor"],
                   passive=["dof_servo_to_hip", "dof_thigh", "dof_right_horn", "dof_knee"],
                   closures=loops), open(dst + ".closures.json", "w"), indent=2)
    print("wrote", dst, "and", dst + ".closures.json")
    for c in loops: print(c["name"], c["a"]["link"], "<->", c["b"]["link"], f"gap at q=0: {c['gap_at_q0_mm']:.3f} mm")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
