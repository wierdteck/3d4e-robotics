import xml.etree.ElementTree as ET, numpy as np
from math import cos,sin
def T(xyz,rpy):
    x,y,z=xyz; R_,P,Y=rpy
    Rx=np.array([[1,0,0],[0,cos(R_),-sin(R_)],[0,sin(R_),cos(R_)]])
    Ry=np.array([[cos(P),0,sin(P)],[0,1,0],[-sin(P),0,cos(P)]])
    Rz=np.array([[cos(Y),-sin(Y),0],[sin(Y),cos(Y),0],[0,0,1]])
    M=np.eye(4); M[:3,:3]=Rz@Ry@Rx; M[:3,3]=[x,y,z]; return M
def parse_origin(e):
    if e is None: return np.eye(4)
    return T([float(v) for v in e.get('xyz','0 0 0').split()],[float(v) for v in e.get('rpy','0 0 0').split()])
def world(path):
    r=ET.parse(path).getroot(); J={}
    for j in r.findall('joint'):
        J[j.find('child').get('link')]=(j.find('parent').get('link'),parse_origin(j.find('origin')))
    W={}
    def fk(l):
        if l not in W: W[l]=np.eye(4) if l not in J else fk(J[l][0])@J[l][1]
        return W[l]
    for l in r.findall('link'): fk(l.get('name'))
    return r,W
