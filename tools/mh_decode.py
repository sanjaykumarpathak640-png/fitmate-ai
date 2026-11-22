import json,sys,numpy as np
import os
# path to the makehuman-data package (npm: makehuman-data), folder public/data/
SRC=os.environ.get('MH_DATA','makehuman-data/public/data/').rstrip('/')+'/'
def load_v3(path):
    d=json.load(open(path))
    V=np.array(d['vertices'],dtype=np.float32).reshape(-1,3)
    uvl=d['uvs'][0] if d['uvs'] else []
    UV=np.array(uvl,dtype=np.float32).reshape(-1,2) if uvl else np.zeros((0,2),np.float32)
    F=d['faces'];i=0;faces=[]
    while i<len(F):
        t=F[i];i+=1
        quad=t&1;hasMat=t&2;hasFUv=t&4;hasFVUv=t&8;hasFN=t&16;hasFVN=t&32;hasFC=t&64;hasFVC=t&128
        n=4 if quad else 3
        vs=F[i:i+n];i+=n
        mat=-1
        if hasMat: mat=F[i];i+=1
        if hasFUv: i+=1
        uvs=None
        if hasFVUv: uvs=F[i:i+n];i+=n
        if hasFN: i+=1
        if hasFVN: i+=n
        if hasFC: i+=1
        if hasFVC: i+=n
        faces.append((vs,uvs,mat))
    return d,V,UV,faces
if __name__=='__main__':
    d,V,UV,faces=load_v3(SRC+'models/human_full_size.json')
    mats=[m.get('DbgName') for m in d['materials']]
    from collections import Counter
    c=Counter(f[2] for f in faces);print('faces',len(faces),'by mat (top)',[(mats[k],n) for k,n in c.most_common(6)])
    body=[f for f in faces if mats[f[2]]=='body'];bv=sorted(set(v for f in body for v in f[0]))
    print('body faces',len(body),'body verts',len(bv),'quads',sum(len(f[0])==4 for f in body),'uv faces',sum(f[1] is not None for f in body))
    B=V[bv];print('body bbox',B.min(0),B.max(0))
    bones=d['bones'];print('bones',len(bones),[b['name'] for b in bones[:40]])
    # arm: find bone positions
    nm={b['name']:i for i,b in enumerate(bones)}
    for k in ['upperarm01.L','lowerarm01.L','wrist.L','upperleg01.L','lowerleg01.L','foot.L','neck01','head','spine05','root','clavicle.L','shoulder01.L']:
        if k in nm: print(k,bones[nm[k]]['pos'],'parent',bones[nm[k]]['parent'])
