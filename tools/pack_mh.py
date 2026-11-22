"""Pack the MakeHuman (CC0 assets) base mesh, selected targets and proxies for FitMate's realistic avatar.
Output: frontend/assets/mh/mh.bin + mh.json + tex/*
Usage: MH_DATA=path/to/makehuman-data/public/data python tools/pack_mh.py
"""
import json, os, glob, numpy as np
from PIL import Image
from mh_decode import load_v3, SRC

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend', 'assets', 'mh') + '/'
os.makedirs(OUT + 'tex', exist_ok=True)
NV = 19158

blob = bytearray(); manifest = {'blocks': {}}
def put(name, arr):
    global blob
    arr = np.ascontiguousarray(arr)
    while len(blob) % 8: blob += b'\0'
    manifest['blocks'][name] = [len(blob), str(arr.dtype), int(arr.size)]
    blob += arr.tobytes()

# ---------- base mesh ----------
d, V, UV, faces = load_v3(SRC + 'models/human_full_size.json')
mats = [m.get('DbgName') for m in d['materials']]
assert V.shape[0] == NV
put('base', V.astype(np.float32).ravel())

def render_mesh(fs, uvarr):
    key = {}; rv = []; ruv = []; tris = []
    for vs, uvs, mat in fs:
        ids = []
        for k, vi in enumerate(vs):
            ui = uvs[k] if uvs is not None else -1
            kk = (vi, ui)
            if kk not in key:
                key[kk] = len(rv); rv.append(vi); ruv.append(uvarr[ui] if ui >= 0 else (0, 0))
            ids.append(key[kk])
        if len(ids) == 4: tris += [ids[0], ids[1], ids[2], ids[0], ids[2], ids[3]]
        else: tris += ids
    return np.array(rv), np.array(ruv, np.float32), np.array(tris)

body = [f for f in faces if mats[f[2]] == 'body']
rv, ruv, tris = render_mesh(body, UV)
put('body_vi', rv.astype(np.uint16)); put('body_uv', ruv.ravel()); put('body_tri', tris.astype(np.uint32))
print('body render verts', len(rv), 'tris', len(tris) // 3)

# ---------- skeleton-derived per-vertex attributes ----------
bones = d['bones']; nb = len(bones)
absb = np.zeros((nb, 3))
for i, b in enumerate(bones):
    absb[i] = np.array(b['pos']) + (absb[b['parent']] if b['parent'] >= 0 else 0)
nm = {b['name']: i for i, b in enumerate(bones)}
J = lambda n: absb[nm[n + '____head']]
si = np.array(d['skinIndices']).reshape(-1, 4); sw = np.array(d['skinWeights']).reshape(-1, 4)
bname = [b['name'].replace('____head', '') for b in bones]
def group(n):
    s = n.split('.')[0].rstrip('0123456789').rstrip('-')
    if s in ('upperarm',): return 3
    if s in ('lowerarm',): return 4
    if s.startswith('wrist') or s.startswith('metacarpal') or s.startswith('finger') or s.startswith('thumb'): return 5
    if s in ('upperleg',): return 6
    if s in ('lowerleg',): return 7
    if s.startswith('foot') or s.startswith('toe'): return 8
    if s.startswith('neck'): return 1
    if s in ('spine', 'root', 'pelvis', 'breast', 'clavicle', 'shoulder'): return 2
    return 0  # head, jaw, face muscles, eyes, tongue
G = np.array([group(n) for n in bname])
gv = G[si]  # (NV,4)
part = np.zeros(NV, np.uint8); armw = np.zeros(NV); legw = np.zeros(NV)
for v in range(NV):
    acc = np.zeros(9)
    for k in range(4): acc[gv[v, k]] += sw[v, k]
    part[v] = acc.argmax()
    armw[v] = acc[3] + acc[4] + acc[5] + 0.35 * sum(sw[v, k] for k in range(4) if bname[si[v, k]].startswith('shoulder'))
    legw[v] = acc[6] + acc[7] + acc[8]
armw = np.clip(armw, 0, 1); legw = np.clip(legw, 0, 1)
def seg_param(P, a, b, c):
    # 0 at a, 1 at b, 2 at c along polyline
    out = np.zeros(len(P))
    for i, p in enumerate(P):
        best = None
        for s0, (u, w) in enumerate(((a, b), (b, c))):
            d_ = w - u; t = np.clip(np.dot(p - u, d_) / np.dot(d_, d_), 0, 1)
            dist = np.linalg.norm(u + t * d_ - p)
            if best is None or dist < best[0]: best = (dist, s0 + t)
        out[i] = best[1]
    return out
# joints from part-boundary rings (robust: centroid of the ring around each limb)
nbr = [set() for _ in range(NV)]
for vs, uvs, mat in body:
    for i in range(len(vs)):
        a_, b_ = vs[i], vs[(i + 1) % len(vs)]; nbr[a_].add(b_); nbr[b_].add(a_)
def ring(pa, pb, side):
    out = [v for v in range(NV) if part[v] == pa and nbr[v] and any(part[u] == pb for u in nbr[v]) and (side == 0 or np.sign(V[v, 0]) == side)]
    return out
RING = {}
for sd, sx in (('L', 1), ('R', -1)):
    RING['sh' + sd] = ring(3, 2, sx); RING['el' + sd] = ring(4, 3, sx); RING['wr' + sd] = ring(5, 4, sx)
    RING['hip' + sd] = ring(6, 2, sx); RING['kn' + sd] = ring(7, 6, sx); RING['an' + sd] = ring(8, 7, sx)
RING['neck'] = ring(1, 2, 0); RING['head'] = ring(0, 1, 0)
JR = {k: V[v].mean(0) for k, v in RING.items()}
print({k: (len(v), JR[k].round(2).tolist()) for k, v in RING.items()})
limb = np.zeros(NV)
for side, sx in (('L', 1), ('R', -1)):
    sh, el, wr = JR['sh' + side], JR['el' + side], JR['wr' + side]
    hp, kn, an = JR['hip' + side], JR['kn' + side], JR['an' + side]
    sel = np.where((part >= 3) & (part <= 5) & (np.sign(V[:, 0]) == sx))[0]
    limb[sel] = seg_param(V[sel], sh, el, wr)
    sel = np.where((part >= 6) & (part <= 8) & (np.sign(V[:, 0]) == sx))[0]
    limb[sel] = seg_param(V[sel], hp, kn, an)
put('part', part); put('limb', np.clip(limb * 100, 0, 255).astype(np.uint8))
put('armw', (armw * 255).astype(np.uint8)); put('legw', (legw * 255).astype(np.uint8))
# joint anchors: body vertices near each joint (runtime pivot = mean of these)
bodyset = np.array(sorted(set(rv.tolist())))
anchors = {k: [int(x) for x in v] for k, v in RING.items()}
manifest['anchors'] = anchors
manifest['top_vi'] = int(bodyset[np.argmax(V[bodyset, 1])]); manifest['bot_vi'] = int(bodyset[np.argmin(V[bodyset, 1])])

# ---------- measurement loops (torso slices, base mesh) ----------
def hull_perim(pts):
    pts = sorted(set(map(tuple, np.round(pts, 5))))
    if len(pts) < 3: return 0
    def cross(o, a, b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lo = []; hi = []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0: hi.pop()
        hi.append(p)
    h = lo[:-1] + hi[:-1]
    return sum(np.hypot(h[i][0]-h[i-1][0], h[i][1]-h[i-1][1]) for i in range(len(h)))
torso = bodyset[part[bodyset] == 2]
ys = np.arange(V[torso, 1].min() + .3, V[torso, 1].max() - .3, .05)
prof = []
for y in ys:
    s = torso[np.abs(V[torso, 1] - y) < .06]
    prof.append(hull_perim(V[s][:, [0, 2]]) if len(s) > 4 else 0)
prof = np.array(prof)
yb = J('breast.L')[1]; yh = J('upperleg01.L')[1]
ib = np.argmin(np.abs(ys - yb));
iw = np.argmin(np.where((ys > yh + .8) & (ys < yb - .8), prof, 1e9))
ih = np.argmax(np.where((ys > yh - 1.2) & (ys < yh + .9), prof, 0))
loops = {}
low = bodyset[(part[bodyset] == 2) | (part[bodyset] == 6)]
ycr = V[bodyset[(part[bodyset] == 2) & (np.abs(V[bodyset, 0]) < .15)], 1].min(); hy = np.arange(ycr + .25, ycr + 1.4, .05); hp_ = [hull_perim(V[low[np.abs(V[low, 1] - y) < .07]][:, [0, 2]]) for y in hy]
yhips = hy[int(np.argmax(hp_))]
loops['bust'] = torso[np.abs(V[torso, 1] - ys[ib]) < .08].tolist()
loops['waist'] = torso[np.abs(V[torso, 1] - ys[iw]) < .14].tolist()
loops['hips'] = low[np.abs(V[low, 1] - yhips) < .1].tolist()
manifest['land'] = {'bust': float(ys[ib]), 'waist': float(ys[iw]), 'hips': float(yhips)}
ua = bodyset[(part[bodyset] == 3) & (V[bodyset, 0] > 0)]
uy = (J('upperarm01.L')[1] + J('lowerarm01.L')[1]) / 2
loops['upperarm'] = ua[np.abs(V[ua, 1] - uy) < .07].tolist()
th = bodyset[(part[bodyset] == 6) & (V[bodyset, 0] > 0)]
ty = J('upperleg01.L')[1] - .9
loops['thigh'] = th[np.abs(V[th, 1] - ty) < .07].tolist()
manifest['loops'] = loops
print('landmarks', manifest['land'], {k: len(v) for k, v in loops.items()})

# ---------- beard / stubble region weight ----------
def sm(e0, e1, x): t = np.clip((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t)
ax = np.abs(V[:, 0]); y = V[:, 1]; z = V[:, 2]
yup = np.where(ax < .3, 6.80 + ax * .1, 6.83 + (ax - .3) * 1.05)
bw = sm(-.04, .04, yup - y) * sm(6.0, 6.14, y) * sm(.62, .8, z + (y - 6.4) * .35) * (ax < .86)
tl0 = sorted(json.load(open(SRC + '../../src/json/targets/target-list.json'))['targets'].keys())
TB0 = np.memmap(SRC + 'targets/targets.bin', dtype=np.int16, mode='r')
def mag(path):
    i = tl0.index('data/targets/' + path + '.target'); D = np.array(TB0[i * NV * 3:(i + 1) * NV * 3]).reshape(NV, 3).astype(np.float32); return np.abs(D).sum(1)
lip = np.maximum(mag('mouth/mouth-upperlip-volume-inflate'), mag('mouth/mouth-lowerlip-volume-inflate'))
lipw = np.clip(lip / (lip.max() * .12), 0, 1)
ear = mag('ears/l-ear-size-big') + mag('ears/r-ear-size-big'); earw = np.clip(ear / (ear.max() * .1), 0, 1)
bw = bw * (1 - lipw) * (1 - earw) * ((part == 0) | (part == 1))
put('beard', (np.clip(bw, 0, 1) * 255).astype(np.uint8))
print('beard verts', int((bw > .05).sum()))

# ---------- targets ----------
tl = sorted(json.load(open(SRC + '../../src/json/targets/target-list.json'))['targets'].keys())
TB = np.memmap(SRC + 'targets/targets.bin', dtype=np.int16, mode='r')
def tgt(path):
    i = tl.index('data/targets/' + path + '.target'); return np.array(TB[i * NV * 3:(i + 1) * NV * 3]).reshape(NV, 3)
want = []
for g in ('female', 'male'):
    for mu in ('minmuscle', 'averagemuscle', 'maxmuscle'):
        for we in ('minweight', 'averageweight', 'maxweight'):
            want.append(f'macrodetails/universal-{g}-young-{mu}-{we}')
for r in ('african', 'asian', 'caucasian'):
    for g in ('female', 'male'): want.append(f'macrodetails/{r}-{g}-young')
for m in ('bust', 'waist', 'hips', 'shoulder', 'upperarm', 'thighcirc', 'neckcirc', 'underbust'):
    want += [f'measure/measure-{m}-decrease', f'measure/measure-{m}-increase']
want += ['breast/female-young-averagemuscle-averageweight-maxcup-averagefirmness', 'breast/female-young-averagemuscle-averageweight-mincup-averagefirmness']
face = ['head/head-' + x for x in ('oval', 'round', 'square', 'rectangular', 'triangular', 'invertedtriangular', 'diamond', 'fat', 'skinny', 'scale-horiz-less', 'scale-horiz-more', 'scale-vert-less', 'scale-vert-more', 'age-less', 'age-more')]
face += ['nose/nose-' + x for x in ('scale-horiz-incr', 'scale-horiz-decr', 'scale-vert-incr', 'scale-vert-decr', 'hump-morehump', 'hump-lesshump', 'point-up', 'point-down', 'nostril-width-max', 'nostril-width-min', 'volume-potato', 'volume-point', 'greek-moregreek', 'curve-convex', 'curve-concave', 'width1-max', 'width1-min', 'flaring-incr', 'scale-depth-incr', 'scale-depth-decr')]
face += ['mouth/mouth-' + x for x in ('upperlip-volume-inflate', 'upperlip-volume-deflate', 'lowerlip-volume-inflate', 'lowerlip-volume-deflate', 'scale-horiz-incr', 'scale-horiz-decr', 'angles-up', 'angles-down', 'cupidsbow-incr', 'dimples-in')]
for s in ('l', 'r'):
    face += [f'eyes/{s}-eye-' + x for x in ('size-big', 'size-small', 'epicanthus-in', 'epicanthus-out', 'corner1-up', 'corner1-down', 'height1-max', 'height1-min')]
    face += [f'cheek/{s}-cheek-' + x for x in ('bones-out', 'bones-in', 'volume-inflate', 'volume-deflate')]
    face += [f'ears/{s}-ear-' + x for x in ('size-big', 'size-small', 'flap-out')]
face += ['chin/chin-' + x for x in ('prominent-more', 'prominent-less', 'width-max', 'width-min', 'triangle', 'jaw-drop-more', 'jaw-drop-less', 'height-max', 'height-min')]
face += ['forehead/forehead-' + x for x in ('nubian-more', 'nubian-less', 'scale-vert-more', 'scale-vert-less')]
face += ['eyebrows/eyebrows-angle-up', 'eyebrows/eyebrows-angle-down', 'neck/neck-scale-horiz-more', 'neck/neck-scale-horiz-less']
want += face
for r in ('african', 'asian', 'caucasian'):
    want += [f'expression/units/{r}/' + x for x in ('mouth-corner-puller', 'eye-left-closure', 'eye-right-closure')]
tmeta = []
for w in want:
    D = tgt(w); nz = np.where(np.abs(D).sum(1) > 0)[0]
    name = w.split('/')[-1] if not w.startswith('expression') else 'expr-' + w.split('/')[-2] + '-' + w.split('/')[-1]
    put('t_' + name + '_i', nz.astype(np.uint16)); put('t_' + name + '_d', D[nz].astype(np.int16).ravel())
    tmeta.append([name, int(len(nz))])
manifest['targets'] = tmeta; manifest['tscale'] = 1e-3
print('targets', len(tmeta), 'nonzero verts total', sum(t[1] for t in tmeta))

# ---------- proxies ----------
PX = SRC + 'proxies/'
def gray_alpha(src, dst, size=512, keep_color=False):
    im = Image.open(src).convert('RGBA')
    if im.size[0] > size: im = im.resize((size, size), Image.LANCZOS)
    if not keep_color:
        r, g, b, a = im.split(); l = np.asarray(Image.merge('RGB', (r, g, b)).convert('L'), np.float32)
        al = np.asarray(a, np.float32); m = al > 128
        if m.sum() > 100:
            lo, hi = np.percentile(l[m], 3), np.percentile(l[m], 97)
            l = np.clip(.42 + (l - lo) / max(1, hi - lo) * .58, 0, 1) * 255
        l = Image.fromarray(l.astype(np.uint8))
        im = Image.merge('RGBA', (l, l, l, a))
    im.save(dst, optimize=True)
def opaque(src, dst, size=512, q=86):
    im = Image.open(src).convert('RGB')
    if im.size[0] > size: im = im.resize((size, size), Image.LANCZOS)
    im.save(dst, quality=q, optimize=True)
proxies = {}
def add_proxy(key, path, tex_mode):
    j = json.load(open(glob.glob(PX + path + '/*.json')[0]))
    nv = len(j['vertices']) // 3
    ref = np.array(j['ref_vIdxs'], np.uint16).ravel(); w = np.array(j['weights'], np.float32).ravel(); off = np.array(j['offsets'], np.float32).ravel()
    assert ref.size == nv * 3
    uvl = np.array(j['uvs'][0], np.float32).reshape(-1, 2) if j['uvs'] and j['uvs'][0] else np.zeros((0, 2), np.float32)
    # decode faces with the same v3 decoder
    tmp = '/tmp/_px.json'; json.dump({'vertices': j['vertices'], 'uvs': j['uvs'], 'faces': j['faces'], 'materials': j['materials']}, open(tmp, 'w'))
    _, PV, PUV, pf = load_v3(tmp)
    prv, pruv, ptri = render_mesh(pf, PUV)
    put(f'p_{key}_ref', ref); put(f'p_{key}_w', w); put(f'p_{key}_off', off)
    put(f'p_{key}_vi', prv.astype(np.uint16)); put(f'p_{key}_uv', pruv.ravel()); put(f'p_{key}_tri', ptri.astype(np.uint32))
    texs = sorted(glob.glob(PX + path + '/textures/*'))
    lic = j['metadata'].get('license', {})
    proxies[key] = {'nv': nv, 'author': lic.get('author'), 'src': path}
    return texs
HAIR = {'short01': 'hair/short01', 'short02': 'hair/short02', 'short03': 'hair/short03', 'short04': 'hair/short04', 'bob01': 'hair/bob01', 'bob02': 'hair/bob02',
        'long01': 'hair/long01', 'ponytail01': 'hair/ponytail01', 'braid01': 'hair/Braid01', 'afro01': 'hair/afro01'}
for k, p in HAIR.items():
    t = [x for x in add_proxy(k, p, 'hair') if 'diffuse' in x.lower() or x.endswith('.png')][0]
    gray_alpha(t, OUT + f'tex/{k}.png', 512); proxies[k]['tex'] = f'tex/{k}.png'
for i in (1, 2, 3, 5, 7, 8, 9, 10, 12):
    k = f'eyebrow{i:03d}'; t = add_proxy(k, f'eyebrows/{k}', 'brow')[0]; gray_alpha(t, OUT + f'tex/{k}.png', 256); proxies[k]['tex'] = f'tex/{k}.png'
for k, p in (('lashes01', 'eyelashes/Eyelashes01'), ('lashes02', 'eyelashes/eyelashes02')):
    t = add_proxy(k, p, 'lash')[0]; gray_alpha(t, OUT + f'tex/{k}.png', 256); proxies[k]['tex'] = f'tex/{k}.png'
add_proxy('eyes', 'eyes/HighPolyEyes', 'eye')
eyecols = {}
for c in ('brown', 'brownlight', 'green', 'blue', 'grey', 'bluegreen'):
    im = Image.open(PX + f'eyes/HighPolyEyes/textures/{c}_eye.png').convert('RGBA').resize((256, 256), Image.LANCZOS)
    A = np.asarray(im, np.float32); rgb = A[..., :3]; lum = rgb.mean(2, keepdims=True)
    mx = rgb.max(2, keepdims=True); mn = rgb.min(2, keepdims=True); sat = (mx - mn) / np.maximum(mx, 1)
    scl = np.clip((lum - 120) / 60, 0, 1) * (sat < .4)  # sclera: desaturate and whiten
    rgb = rgb * (1 - scl * .75) + (lum * np.array([1.0, .99, .97]) * 1.06) * scl * .75
    iris = ((sat > .42) & (lum < 140)).astype(np.float32)
    if c == 'brown': rgb = rgb * (1 - iris) + iris * (lum / max(1, float(lum[iris[..., 0] > .5].mean() if (iris > .5).any() else 1)) * np.array([66, 42, 26.]))
    if c == 'brownlight': rgb = rgb * (1 - iris) + iris * (rgb * np.array([.85, .95, .9]))
    A[..., :3] = np.clip(rgb, 0, 255); Image.fromarray(A.astype(np.uint8)).save(OUT + f'tex/eye_{c}.png', optimize=True); eyecols[c] = f'tex/eye_{c}.png'
manifest['eyes'] = eyecols
for k in ('shoes05', 'shoes06', 'shoes03', 'shoes01'):
    t = [x for x in add_proxy(k, f'clothes/{k}', 'shoe') if 'diffuse' in x][0]; opaque(t, OUT + f'tex/{k}.jpg', 512); proxies[k]['tex'] = f'tex/{k}.jpg'
manifest['proxies'] = proxies
skins = {}
for r, tone in (('caucasian', 'lightskinned'), ('asian', 'lightskinned'), ('african', 'darkskinned')):
    for g in ('female', 'male'):
        t = glob.glob(SRC + f'skins/young_{r}_{g}/textures/*diffuse*.png')
        t = sorted(t, key=lambda x: ('3' not in x and r == 'asian', 'diffuse2' in x))[0]
        opaque(t, OUT + f'tex/skin_{r}_{g}.jpg', 512, 90); skins[f'{r}_{g}'] = f'tex/skin_{r}_{g}.jpg'
manifest['skins'] = skins
manifest['license'] = 'MakeHuman base mesh, targets, proxies and textures by the MakeHuman team (Manuel Bastioni, Jonas Hauquier, Thomas Larsson and contributors), bundled MakeHuman assets released under CC0 1.0. https://www.makehumancommunity.org'
open(OUT + 'mh.bin', 'wb').write(blob)
import base64; open(OUT + 'mh.b64.txt', 'w').write(base64.b64encode(bytes(blob)).decode())  # for hosts that do not serve .bin
json.dump(manifest, open(OUT + 'mh.json', 'w'))
print('bin MB', round(len(blob) / 1e6, 2), 'json KB', round(os.path.getsize(OUT + 'mh.json') / 1e3))
