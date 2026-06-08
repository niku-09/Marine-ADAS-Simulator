"""
╔══════════════════════════════════════════════════════════════════╗
║         MARINE ADAS SIMULATOR  —  SENSOR FUSION EDITION         ║
║──────────────────────────────────────────────────────────────────║
║  ↑ / ↓   Throttle / Brake       ← / →   Port / Starboard        ║
║  P       Autopilot  (set waypoint first via double-click)        ║
║  G = GPS off    L = LiDAR 30%    I = IMU spike                   ║
║  [ / ] = Sea state  Beaufort 0–5                                 ║
╚══════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon, Rectangle, Circle, Ellipse, Arc
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Button
from collections import deque

try:
    from scipy.ndimage import gaussian_filter
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    from PIL import Image, ImageDraw
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

# ═══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════
MAP_W = MAP_H = 30
DT        = 0.05
SIM_TIME  = 600
LDR_RANGE = 10.0
LDR_BEAMS = 360
MAX_SPD   = 3.0
MAX_OMG   = 2.5
SURGE_TAU = 4.0    # surge time constant (s) — forward inertia/drag
YAW_TAU   = 2.0    # yaw damping time constant (s)
BOAT_L    = 2.5
BOAT_W    = 1.2
COLL_D    = 2.2
GPS_EVERY = 20

CPA_RED   = 1.8
CPA_AMBER = 3.5
TCPA_WARN = 40.0

PPI_N = 120
PPI_R = PPI_N // 2
PPI_SCALE = PPI_R / LDR_RANGE
PPI_DECAY = 0.955

LDR_ANGLES = np.linspace(-np.pi, np.pi, LDR_BEAMS, endpoint=False)
_STEPS     = np.arange(0.15, LDR_RANGE, 0.15)

_ii, _jj = np.mgrid[0:PPI_N, 0:PPI_N]
PPI_MASK = np.hypot(_ii - PPI_R, _jj - PPI_R) > (PPI_R - 1)

CHI2_95 = 5.991

# ═══════════════════════════════════════════════════════════════════
#  PALETTE  — dark nav view + light analytical panels
# ═══════════════════════════════════════════════════════════════════
# Navigation view (left): deep navy
DARK_BG   = '#060D14'
PANEL_BG  = '#091520'
BORDER_DIM= '#162840'
BORDER_HI = '#1E4868'
TEXT_DIM  = '#28486A'
TEXT_MID  = '#4878A0'
TEXT_HI   = '#80B8D0'
PHOSPHOR  = '#00C0A8'
PHOSPHOR2 = '#009888'
CYAN_HI   = '#40D0F0'
AMBER     = '#E8A020'
RED_HI    = '#D83050'
ORANGE_HI = '#D86820'
MAGENTA_P = '#B040A8'
WHITE_DIM = '#7898A8'
GRID_CLR  = '#0C1E2E'

THREAT_CLR = {'green': PHOSPHOR, 'amber': AMBER, 'red': RED_HI}

# Analytical panels (right): light clinical theme
LP_BG   = '#EDF1F6'   # off-white blue-gray
LP_TEXT = '#1C2A3A'   # near-black for readability
LP_DIM  = '#506888'   # secondary text
LP_GRID = '#C0CCD8'   # grid lines / separators
LP_EDGE = '#7A90A8'   # panel border
LP_ACC  = '#1060B8'   # active/ok accent (blue)

_ppi_cmap = LinearSegmentedColormap.from_list(
    'ppi', [(0,'#000810'), (0.08,'#001828'), (0.40,'#003848'), (1.0,'#00C0A8')]
)
_ocean_cmap = LinearSegmentedColormap.from_list(
    'ocean_retro',
    [(0,'#010A12'), (0.4,'#031825'), (0.75,'#052030'), (1.0,'#082838')]
)
# SLAM display colourmap (value = 0.0→free  0.5→unknown  1.0→occupied)
# Unknown cells (0.5) render as clear medium gray — matches ROS OccupancyGrid convention
_slam_cmap = LinearSegmentedColormap.from_list(
    'slam_industry',
    [(0.00,'#FAFAFA'),   # free: white
     (0.38,'#EEF0F2'),   # mostly free: off-white
     (0.46,'#B8C0CA'),   # transition
     (0.50,'#9AA4AE'),   # unknown: medium gray
     (0.54,'#7880A0'),   # transition
     (0.72,'#303840'),   # occupied: very dark
     (1.00,'#080C10')]   # definite obstacle: near-black
)

# ═══════════════════════════════════════════════════════════════════
#  VESSEL SPRITE RENDERER
#  128×128 RGBA canvases. Bow points +x (east). Port is +y (up).
#  PIL y goes down → _w2p negates y so port maps to lower row index
#  (= higher world y when displayed origin='upper'). PIL rotate(deg)
#  is CCW which matches CCW game angles. No expand → fixed 128px canvas.
# ═══════════════════════════════════════════════════════════════════
SPRITE_SZ    = 128
SPRITE_WORLD = 4.0
_SPX         = SPRITE_SZ / SPRITE_WORLD   # 32 px per map unit

def _w2p(wx, wy):
    return (int(SPRITE_SZ//2 + wx*_SPX), int(SPRITE_SZ//2 - wy*_SPX))

def _make_vessel_sprite(L, W, own=True):
    if not _HAS_PIL:
        return np.zeros((SPRITE_SZ, SPRITE_SZ, 4), np.uint8)
    img = Image.new('RGBA', (SPRITE_SZ, SPRITE_SZ), (0, 0, 0, 0))
    d   = ImageDraw.Draw(img)
    hl, hw = L/2, W/2

    def poly(pts, fc, ec=None, ew=1):
        px = [_w2p(x, y) for x, y in pts]
        d.polygon(px, fill=fc, outline=ec, width=ew)

    def circ(cx, cy, r, fc, ec=None, ew=1):
        p, q = _w2p(cx, cy)
        rp   = max(1, int(r*_SPX))
        d.ellipse([p-rp, q-rp, p+rp, q+rp], fill=fc, outline=ec, width=ew)

    def seg(x0, y0, x1, y1, col, w=1):
        d.line([_w2p(x0, y0), _w2p(x1, y1)], fill=col, width=w)

    if own:
        # ── Cyan glow halo ─────────────────────────────────────────
        hs = [(hl+.07,0),(hl*.82,hw*.53),(hl*.38,hw+.06),(-hl*.36,hw+.06),
              (-hl-.05,hw*.55),(-hl-.05,0),(-hl-.05,-hw*.55),
              (-hl*.36,-hw-.06),(hl*.38,-hw-.06),(hl*.82,-hw*.53)]
        poly(hs, fc=(0, 200, 220, 38))

        # Hull (dark steel)
        h = [(hl,0),(hl*.82,hw*.50),(hl*.38,hw*.94),(-hl*.36,hw*.94),
             (-hl*.90,hw*.52),(-hl,0),(-hl*.90,-hw*.52),(-hl*.36,-hw*.94),
             (hl*.38,-hw*.94),(hl*.82,-hw*.50)]
        poly(h, fc=(12, 24, 44, 255), ec=(30, 72, 115, 255), ew=2)

        # Deck plating
        dk = [(hl*.88,0),(hl*.72,hw*.42),(hl*.30,hw*.82),(-hl*.28,hw*.82),
              (-hl*.82,hw*.46),(-hl*.88,0),(-hl*.82,-hw*.46),(-hl*.28,-hw*.82),
              (hl*.30,-hw*.82),(hl*.72,-hw*.42)]
        poly(dk, fc=(22, 44, 68, 255))

        # Deck plank texture lines
        for fy in (-.52, -.22, .22, .52):
            x0 = -hl*.80 if abs(fy) > .3 else -hl*.85
            seg(x0, fy*hw, hl*.73, fy*hw, (16, 33, 54, 155))

        # Aft helipad plate
        poly([(-hl*.70,hw*.38),(-hl*.28,hw*.38),(-hl*.28,-hw*.38),(-hl*.70,-hw*.38)],
             fc=(18, 36, 58, 255), ec=(30, 60, 92, 200), ew=1)
        # H marking
        for dx in (-.055, .055):
            seg(-hl*.49+dx, hw*.26, -hl*.49+dx, -hw*.26, (38, 88, 130, 145))
        seg(-hl*.555, 0, -hl*.425, 0, (38, 88, 130, 145))

        # Superstructure base block
        poly([(hl*.05,hw*.50),(hl*.34,hw*.50),(hl*.34,-hw*.50),(hl*.05,-hw*.50)],
             fc=(30, 56, 84, 255), ec=(50, 90, 132, 205), ew=1)

        # Bridge block
        poly([(hl*.10,hw*.37),(hl*.30,hw*.37),(hl*.30,-hw*.37),(hl*.10,-hw*.37)],
             fc=(44, 74, 107, 255), ec=(65, 112, 158, 225), ew=1)

        # Bridge windows (3 lit rectangles)
        for wy in (-hw*.22, 0, hw*.22):
            poly([(hl*.12, wy-hw*.07),(hl*.17, wy-hw*.07),
                  (hl*.17, wy+hw*.07),(hl*.12, wy+hw*.07)], fc=(78, 158, 200, 215))

        # Funnel/stack
        circ(-hl*.08, 0, hw*.14, fc=(26, 48, 72, 255), ec=(44, 82, 118, 205), ew=1)
        circ(-hl*.08, 0, hw*.06, fc=(10, 20, 34, 255))

        # Fore gun turret
        circ(hl*.58, 0, hw*.21, fc=(28, 54, 80, 255), ec=(55, 102, 148, 235), ew=2)
        circ(hl*.58, 0, hw*.11, fc=(18, 36, 58, 255))
        seg(hl*.58, 0, hl*.94, 0, (55, 102, 148, 235), 2)

        # Aft mount
        circ(-hl*.22, 0, hw*.16, fc=(26, 50, 76, 255), ec=(50, 92, 135, 205), ew=1)
        seg(-hl*.22, 0, -hl*.48, 0, (50, 92, 135, 180), 2)

        # Radar/mast dot
        circ(hl*.20, 0, hw*.056, fc=(90, 162, 215, 248))

        # Waterline highlight arc
        wh = [(hl*.88,0),(hl*.72,hw*.42),(hl*.30,hw*.82),
              (-hl*.28,hw*.82),(-hl*.82,hw*.46),(-hl*.88,0)]
        d.polygon([_w2p(x,y) for x,y in wh], fill=None, outline=(58, 118, 178, 68), width=1)

    else:
        # ── AIS — commercial cargo vessel ──────────────────────────
        h = [(hl*.88,0),(hl*.72,hw*.65),(hl*.28,hw),(-hl*.48,hw),(-hl,hw*.55),
             (-hl,0),(-hl,-hw*.55),(-hl*.48,-hw),(hl*.28,-hw),(hl*.72,-hw*.65)]
        poly(h, fc=(16, 28, 20, 255), ec=(38, 78, 48, 225), ew=2)

        dk = [(hl*.74,0),(hl*.60,hw*.55),(hl*.20,hw*.88),(-hl*.40,hw*.88),
              (-hl*.85,hw*.50),(-hl*.85,0),(-hl*.85,-hw*.50),(-hl*.40,-hw*.88),
              (hl*.20,-hw*.88),(hl*.60,-hw*.55)]
        poly(dk, fc=(20, 38, 27, 255))

        # Cargo hold hatches
        for hx in (hl*.38, hl*.08, -hl*.22, -hl*.52):
            poly([(hx-hw*.17, hw*.53),(hx+hw*.17, hw*.53),
                  (hx+hw*.17,-hw*.53),(hx-hw*.17,-hw*.53)],
                 fc=(24, 44, 30, 255), ec=(40, 80, 48, 178), ew=1)

        # Bridge tower (aft)
        poly([(-hl*.28,hw*.44),(-hl*.10,hw*.44),(-hl*.10,-hw*.44),(-hl*.28,-hw*.44)],
             fc=(30, 56, 38, 255), ec=(50, 96, 58, 205), ew=1)
        for wy in (-hw*.25, hw*.25):
            poly([(-hl*.26, wy-hw*.09),(-hl*.12, wy-hw*.09),
                  (-hl*.12, wy+hw*.09),(-hl*.26, wy+hw*.09)], fc=(55, 158, 75, 185))

        # Bow anchor chain plate
        circ(hl*.72, 0, hw*.09, fc=(28, 52, 34, 255), ec=(45, 90, 52, 180), ew=1)

    return np.array(img, dtype=np.uint8)


def _rot_sprite(arr, theta_rad):
    """Rotate RGBA uint8 array by theta_rad CCW. Returns 128×128 uint8 RGBA."""
    if not _HAS_PIL:
        return arr
    img = Image.fromarray(arr, 'RGBA')
    rotated = img.rotate(np.degrees(theta_rad), resample=2, expand=False)
    return np.array(rotated, dtype=np.uint8)

# Per-vessel rotation cache: only re-rotate when angle changes by ≥1°
_own_rot_cache = {'deg': None, 'data': None}
_ais_rot_caches = [{'deg': None, 'data': None} for _ in range(3)]

def _cached_rot(spr_arr, cache, theta_rad):
    deg = int(np.degrees(theta_rad) % 360)
    if cache['deg'] != deg:
        cache['deg'] = deg
        cache['data'] = _rot_sprite(spr_arr, theta_rad)
    return cache['data']


# ═══════════════════════════════════════════════════════════════════
#  BRESENHAM  (inverse SLAM)
# ═══════════════════════════════════════════════════════════════════
def _bresenham(x0, y0, x1, y1):
    pts = []
    dx, dy = abs(x1-x0), abs(y1-y0)
    sx = 1 if x0<x1 else -1
    sy = 1 if y0<y1 else -1
    err = dx - dy
    while True:
        pts.append((x0, y0))
        if x0==x1 and y0==y1: break
        e2 = 2*err
        if e2 > -dy: err -= dy; x0 += sx
        if e2 <  dx: err += dx; y0 += sy
    return pts

# ═══════════════════════════════════════════════════════════════════
#  EKF  — state [x, y, θ, v]
# ═══════════════════════════════════════════════════════════════════
class EKF:
    def __init__(self, x0, y0, th0):
        self.mu = np.array([x0, y0, th0, 0.0])
        self.P  = np.diag([0.5, 0.5, 0.1, 0.2])
        self.Q  = np.diag([0.025, 0.025, 0.008, 0.06])
        self.R_gps = np.diag([0.7, 0.7])
        self.R_imu = np.diag([0.04, 0.012])
        self.nis_log = deque(maxlen=80)

    def predict(self, v, omega, dt):
        x, y, th, _ = self.mu
        self.mu = np.array([x+v*np.cos(th)*dt, y+v*np.sin(th)*dt,
                            th+omega*dt, v])
        self.mu[2] = _wrap(self.mu[2])
        F = np.array([[1,0,-v*np.sin(th)*dt, np.cos(th)*dt],
                      [0,1, v*np.cos(th)*dt, np.sin(th)*dt],
                      [0,0,1,0],[0,0,0,1]])
        self.P = F @ self.P @ F.T + self.Q

    def _update(self, z, H, R):
        innov = z - H@self.mu
        S = H@self.P@H.T + R
        K = self.P@H.T@np.linalg.inv(S)
        self.mu += K@innov
        self.mu[2] = _wrap(self.mu[2])
        IKH = np.eye(4) - K@H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T   # Joseph form: numerically stable
        return innov, S

    def update_gps(self, gx, gy):
        H = np.array([[1,0,0,0],[0,1,0,0]])
        innov, S = self._update(np.array([gx,gy]), H, self.R_gps)
        self.nis_log.append(float(innov@np.linalg.inv(S)@innov))

    # NOTE: IMU is used as the process model control input in predict().
    # A separate measurement update for IMU would require a magnetometer
    # (for absolute heading) or a different sensor architecture. The
    # update_imu() method is intentionally not called in the main loop.
    def update_imu(self, v_m, w_m):
        H = np.array([[0,0,0,1],[0,0,1,0]])
        self._update(np.array([v_m,w_m]), H, self.R_imu)

    def ellipse_params(self):
        vals, vecs = np.linalg.eigh(np.maximum(self.P[:2,:2], 1e-9*np.eye(2)))
        idx = vals.argsort()[::-1]
        vals, vecs = vals[idx], vecs[:,idx]
        return (3*np.sqrt(max(vals[0],1e-9)), 3*np.sqrt(max(vals[1],1e-9)),
                np.degrees(np.arctan2(vecs[1,0],vecs[0,0])))

    @property
    def pose(self): return tuple(self.mu[:3])

def _wrap(a): return (a+np.pi)%(2*np.pi)-np.pi

# ═══════════════════════════════════════════════════════════════════
#  WORLD
# ═══════════════════════════════════════════════════════════════════
STATIC = [
    {'pos':(6.0,6.0),   'type':'rock',   'r':0.8},
    {'pos':(9.0,15.0),  'type':'rock',   'r':0.6},
    {'pos':(19.0,8.0),  'type':'rock',   'r':1.0},
    {'pos':(23.0,19.0), 'type':'rock',   'r':0.7},
    {'pos':(14.0,23.0), 'type':'rock',   'r':0.9},
    {'pos':(4.0,12.0),  'type':'rock',   'r':0.5},
    {'pos':(26.0,11.0), 'type':'buoy',   'r':0.4},
    {'pos':(10.0,26.0), 'type':'buoy',   'r':0.4},
    {'pos':(4.0,25.0),  'type':'buoy',   'r':0.4},
    {'pos':(21.0,4.0),  'type':'wreck',  'l':3.5,'w':1.2},
    {'pos':(5.0,20.0),  'type':'vessel', 'l':4.0,'w':1.5},
]
OBS_COLORS = {
    'rock':  '#4A4850', 'buoy':  '#D84010',
    'wreck': '#3A2818', 'vessel':'#1A2838',
}
OBS_EDGE = {
    'rock':  '#90A0B0', 'buoy':  '#FFB040',
    'wreck': '#705030', 'vessel':'#4080B0',
}

DYNAMIC = [
    {'pos':np.array([24.0,15.0]),'vel':np.array([-0.55, 0.20]),
     'l':3.5,'w':1.2,'theta':np.pi,    'name':'AIS-01'},
    {'pos':np.array([ 5.0,12.0]),'vel':np.array([ 0.40,-0.18]),
     'l':3.0,'w':1.0,'theta':0.0,      'name':'AIS-02'},
    {'pos':np.array([15.0,27.0]),'vel':np.array([ 0.25,-0.65]),
     'l':3.2,'w':1.1,'theta':-np.pi/2,'name':'AIS-03'},
]

def build_grid():
    g = np.zeros((MAP_W,MAP_H),np.float32)
    for obs in STATIC:
        x,y = obs['pos']
        if 'r' in obs:
            r=obs['r']
            for i in range(max(0,int(x-r)-1),min(MAP_W,int(x+r)+2)):
                for j in range(max(0,int(y-r)-1),min(MAP_H,int(y+r)+2)):
                    if (i-x)**2+(j-y)**2<=(r+0.25)**2: g[i,j]=1.0
        else:
            hl,hw=obs['l']/2,obs['w']/2
            for i in range(max(0,int(x-hl)-1),min(MAP_W,int(x+hl)+2)):
                for j in range(max(0,int(y-hw)-1),min(MAP_H,int(y+hw)+2)):
                    g[i,j]=1.0
    return g

GRID = build_grid()

# Pre-baked depth map
_DX,_DY = np.mgrid[0:MAP_W,0:MAP_H].astype(float)
_DEPTH = np.maximum(1.5,
    12.0+4.0*np.sin(_DX*0.4)*np.cos(_DY*0.35)
    +3.5*np.exp(-0.015*((_DX-MAP_W/2)**2+(_DY-MAP_H/2)**2))
    -sum(np.maximum(0,3.5-np.hypot(_DX-o['pos'][0],_DY-o['pos'][1]))for o in STATIC))

# ═══════════════════════════════════════════════════════════════════
#  RETRO OCEAN  — dark tactical map aesthetic
# ═══════════════════════════════════════════════════════════════════
_OX,_OY = np.meshgrid(np.linspace(0,MAP_W,MAP_W*4),
                       np.linspace(0,MAP_H,MAP_H*4))

def ocean_frame(phase, sea_b=0):
    b = min(1.0, sea_b/5.0)
    # Primary swell (large slow waves)
    w  = (np.sin(0.38*_OX + phase)       * np.cos(0.28*_OY + 0.68*phase) + 1) / 2
    # Cross-swell
    w += (np.sin(0.58*_OX - 0.48*phase)  * np.cos(0.18*_OY + 0.38*phase) + 1) / 4
    # High-sea-state chop
    w += b*(np.sin(1.10*_OX + 1.4*phase) * np.cos(0.85*_OY + 1.1*phase)  + 1) / 8
    # Fine surface ripples
    rip = (np.sin(2.20*_OX + 0.9*phase + 0.5) * np.cos(1.8*_OY - 1.1*phase) + 1) / 16
    w   = (w + rip) / (w.max() + 1e-9)

    # Deep navy base — slightly brighter peaks for visible structure
    r  = np.clip(0.008 + 0.026*w,              0, 1)
    g  = np.clip(0.025 + 0.078*w + b*0.016,    0, 1)
    bl = np.clip(0.115 + 0.215*w - b*0.022,    0, 1)

    # Foam / whitecapping at wave crests
    foam_t = 0.80
    foam   = np.where(w > foam_t, (w - foam_t) / (1.0 - foam_t), 0.0)
    fs     = (0.16 + b*0.28) * foam            # stronger foam at high sea state
    r  = np.clip(r  + fs*0.38, 0, 1)
    g  = np.clip(g  + fs*0.50, 0, 1)
    bl = np.clip(bl + fs*0.60, 0, 1)

    # Cool-teal shimmer at extreme crests (bioluminescence / spray)
    glow = np.where(w > 0.92, b*0.08*(w - 0.92)/0.08, 0)
    return np.stack([r + glow*0.04, g + glow*0.20, bl + glow*0.55], -1).clip(0, 1)

# Precomputed scanline texture (static, baked once)
_SL_H, _SL_W = MAP_H*4, MAP_W*4
_scanlines_rgba = np.zeros((_SL_H,_SL_W,4),np.float32)
_scanlines_rgba[::2,:,3]  = 0.018   # subtle texture only — not distracting
_scanlines_rgba[1::2,:,3] = 0.0

# ═══════════════════════════════════════════════════════════════════
#  SIMULATION PARAMETERS  (keyboard-controlled)
# ═══════════════════════════════════════════════════════════════════
SP = {
    'gps_offline': False,
    'lidar_deg':   False,
    'imu_spike':   False,
    'sea_state':   0,
    'autopilot':   False,
}

# ═══════════════════════════════════════════════════════════════════
#  SENSORS
# ═══════════════════════════════════════════════════════════════════
def cast_lidar(pose):
    x, y, th = pose
    eff   = LDR_RANGE * (0.30 if SP['lidar_deg'] else 1.0)
    ga    = th + LDR_ANGLES
    cos_a = np.cos(ga); sin_a = np.sin(ga)

    # Fully vectorised: all steps × all beams in one batch (no Python loop)
    steps = _STEPS[_STEPS < eff + 0.01]          # (N,)
    px = x + np.outer(steps, cos_a)               # (N, 360)
    py = y + np.outer(steps, sin_a)               # (N, 360)

    oob  = (px < 0) | (px >= MAP_W - 0.01) | (py < 0) | (py >= MAP_H - 0.01)
    pxi  = np.clip(px.astype(np.int32), 0, MAP_W - 1)
    pyi  = np.clip(py.astype(np.int32), 0, MAP_H - 1)
    hit  = oob | (GRID[pxi, pyi] > 0)
    for obs in DYNAMIC:
        hit |= np.hypot(px - obs['pos'][0], py - obs['pos'][1]) < (obs['l'] + obs['w']) / 4.0

    has_hit   = hit.any(axis=0)                   # (360,) — any step blocked?
    first_idx = hit.argmax(axis=0)                # (360,) — index of first hit
    ranges    = np.where(has_hit, steps[first_idx], eff).astype(np.float32)

    return ranges.astype(float) + np.random.normal(0, 0.06 + SP['sea_state'] * 0.018, LDR_BEAMS)

def gps_reading(true_pose):
    if SP['gps_offline']: return None
    if np.random.rand()<0.08+SP['sea_state']*0.016: return None
    is_mp = np.random.rand()<0.04
    noise = 1.4 if is_mp else (0.28+SP['sea_state']*0.07)
    x,y,_ = true_pose
    return (x+np.random.normal(0,noise), y+np.random.normal(0,noise), is_mp)

def imu_reading(v,omega,t):
    # Drift is a yaw-rate bias (rad/s) — characteristic of MEMS gyros.
    # It accumulates as a slowly growing heading error, not a speed offset.
    ang_drift = 0.00015*t + SP['sea_state']*0.00008*t
    spike     = 0.45 if SP['imu_spike'] else 0.0
    return (v     + np.random.normal(0,0.025+SP['sea_state']*0.009) + spike,
            omega + np.random.normal(0,0.007+SP['sea_state']*0.003) + ang_drift)

# ═══════════════════════════════════════════════════════════════════
#  DWA — trajectory rollout  (5×7 samples, 1.5 s horizon)
# ═══════════════════════════════════════════════════════════════════
def dwa(pose, vel, ranges, target_heading=None):
    if target_heading is None: target_heading=pose[2]
    v_lo=max(0.0,vel[0]-0.9); v_hi=min(MAX_SPD,vel[0]+0.5)
    o_lo=max(-MAX_OMG,vel[1]-2.0); o_hi=min(MAX_OMG,vel[1]+2.0)
    fwd_min=float(ranges[LDR_BEAMS//2-22:LDR_BEAMS//2+22].min())
    best_sc,best_v,best_o=-np.inf,vel[0],vel[1]
    for v_t in np.linspace(v_lo,v_hi,5):
        for o_t in np.linspace(o_lo,o_hi,7):
            x,y,th=pose; ok=True
            for _ in range(30):
                x+=v_t*np.cos(th)*DT; y+=v_t*np.sin(th)*DT; th+=o_t*DT
                if not(0.6<x<MAP_W-0.6 and 0.6<y<MAP_H-0.6): ok=False; break
                ix,iy=int(np.clip(x,0,MAP_W-1)),int(np.clip(y,0,MAP_H-1))
                if GRID[ix,iy]: ok=False; break
                for obs in DYNAMIC:
                    if np.hypot(x-obs['pos'][0],y-obs['pos'][1])<1.3: ok=False; break
                if not ok: break
            if not ok: continue
            h_err=abs(_wrap(target_heading-th))
            sc=2.0*(1.0-h_err/np.pi)+1.5*(v_t/MAX_SPD)+1.0*min(fwd_min/LDR_RANGE,1.0)
            if sc>best_sc: best_sc,best_v,best_o=sc,v_t,o_t
    v_sc=(best_v/vel[0]) if vel[0]>0.02 else 1.0
    return float(np.clip(v_sc,0.0,1.2)), float(best_o-vel[1])

# ═══════════════════════════════════════════════════════════════════
#  INVERSE-MODEL SLAM
# ═══════════════════════════════════════════════════════════════════
slam_map     = np.zeros((MAP_W, MAP_H), np.float32)
_slam_visited= np.zeros((MAP_W, MAP_H), dtype=bool)   # explored cells

def update_slam(pose, ranges):
    x,y,th=pose
    ga=th+LDR_ANGLES; eff=LDR_RANGE*(0.30 if SP['lidar_deg'] else 1.0)
    for r,a in zip(ranges[::3],ga[::3]):
        ex=int(np.clip(x+r*np.cos(a),0,MAP_W-1))
        ey=int(np.clip(y+r*np.sin(a),0,MAP_H-1))
        for cx,cy in _bresenham(int(np.clip(x,0,MAP_W-1)),
                                  int(np.clip(y,0,MAP_H-1)),ex,ey)[:-1:2]:
            if 0<=cx<MAP_W and 0<=cy<MAP_H:
                slam_map[cx,cy]=max(0.0,slam_map[cx,cy]-0.04)
                _slam_visited[cx,cy]=True
        if r<eff*0.97:
            slam_map[ex,ey]=min(1.0,slam_map[ex,ey]+0.22)
        _slam_visited[ex,ey]=True

# ═══════════════════════════════════════════════════════════════════
#  GEOMETRY
# ═══════════════════════════════════════════════════════════════════
def boat_verts(x,y,th,L=BOAT_L,W=BOAT_W):
    pts=np.array([[L*0.52,0],[L*0.32,W*0.50],
                  [-L*0.50,W*0.50],[-L*0.50,-W*0.50],[L*0.32,-W*0.50]])
    c,s=np.cos(th),np.sin(th)
    return pts@np.array([[c,-s],[s,c]]).T+[x,y]

def nav_light_pos(x,y,th):
    c,s=np.cos(th),np.sin(th)
    def r(lx,ly): return x+lx*c-ly*s, y+lx*s+ly*c
    return r(-BOAT_L*0.05, BOAT_W*0.48), r(-BOAT_L*0.05,-BOAT_W*0.48), r(BOAT_L*0.38,0)

def reticle_verts(cx,cy,r,angle):
    """Rotating diamond reticle around an obstacle."""
    a=np.array([0,np.pi/2,np.pi,3*np.pi/2])+angle
    inner=r*0.65; outer=r
    verts=[]
    for ai in a:
        verts.append((cx+inner*np.cos(ai-0.35), cy+inner*np.sin(ai-0.35)))
        verts.append((cx+outer*np.cos(ai),       cy+outer*np.sin(ai)))
        verts.append((cx+inner*np.cos(ai+0.35), cy+inner*np.sin(ai+0.35)))
    return verts

# ═══════════════════════════════════════════════════════════════════
#  ADAS HELPERS
# ═══════════════════════════════════════════════════════════════════
def compute_cpa(rx,ry,rvx,rvy,ox,oy,ovx,ovy):
    dpx,dpy=ox-rx,oy-ry; dvx,dvy=ovx-rvx,ovy-rvy
    dv2=dvx**2+dvy**2
    if dv2<1e-9: return float(np.hypot(dpx,dpy)),0.0
    tcpa=-(dpx*dvx+dpy*dvy)/dv2
    return float(np.hypot(dpx+tcpa*dvx,dpy+tcpa*dvy)),float(tcpa)

def threat_level(cpa_d,tcpa):
    if tcpa<0 or tcpa>TCPA_WARN: return 'green'
    if cpa_d<CPA_RED:   return 'red'
    if cpa_d<CPA_AMBER: return 'amber'
    return 'green'

def colregs_rule(own_hdg,own_xy,obs_xy):
    # Bearing from own vessel to target (relative to own heading)
    bearing_to_target = _wrap(np.arctan2(obs_xy[1]-own_xy[1],obs_xy[0]-own_xy[0]) - own_hdg)
    # Bearing from target back to own (for Rule 14 reciprocal-course check)
    bearing_back = _wrap(np.arctan2(own_xy[1]-obs_xy[1],own_xy[0]-obs_xy[0]) - own_hdg)
    bd = np.degrees(bearing_to_target)
    # Rule 14 HEAD-ON: target ahead AND approaching on reciprocal course
    reciprocal_check = abs(np.degrees(_wrap(bearing_to_target - bearing_back + np.pi))) < 45.0
    if abs(bd) < 22.5 and reciprocal_check: return "HEAD-ON"
    if  22.5 <= bd <=  112.5: return "GIVE WAY"    # Rule 15: target on starboard
    if -112.5 <= bd < -22.5:  return "STAND ON"    # Rule 15: target on port
    if abs(bd) > 112.5:       return "OVERTAKING"  # Rule 13: from astern
    return "CROSSING"

def sonar_at(x,y):
    ix,iy=int(np.clip(x,0,MAP_W-1)),int(np.clip(y,0,MAP_H-1))
    return float(_DEPTH[ix,iy]+np.random.normal(0,0.08))

# ═══════════════════════════════════════════════════════════════════
#  PPI BUFFER
# ═══════════════════════════════════════════════════════════════════
ppi_img = np.zeros((PPI_N,PPI_N),np.float32)

def update_ppi(pose, ranges, radar_ang):
    ga = pose[2] + LDR_ANGLES
    ppi_img[:] *= PPI_DECAY
    eff = LDR_RANGE * (0.30 if SP['lidar_deg'] else 1.0)

    # Fully vectorised — no Python loop over beams
    r2    = ranges[::2];  a2 = ga[::2]
    near2 = (np.abs(((a2 - radar_ang + np.pi) % (2*np.pi)) - np.pi) < 0.42)
    mask  = r2 < eff * 0.97
    r2, a2, near2 = r2[mask], a2[mask], near2[mask]

    px = (PPI_R + r2 * PPI_SCALE * np.cos(a2)).astype(np.int32)
    py = (PPI_R + r2 * PPI_SCALE * np.sin(a2)).astype(np.int32)
    valid = (px >= 0) & (px < PPI_N) & (py >= 0) & (py < PPI_N)
    px, py, near2 = px[valid], py[valid], near2[valid]

    np.add.at(ppi_img, (py, px), np.where(near2, 0.80, 0.13))
    np.clip(ppi_img, 0.0, 1.0, out=ppi_img)
    ppi_img[PPI_MASK] = 0.0

# ═══════════════════════════════════════════════════════════════════
#  AUTOPILOT
# ═══════════════════════════════════════════════════════════════════
def autopilot_cmd(pose,waypoint,cpa_data,vel):
    if waypoint is None: return vel[0]*0.95,vel[1]*0.80
    dx,dy=waypoint[0]-pose[0],waypoint[1]-pose[1]
    dist=np.hypot(dx,dy)
    if dist<1.5: return 0.0,0.0
    h_err=_wrap(np.arctan2(dy,dx)-pose[2])
    for _,_,_,rule in cpa_data:
        if rule=="GIVE WAY": h_err-=0.40
    return float(np.clip(dist*0.55,0,MAX_SPD*0.85)), float(np.clip(h_err*2.2,-MAX_OMG,MAX_OMG))

# ═══════════════════════════════════════════════════════════════════
#  SIMULATION ALERT SYSTEM  (events, not game notifications)
# ═══════════════════════════════════════════════════════════════════
_alert_queue  = []
_alert_timer  = [0]
_alert_colors = []   # parallel colour list

def _fire_alert(msg, color=AMBER):
    _alert_queue.append((msg, color))

# ═══════════════════════════════════════════════════════════════════
#  STATE
# ═══════════════════════════════════════════════════════════════════
INIT=[3.0,3.0,0.4]
pose=INIT[:]
velocity=[0.0,0.0]
pressed=set()
ekf=EKF(*INIT)

act_x,act_y=[INIT[0]],[INIT[1]]
est_x,est_y=[INIT[0]],[INIT[1]]
gps_xs,gps_ys=[],[]
gps_mp_xs,gps_mp_ys=[],[]
wake=deque(maxlen=140)
gps_timer=0; gps_locked=False
radar_ang=0.0; wave_phase=0.0
_waypoint=[None]
rmse_log=deque(maxlen=80)
reticle_angle=0.0   # global rotation for detection reticles

# ═══════════════════════════════════════════════════════════════════
#  FIGURE + AXES
# ═══════════════════════════════════════════════════════════════════
plt.rcParams.update({'font.family':'monospace','text.color':TEXT_HI,
                     'axes.labelcolor':TEXT_DIM,'xtick.color':TEXT_DIM,
                     'ytick.color':TEXT_DIM})

fig = plt.figure(figsize=(22,13), facecolor='#D4DAE2', dpi=96)
fig.suptitle(
    "MARINE ADAS SIMULATOR   ·   EKF · DWA · SLAM · CPA / TCPA · COLREGS",
    color='#1C2A3A', fontsize=11.5, fontweight='bold', y=0.997, family='monospace',
    alpha=0.92
)

gs = gridspec.GridSpec(5,2,figure=fig,
                       left=0.030,right=0.982,top=0.958,bottom=0.050,
                       wspace=0.05,hspace=0.42,
                       width_ratios=[58,42],
                       height_ratios=[1.5, 1.8, 1.8, 1.2, 1.2])

ax_main  = fig.add_subplot(gs[:, 0])
ax_radar = fig.add_subplot(gs[0,   1])
ax_slam  = fig.add_subplot(gs[1:3, 1])   # spans 2 rows — bigger SLAM map
ax_sens  = fig.add_subplot(gs[3,   1])
ax_perf  = fig.add_subplot(gs[4,   1])

# Bottom info strip
ax_info = fig.add_axes([0.030,0.008,0.952,0.034])
ax_info.axis('off'); ax_info.set_facecolor('#BFC8D2')
ax_info.patch.set_visible(True)

ax_main.set_facecolor(DARK_BG)
for sp in ax_main.spines.values():
    sp.set_edgecolor(BORDER_HI); sp.set_linewidth(1.6)

def _style_light(ax):
    """Apply light-panel theme; safe to call before or after axis('off')."""
    ax.set_facecolor(LP_BG)
    ax.patch.set_facecolor(LP_BG)   # explicit patch for axis('off') safety
    ax.patch.set_visible(True)
    ax.patch.set_alpha(1.0)
    for sp in ax.spines.values():
        sp.set_edgecolor(LP_EDGE); sp.set_linewidth(1.4)
    ax.tick_params(colors=LP_DIM, labelcolor=LP_DIM, labelsize=5.5)
    ax.xaxis.label.set_color(LP_DIM)
    ax.yaxis.label.set_color(LP_DIM)

for ax in (ax_radar, ax_slam, ax_sens, ax_perf):
    _style_light(ax)

# Clean panel border with single subtle bloom
def _glow_border(ax, color=PHOSPHOR):
    # Outer bloom — very subtle
    ax.add_patch(Rectangle((0,0),1,1,transform=ax.transAxes,
                            fill=False,ec=color,lw=5,alpha=0.05,
                            clip_on=False,zorder=50))
    # Crisp inner border — refined, not harsh
    ax.add_patch(Rectangle((0,0),1,1,transform=ax.transAxes,
                            fill=False,ec=color,lw=0.9,alpha=0.45,
                            clip_on=False,zorder=50))

_glow_border(ax_main, PHOSPHOR)
for ax in (ax_radar, ax_slam, ax_sens, ax_perf):
    _glow_border(ax, LP_EDGE)

# ═══════════════════════════════════════════════════════════════════
#  MAIN NAVIGATION VIEW
# ═══════════════════════════════════════════════════════════════════
ax_main.set_xlim(0,MAP_W); ax_main.set_ylim(0,MAP_H)
ax_main.set_aspect('equal')
ax_main.set_xlabel("EAST  (m)", fontsize=7, color=TEXT_MID)
ax_main.set_ylabel("NORTH (m)", fontsize=7, color=TEXT_MID)
ax_main.grid(True,color=GRID_CLR,linewidth=0.50,alpha=1.0)

# Ocean + scanline overlay
ocean_im = ax_main.imshow(ocean_frame(0), origin='lower',
                           extent=(0,MAP_W,0,MAP_H), alpha=0.92, zorder=0)
scan_im  = ax_main.imshow(_scanlines_rgba, origin='lower',
                           extent=(0,MAP_W,0,MAP_H), alpha=1.0,
                           zorder=16, interpolation='nearest')

# Depth contours in phosphor green
_dc_x=np.linspace(0,MAP_W,120); _dc_y=np.linspace(0,MAP_H,120)
_dc_xx,_dc_yy=np.meshgrid(_dc_x,_dc_y)
_dc_d=_DEPTH[np.clip(_dc_xx.astype(int),0,MAP_W-1),
             np.clip(_dc_yy.astype(int),0,MAP_H-1)]
ax_main.contour(_dc_xx,_dc_yy,_dc_d,levels=[4,7,10,13],
                colors=[PHOSPHOR2],linewidths=0.40,alpha=0.30,zorder=1)

# Range rings from map centre (tactical reference circles)
_mc=(MAP_W/2,MAP_H/2)
for _r,_a in ((8,0.10),(15,0.07),(22,0.05)):
    ax_main.add_patch(Circle(_mc,_r,fill=False,ec=PHOSPHOR,lw=0.5,
                              alpha=_a,ls=':',zorder=1))

# Static obstacles — type-specific detailed rendering
for obs in STATIC:
    x,y=obs['pos']
    fc=OBS_COLORS[obs['type']]; ec=OBS_EDGE[obs['type']]
    if 'r' in obs:
        r=obs['r']
        if obs['type']=='rock':
            # Irregular polygon for natural rocky appearance
            _seed=int(x*113+y*71)&0xFFFF
            _rng=np.random.default_rng(_seed)
            _n=9; _ang=np.linspace(0,2*np.pi,_n,endpoint=False)
            _rad=r*(0.72+0.32*_rng.random(_n))
            _vx=x+_rad*np.cos(_ang); _vy=y+_rad*np.sin(_ang)
            ax_main.add_patch(Polygon(list(zip(_vx,_vy)),closed=True,
                                       fc=fc,ec=ec,lw=1.8,alpha=0.95,zorder=4))
            # Inner highlight facet
            _rad2=r*(0.38+0.18*_rng.random(_n))
            _vx2=x+_rad2*np.cos(_ang+0.3); _vy2=y+_rad2*np.sin(_ang+0.3)
            ax_main.add_patch(Polygon(list(zip(_vx2,_vy2)),closed=True,
                                       fc='#7888A0',ec='none',alpha=0.30,zorder=5))
            ax_main.add_patch(Circle((x,y),r+0.22,color=ec,alpha=0.18,
                                      linewidth=0,zorder=3))
        elif obs['type']=='buoy':
            # Bright navigational buoy — solid orange disc + cross marker
            ax_main.add_patch(Circle((x,y),r,fc=fc,ec=ec,lw=2.2,
                                      alpha=0.95,zorder=5))
            ax_main.add_patch(Circle((x,y),r+0.25,color=ec,alpha=0.25,
                                      linewidth=0,zorder=4))
            ax_main.plot([x-r*0.55,x+r*0.55],[y,y],color='white',lw=1.0,alpha=0.85,zorder=6)
            ax_main.plot([x,x],[y-r*0.55,y+r*0.55],color='white',lw=1.0,alpha=0.85,zorder=6)
        else:
            ax_main.add_patch(Circle((x,y),r,fc=fc,ec=ec,lw=1.6,
                                      alpha=0.90,zorder=4))
            ax_main.add_patch(Circle((x,y),r+0.20,color=ec,alpha=0.12,
                                      linewidth=0,zorder=3))
        r_lbl=r
    else:
        # Wreck — elongated shape with internal girder lines
        ax_main.add_patch(Rectangle((x-obs['l']/2,y-obs['w']/2),
                                     obs['l'],obs['w'],fc=fc,ec=ec,
                                     lw=1.8,alpha=0.92,zorder=4))
        # Cross-bracing detail
        ax_main.plot([x-obs['l']/2,x+obs['l']/2],[y,y],
                     color=ec,lw=0.7,alpha=0.45,zorder=5)
        ax_main.plot([x,x],[y-obs['w']/2,y+obs['w']/2],
                     color=ec,lw=0.7,alpha=0.45,zorder=5)
        r_lbl=obs['w']/2
    ax_main.text(x,y+r_lbl+0.42,obs['type'].upper(),
                 color=ec,fontsize=4,ha='center',va='bottom',
                 family='monospace',alpha=0.80,zorder=6)

# Detection reticles — rotating diamond polygons (replace circles)
det_rings=[]
for obs in STATIC:
    x,y=obs['pos']
    r_ret=(obs.get('r',max(obs.get('l',2),obs.get('w',1))/2)+0.75)
    ring=Polygon(reticle_verts(x,y,r_ret,0),closed=True,
                 ec=PHOSPHOR,fill=False,lw=1.8,visible=False,
                 alpha=0.90,zorder=6)
    ax_main.add_patch(ring); det_rings.append((ring,x,y,r_ret))

# AIS vessels — glow + hull + heading vectors + CPA markers
dyn_patches=[]; dyn_glows=[]; dyn_labels=[]
ais_hdg_lines=[]; cpa_markers=[]
for obs in DYNAMIC:
    gw=Polygon(boat_verts(*obs['pos'],obs['theta'],obs['l'],obs['w']),
               closed=True,facecolor=PHOSPHOR,ec='none',alpha=0.12,zorder=7)
    p =Polygon(boat_verts(*obs['pos'],obs['theta'],obs['l'],obs['w']),
               closed=True,facecolor=OBS_COLORS['vessel'],
               ec=PHOSPHOR,lw=1.8,zorder=8)
    ax_main.add_patch(gw); ax_main.add_patch(p)
    dyn_glows.append(gw); dyn_patches.append(p)
    lbl=ax_main.text(*obs['pos'],obs['name'],color=PHOSPHOR,
                     fontsize=5,ha='center',va='bottom',
                     family='monospace',fontweight='bold',zorder=9)
    dyn_labels.append(lbl)
    hl,=ax_main.plot([],[],lw=1.6,ls='--',alpha=0.85,zorder=8)
    ais_hdg_lines.append(hl)
    cm,=ax_main.plot([],[],'x',ms=8,mew=2.2,zorder=9,alpha=0.90)
    cpa_markers.append(cm)

# Trajectories
traj_act,=ax_main.plot([],[],color=PHOSPHOR2,lw=1.0,alpha=0.55,
                        label='Actual path',zorder=7)
traj_est,=ax_main.plot([],[],color=MAGENTA_P,lw=1.0,alpha=0.55,
                        ls='--',label='EKF estimate',zorder=7)
wake_sc=ax_main.scatter([],[],c=PHOSPHOR,s=8,alpha=0.28,
                         zorder=3,edgecolors='none')

# LiDAR beams
lidar_lines=[ax_main.plot([],[],lw=0.28,alpha=0.10,
                            solid_capstyle='round')[0] for _ in range(LDR_BEAMS)]

# Radar sweep — double draw for glow
radar_glow,=ax_main.plot([],[],color=PHOSPHOR,lw=6.0,alpha=0.06,zorder=9)
radar_line,=ax_main.plot([],[],color=PHOSPHOR,lw=1.8,alpha=0.85,zorder=9)
radar_sweep=Polygon([(0,0)]*30,closed=True,facecolor=PHOSPHOR,
                     alpha=0.04,zorder=8,edgecolor='none')
ax_main.add_patch(radar_sweep)

# GPS markers
gps_sc   =ax_main.scatter([],[],marker='+',s=55,color=AMBER,
                            linewidths=1.4,alpha=0.60,zorder=8,label='GPS fix')
gps_mp_sc=ax_main.scatter([],[],marker='+',s=80,color=RED_HI,
                            linewidths=1.8,alpha=0.75,zorder=8,label='GPS multipath')

# EKF ellipse — glow + crisp
cov_glow=Ellipse((INIT[0],INIT[1]),1,1,ec=TEXT_HI,fill=False,
                  lw=5,ls=':',alpha=0.10,zorder=10)
cov_ell =Ellipse((INIT[0],INIT[1]),0.5,0.5,ec=TEXT_HI,fill=False,
                  lw=1.6,ls=':',zorder=11)
ax_main.add_patch(cov_glow); ax_main.add_patch(cov_ell)

# Pre-bake vessel sprites
_own_spr  = _make_vessel_sprite(BOAT_L, BOAT_W, own=True)
_ais_sprs = [_make_vessel_sprite(o['l'], o['w'], own=False) for o in DYNAMIC]

# Own vessel — PIL sprite (shown when PIL available) + polygon fallback
_init_rot  = _cached_rot(_own_spr, _own_rot_cache, INIT[2])
_sw        = SPRITE_WORLD / 2
boat_im    = ax_main.imshow(
    _init_rot, origin='upper',
    extent=[INIT[0]-_sw, INIT[0]+_sw, INIT[1]-_sw, INIT[1]+_sw],
    zorder=10, interpolation='bicubic', alpha=1.0)
boat_glow  = Polygon(boat_verts(*INIT),closed=True,fc=CYAN_HI,
                     ec='none',alpha=0.12,zorder=9)
boat_patch = Polygon(boat_verts(*INIT),closed=True,fc='#0C1E38',
                     ec=CYAN_HI,lw=2.4,zorder=10)
ax_main.add_patch(boat_glow); ax_main.add_patch(boat_patch)
# Hide polygon fallback when PIL available; keep for blit return list
boat_glow.set_visible(not _HAS_PIL)
boat_patch.set_visible(not _HAS_PIL)

# V-shaped Kelvin wake
wake_v_l, = ax_main.plot([], [], color='#B0D8EC', lw=1.5, alpha=0.0, zorder=4)
wake_v_r, = ax_main.plot([], [], color='#B0D8EC', lw=1.5, alpha=0.0, zorder=4)
wake_v_l2,= ax_main.plot([], [], color='#D0EAF8', lw=0.7, alpha=0.0, zorder=4)
wake_v_r2,= ax_main.plot([], [], color='#D0EAF8', lw=0.7, alpha=0.0, zorder=4)

# AIS vessel PIL sprites (overlay on top of polygon glows)
ais_ims = []
for i, obs in enumerate(DYNAMIC):
    _arot = _cached_rot(_ais_sprs[i], _ais_rot_caches[i], obs['theta'])
    _aim  = ax_main.imshow(
        _arot, origin='upper',
        extent=[obs['pos'][0]-_sw, obs['pos'][0]+_sw,
                obs['pos'][1]-_sw, obs['pos'][1]+_sw],
        zorder=9, interpolation='bicubic', alpha=1.0)
    ais_ims.append(_aim)
    # Hide polygon fallback patches
    dyn_patches[i].set_visible(not _HAS_PIL)
    dyn_glows[i].set_visible(not _HAS_PIL)

hdg_glow,=ax_main.plot([],[],color=CYAN_HI,lw=5.5,alpha=0.12,zorder=11)
hdg_line,=ax_main.plot([],[],color=CYAN_HI,lw=2.4,zorder=12,
                        solid_capstyle='round')

# Nav lights
lp=Circle((0,0),0.18,color=RED_HI, zorder=13,alpha=0.92)
ls=Circle((0,0),0.18,color=PHOSPHOR,zorder=13,alpha=0.92)
lm=Circle((0,0),0.13,color='white', zorder=13,alpha=0.85)
lpg=Circle((0,0),0.45,color=RED_HI, alpha=0.14,linewidth=0,zorder=12)
lsg=Circle((0,0),0.45,color=PHOSPHOR,alpha=0.14,linewidth=0,zorder=12)
for p in (lp,ls,lm,lpg,lsg): ax_main.add_patch(p)

# Waypoint
wp_dot, =ax_main.plot([],[],marker=(4,1,0),ms=13,color=AMBER,
                       zorder=13,mew=1.5,mec='white',label='Waypoint')
wp_line,=ax_main.plot([],[],color=AMBER,lw=0.9,ls=':',alpha=0.55,zorder=7)

ax_main.legend(loc='lower left',facecolor='#060D14',edgecolor=BORDER_HI,
               labelcolor=TEXT_HI,fontsize=6,framealpha=0.92,
               fancybox=False)

# ── HUD  (retro terminal style) ───────────────────────────────────
_hb=dict(facecolor='#060D14',alpha=0.92,edgecolor=BORDER_HI,
         pad=3,boxstyle='square,pad=0.35')
hud1   =ax_main.text(0.020,0.978,'',transform=ax_main.transAxes,
                      color=PHOSPHOR,fontsize=9.5,va='top',family='monospace',
                      fontweight='bold',bbox=_hb)
hud2   =ax_main.text(0.020,0.918,'',transform=ax_main.transAxes,
                      color=TEXT_HI,fontsize=7.5,va='top',family='monospace',bbox=_hb)
hud3   =ax_main.text(0.020,0.862,'',transform=ax_main.transAxes,
                      color=PHOSPHOR2,fontsize=7.5,va='top',family='monospace',bbox=_hb)
dwa_txt=ax_main.text(0.020,0.808,'',transform=ax_main.transAxes,
                      color=AMBER,fontsize=7.5,va='top',family='monospace',bbox=_hb)
wp_txt =ax_main.text(0.500,0.978,'',transform=ax_main.transAxes,
                      color=AMBER,fontsize=7.5,va='top',ha='center',
                      family='monospace',bbox=_hb)

# Anti-grounding alarm bar
ag_txt=ax_main.text(0.500,0.028,'',transform=ax_main.transAxes,
                     color='white',fontsize=11,va='bottom',ha='center',
                     fontweight='bold',family='monospace',
                     bbox=dict(facecolor=RED_HI,alpha=0.0,edgecolor='none',
                               pad=5,boxstyle='square,pad=0.4'))

# Simulation alert toast (CPA events, GPS dropout, etc.)
alert_txt=ax_main.text(0.500,0.500,'',transform=ax_main.transAxes,
                        color=AMBER,fontsize=13,va='center',ha='center',
                        fontweight='bold',family='monospace',alpha=0.0,zorder=25,
                        bbox=dict(facecolor='#060D14',alpha=0.0,
                                  edgecolor=AMBER,pad=7,
                                  boxstyle='square,pad=0.5'))

# Failure/mode badges (right side, retro block style)
_bb=dict(pad=3,boxstyle='square,pad=0.28')
badge_gps=ax_main.text(0.980,0.862,'',transform=ax_main.transAxes,
                        color='white',fontsize=7,va='top',ha='right',
                        family='monospace',
                        bbox=dict(facecolor=RED_HI,alpha=0.0,edgecolor='none',**_bb))
badge_ldr=ax_main.text(0.980,0.824,'',transform=ax_main.transAxes,
                        color='white',fontsize=7,va='top',ha='right',
                        family='monospace',
                        bbox=dict(facecolor=ORANGE_HI,alpha=0.0,edgecolor='none',**_bb))
badge_ap =ax_main.text(0.980,0.786,'',transform=ax_main.transAxes,
                        color='white',fontsize=7,va='top',ha='right',
                        family='monospace',
                        bbox=dict(facecolor=TEXT_HI,alpha=0.0,edgecolor='none',**_bb))

# Compass rose (retro tactical style)
_cr=(MAP_W-2.4,2.4)
ax_main.add_patch(Circle(_cr,1.10,fill=False,ec=BORDER_HI,lw=0.8,zorder=5))
ax_main.add_patch(Circle(_cr,1.10,fill=False,ec=PHOSPHOR, lw=0.3,alpha=0.4,zorder=5))
for ang,lbl,col in [(0,'E',TEXT_MID),(np.pi/2,'N',PHOSPHOR),
                     (np.pi,'W',TEXT_MID),(-np.pi/2,'S',TEXT_MID)]:
    ax_main.text(_cr[0]+1.42*np.cos(ang),_cr[1]+1.42*np.sin(ang),lbl,
                 color=col,fontsize=6,ha='center',va='center',family='monospace')

# ═══════════════════════════════════════════════════════════════════
#  CIRCULAR GAUGES  (retro phosphor dials)
# ═══════════════════════════════════════════════════════════════════
def _make_gauge(ax,label,full_circle=False):
    ax.set_facecolor('#060D14'); ax.set_xlim(-1.3,1.3); ax.set_ylim(-1.3,1.3)
    ax.set_aspect('equal'); ax.axis('off')
    glow_col=PHOSPHOR
    if full_circle:
        # outer glow ring
        ax.add_patch(Circle((0,0),1.05,fill=False,ec=glow_col,lw=4,alpha=0.08))
        ax.add_patch(Circle((0,0),1.00,fill=False,ec=glow_col,lw=1.4,alpha=0.70))
        for a_d in range(0,360,30):
            a=np.radians(a_d); inner=0.68 if a_d%90==0 else 0.80
            ax.plot([inner*np.cos(a),0.94*np.cos(a)],
                    [inner*np.sin(a),0.94*np.sin(a)],
                    color=BORDER_HI,lw=(1.0 if a_d%90==0 else 0.5))
        for a_d,txt,c in [(90,'N',RED_HI),(0,'E',TEXT_MID),
                           (-90,'S',TEXT_MID),(180,'W',TEXT_MID)]:
            a=np.radians(a_d)
            ax.text(1.22*np.cos(a),1.22*np.sin(a),txt,color=c,
                    ha='center',va='center',fontsize=5.5,family='monospace',fontweight='bold')
    else:
        ax.add_patch(Arc((0,0),2.12,2.12,angle=0,theta1=-210,theta2=30,
                         color=glow_col,lw=4,alpha=0.08))
        ax.add_patch(Arc((0,0),2.0,2.0,angle=0,theta1=-210,theta2=30,
                         color=glow_col,lw=1.5,alpha=0.70))
        for f in np.linspace(0,1,7):
            a=np.radians(-210+f*240)
            ax.plot([0.76*np.cos(a),0.94*np.cos(a)],
                    [0.76*np.sin(a),0.94*np.sin(a)],color=BORDER_HI,lw=0.8)
    ax.text(0,-0.62,label,color=TEXT_MID,ha='center',fontsize=5,family='monospace')
    # Center hub glow
    ax.add_patch(Circle((0,0),0.12,color=glow_col,alpha=0.25,linewidth=0))
    ax.add_patch(Circle((0,0),0.08,color=glow_col,alpha=0.90))
    # Needle glow + crisp
    nglow,=ax.plot([],[],color=glow_col,lw=4.5,alpha=0.15,solid_capstyle='round')
    needle,=ax.plot([],[],color=glow_col,lw=1.8,solid_capstyle='round',zorder=5)
    val_t=ax.text(0,-0.84,'0',color=glow_col,ha='center',fontsize=7,
                  family='monospace',fontweight='bold')
    return nglow,needle,val_t

ax_spd=ax_main.inset_axes([0.695,0.026,0.116,0.222])
ax_hdg=ax_main.inset_axes([0.820,0.026,0.116,0.222])
spd_glow,spd_needle,spd_val=_make_gauge(ax_spd,'SPD m/s',full_circle=False)
hdg_glow,hdg_needle,hdg_val=_make_gauge(ax_hdg,'HDG °',  full_circle=True)

# ═══════════════════════════════════════════════════════════════════
#  PPI RADAR PANEL
# ═══════════════════════════════════════════════════════════════════
ax_radar.set_xlim(0,PPI_N); ax_radar.set_ylim(0,PPI_N)
ax_radar.set_aspect('equal')
ax_radar.set_title("[ RADAR SWEEP ]", color=LP_TEXT,
                    fontsize=7.5,pad=4,family='monospace',fontweight='bold')
ax_radar.axis('off')
_style_light(ax_radar)   # re-apply after axis('off') clears the patch

ppi_im=ax_radar.imshow(ppi_img,origin='lower',cmap=_ppi_cmap,
                        vmin=0,vmax=1,zorder=2,extent=(0,PPI_N,0,PPI_N))
# Range rings + labels
for frac,lbl in ((0.25,'2.5'),(0.50,'5.0'),(0.75,'7.5'),(1.0,'10')):
    ax_radar.add_patch(Circle((PPI_R,PPI_R),PPI_R*frac,
                               fill=False,ec=LP_GRID,lw=0.9,zorder=3))
    ax_radar.text(PPI_R+PPI_R*frac*0.71,PPI_R+PPI_R*frac*0.71,
                  f'{lbl}m',color=LP_DIM,fontsize=4,
                  ha='center',va='center',family='monospace',zorder=3)
# Cardinal bearing lines
for a in np.linspace(0,np.pi,4,endpoint=False):
    ax_radar.plot([PPI_R+PPI_R*np.cos(a),PPI_R-PPI_R*np.cos(a)],
                  [PPI_R+PPI_R*np.sin(a),PPI_R-PPI_R*np.sin(a)],
                  color=LP_GRID,lw=0.6,zorder=3)
# Crosshair
ax_radar.plot([PPI_R-3,PPI_R+3],[PPI_R,PPI_R],color=LP_ACC,lw=0.7,alpha=0.55,zorder=4)
ax_radar.plot([PPI_R,PPI_R],[PPI_R-3,PPI_R+3],color=LP_ACC,lw=0.7,alpha=0.55,zorder=4)

# ARPA vessel tracks
arpa_vecs =[ax_radar.plot([],[],color=AMBER,lw=1.2,alpha=0.85,zorder=5)[0] for _ in DYNAMIC]
arpa_dots =[ax_radar.plot([],[],'o',ms=4,zorder=6)[0]                      for _ in DYNAMIC]
# Sweep — double draw
ppi_sweep_glow,=ax_radar.plot([],[],color=PHOSPHOR,lw=6,alpha=0.07,zorder=5)
ppi_sweep_line,=ax_radar.plot([],[],color=PHOSPHOR,lw=1.1,alpha=0.92,zorder=5)
ppi_vessel,    =ax_radar.plot([PPI_R],[PPI_R],'^',ms=5,color=CYAN_HI,zorder=6)

# ═══════════════════════════════════════════════════════════════════
#  SLAM PANEL
# ═══════════════════════════════════════════════════════════════════
ax_slam.set_xlim(0,MAP_W); ax_slam.set_ylim(0,MAP_H)
ax_slam.set_aspect('auto')
ax_slam.set_title("[ OCCUPANCY MAP ]",color=LP_TEXT,
                   fontsize=7.5,pad=4,family='monospace',fontweight='bold')
ax_slam.set_xlabel("East (m)", fontsize=6, color=LP_DIM)
ax_slam.set_ylabel("North (m)", fontsize=6, color=LP_DIM)
# 5 m grid — like standard robot mapping software
for _xi in range(0,MAP_W+1,5):
    ax_slam.axvline(_xi,color=LP_GRID,lw=0.5,alpha=0.8,zorder=1)
for _yi in range(0,MAP_H+1,5):
    ax_slam.axhline(_yi,color=LP_GRID,lw=0.5,alpha=0.8,zorder=1)
slam_im  =ax_slam.imshow(slam_map.T,origin='lower',extent=(0,MAP_W,0,MAP_H),
                          cmap=_slam_cmap,vmin=0,vmax=1,zorder=2,alpha=1.0)
slam_dot ,=ax_slam.plot([],[],'o',ms=7,color='#E03020',zorder=5,
                         markeredgecolor='white',markeredgewidth=1.2)
slam_traj,=ax_slam.plot([],[],color=LP_ACC,lw=1.4,alpha=0.80,zorder=3)
slam_cov =ax_slam.text(0.02,0.97,'',transform=ax_slam.transAxes,
                        color=LP_TEXT,fontsize=6.5,va='top',family='monospace',
                        bbox=dict(facecolor='white',alpha=0.7,edgecolor='none',pad=2))
# Legend: free / unknown / occupied
for _lx, _lc, _lt in ((0.70,'#FAFAFA','FREE'), (0.80,'#9AA4AE','UNKNOWN'), (0.90,'#080C10','OCCUPIED')):
    ax_slam.add_patch(Rectangle((_lx,0.945),0.03,0.038,
                                 transform=ax_slam.transAxes,
                                 facecolor=_lc,edgecolor=LP_EDGE,lw=0.6,
                                 clip_on=False,zorder=10))
    ax_slam.text(_lx+0.038,0.964,_lt,transform=ax_slam.transAxes,
                  color=LP_TEXT,fontsize=5.0,va='center',family='monospace')

# ═══════════════════════════════════════════════════════════════════
#  SENSOR STATUS + AIS THREATS PANEL
# ═══════════════════════════════════════════════════════════════════
ax_sens.set_xlim(0,1); ax_sens.set_ylim(0,1); ax_sens.axis('off')
_style_light(ax_sens)   # restore patch after axis('off')
ax_sens.set_title("[ SENSOR STATUS  ·  VESSEL TRACKING ]",color=LP_TEXT,
                   fontsize=7.5,pad=4,family='monospace',fontweight='bold')

def _srow(ax,y,label):
    ax.text(0.03,y,label,color=LP_TEXT,fontsize=7,va='center',family='monospace')
    dot=ax.scatter([0.52],[y],s=72,color=LP_DIM,zorder=5)
    ax.add_patch(Rectangle((0.57,y-0.028),0.37,0.056,color=LP_GRID,zorder=3))
    bar=Rectangle((0.57,y-0.028),0.0,0.056,color=LP_DIM,zorder=4)
    ax.add_patch(bar)
    vt=ax.text(0.97,y,'—',color=LP_TEXT,fontsize=6,va='center',
               ha='right',family='monospace')
    return dot,bar,vt

gps_dot,gps_bar,gps_val=_srow(ax_sens,0.90,"[GPS]  GPS")
imu_dot,imu_bar,imu_val=_srow(ax_sens,0.80,"[IMU]  IMU")
lid_dot,lid_bar,lid_val=_srow(ax_sens,0.70,"[LDR] LIDAR")
ekf_dot,ekf_bar,ekf_val=_srow(ax_sens,0.60,"[EKF]  FUSED")

ax_sens.axhline(0.52,color=LP_GRID,lw=0.9)
ax_sens.text(0.50,0.486,'NEARBY VESSEL TRACKING',
             color=LP_DIM,fontsize=5.5,ha='center',family='monospace',
             fontweight='bold')
ais_td,ais_tt=[],[]
for i,obs in enumerate(DYNAMIC):
    y=0.40-i*0.13
    ax_sens.text(0.03,y,obs['name'],color=LP_TEXT,fontsize=7,
                 va='center',family='monospace',fontweight='bold')
    d=ax_sens.scatter([0.40],[y],s=72,color=LP_DIM,zorder=5)
    t=ax_sens.text(0.48,y,'CPA:--  T:--',color=LP_TEXT,fontsize=5.5,
                   va='center',family='monospace')
    ais_td.append(d); ais_tt.append(t)
ax_sens.axhline(0.065,color=LP_GRID,lw=0.9)
ax_sens.text(0.50,0.028,'SENSOR QUALITY',color=LP_DIM,
             fontsize=5,ha='center',family='monospace')

# ═══════════════════════════════════════════════════════════════════
#  PERFORMANCE / NIS PANEL
# ═══════════════════════════════════════════════════════════════════
ax_perf.set_title("[ EKF PERFORMANCE  ·  NIS MONITOR ]",color=LP_TEXT,
                   fontsize=7.5,pad=4,family='monospace',fontweight='bold')
ax_perf.set_facecolor(LP_BG)
ax_perf.set_xlim(0,80); ax_perf.set_ylim(0,16)
ax_perf.tick_params(labelsize=5.5,colors=LP_DIM,labelcolor=LP_DIM)
ax_perf.set_ylabel('NIS',color=LP_DIM,fontsize=7)
ax_perf.axhline(CHI2_95,color=RED_HI,lw=1.0,ls='--',alpha=0.70,label='χ²(2) 95%')
ax_perf.axhline(CHI2_95/2,color=AMBER,lw=0.6,ls=':',alpha=0.55)
nis_line, =ax_perf.plot([],[],color=LP_ACC,lw=1.3,alpha=0.90)
rmse_line,=ax_perf.plot([],[],color='#A020A0',lw=1.0,alpha=0.80,label='RMSE×2(m)')
ax_perf.legend(loc='upper right',facecolor='white',edgecolor=LP_GRID,
               labelcolor=LP_TEXT,fontsize=5.5,fancybox=False)
nis_stat=ax_perf.text(0.02,0.06,'',transform=ax_perf.transAxes,
                       color=LP_TEXT,fontsize=6,va='bottom',family='monospace')

# State bars inset
ax_state=ax_perf.inset_axes([0.0,0.60,1.0,0.38])
ax_state.set_xlim(0,1); ax_state.set_ylim(0,4.2); ax_state.axis('off')
ax_state.set_facecolor(LP_BG)
ax_state.patch.set_facecolor(LP_BG); ax_state.patch.set_visible(True); ax_state.patch.set_alpha(1.0)
_SL=['x(m)','y(m)','θ(°)','v(m/s)']; _SC=['#1890C0','#1890C0',AMBER,LP_ACC]
s_bars,s_txts=[],[]
for i,(lbl,col) in enumerate(zip(_SL,_SC)):
    y=3.70-i*0.93
    ax_state.text(0.02,y,lbl,color=LP_TEXT,fontsize=6.5,va='center',family='monospace')
    ax_state.add_patch(Rectangle((0.28,y-0.18),0.56,0.36,color=LP_GRID))
    bar=Rectangle((0.28,y-0.18),0.0,0.36,color=col)
    ax_state.add_patch(bar)
    txt=ax_state.text(0.87,y,'0.00',color=LP_TEXT,fontsize=6.5,
                       va='center',ha='right',family='monospace')
    s_bars.append(bar); s_txts.append(txt)
ekf_err=ax_state.text(0.50,0.02,'POS ERR: —',color=LP_ACC,fontsize=6.5,
                       ha='center',va='bottom',family='monospace')

# ═══════════════════════════════════════════════════════════════════
#  BOTTOM STRIP
# ═══════════════════════════════════════════════════════════════════
ax_info.text(0.50,0.70,
    "P=AUTOPILOT   G=GPS_OFF   L=LIDAR_30%   I=IMU_SPIKE   [ / ]=SEA_STATE   DBL-CLICK=WAYPOINT   ARROWS=HELM",
    color='#2A4060',fontsize=6.5,ha='center',va='center',family='monospace',
    transform=ax_info.transAxes)
param_txt=ax_info.text(0.50,0.15,'',color='#1060B8',fontsize=6.5,
                        ha='center',va='bottom',family='monospace',
                        transform=ax_info.transAxes)

# ═══════════════════════════════════════════════════════════════════
#  KEY BINDINGS
# ═══════════════════════════════════════════════════════════════════
_prev_sp=dict(SP)

def _on_key(event):
    k=event.key; pressed.add(k)
    if k=='g':
        SP['gps_offline']=not SP['gps_offline']
        _fire_alert(f"GPS {'OFFLINE' if SP['gps_offline'] else 'ONLINE'}",
                    RED_HI if SP['gps_offline'] else PHOSPHOR)
    elif k=='l':
        SP['lidar_deg']=not SP['lidar_deg']
        _fire_alert(f"LIDAR {'DEGRADED 30%' if SP['lidar_deg'] else 'NOMINAL'}",AMBER)
    elif k=='i':
        SP['imu_spike']=not SP['imu_spike']
        _fire_alert(f"IMU SPIKE {'ON' if SP['imu_spike'] else 'OFF'}",AMBER)
    elif k=='p':
        SP['autopilot']=not SP['autopilot']
        _fire_alert(f"AUTOPILOT {'ENGAGED' if SP['autopilot'] else 'DISENGAGED'}",TEXT_HI)
    elif k==']': SP['sea_state']=min(5,SP['sea_state']+1); _fire_alert(f"SEA STATE: BEAUFORT {SP['sea_state']}",AMBER)
    elif k=='[': SP['sea_state']=max(0,SP['sea_state']-1); _fire_alert(f"SEA STATE: BEAUFORT {SP['sea_state']}",PHOSPHOR)

fig.canvas.mpl_connect('key_press_event',   _on_key)
fig.canvas.mpl_connect('key_release_event', lambda e: pressed.discard(e.key))

def _on_click(event):
    if event.inaxes==ax_main and event.dblclick and event.button==1:
        _waypoint[0]=[event.xdata,event.ydata]
        _fire_alert(f"WAYPOINT SET  ({event.xdata:.1f}, {event.ydata:.1f})",AMBER)
fig.canvas.mpl_connect('button_press_event',_on_click)

# ═══════════════════════════════════════════════════════════════════
#  ANIMATION
# ═══════════════════════════════════════════════════════════════════
_prev_dwa=False
_prev_ag =False
_popup_open=[True]              # popup state flag
_lidar_cache=[None]             # cached lidar ranges (updated every 2 frames)
_dwa_cache  =[1.0, 0.0, -99]   # [v_sc, om_del, last_frame]

def animate(frame):
    global wave_phase,gps_timer,gps_locked,radar_ang,reticle_angle
    global _prev_dwa,_prev_ag

    t=frame*DT

    # 1. Controls / autopilot
    if SP['autopilot'] and _waypoint[0] is not None:
        own_vx=velocity[0]*np.cos(pose[2]); own_vy=velocity[0]*np.sin(pose[2])
        pre_cpa=[]
        for obs in DYNAMIC:
            cd,tc=compute_cpa(pose[0],pose[1],own_vx,own_vy,
                               obs['pos'][0],obs['pos'][1],obs['vel'][0],obs['vel'][1])
            pre_cpa.append((cd,tc,threat_level(cd,tc),colregs_rule(pose[2],pose[:2],obs['pos'])))
        vc,oc=autopilot_cmd(pose,_waypoint[0],pre_cpa,velocity)
        velocity[0]=float(np.clip(vc,0,MAX_SPD))
        velocity[1]=float(np.clip(oc,-MAX_OMG,MAX_OMG))
    else:
        if   'up'    in pressed: velocity[0]=min(MAX_SPD,  velocity[0]+12.0*DT)
        elif 'down'  in pressed: velocity[0]=max(0.0,      velocity[0]-14.0*DT)
        else:                    velocity[0]*=0.970          # coast drag
        if   'left'  in pressed: velocity[1]=max(-MAX_OMG, velocity[1]-14.0*DT)
        elif 'right' in pressed: velocity[1]=min(MAX_OMG,  velocity[1]+14.0*DT)
        else:                    velocity[1]*=0.78
        # Decay factors approximate first-order hydrodynamic damping:
        # tau_surge = -DT/ln(0.970) ≈ 1.6 s,  tau_yaw = -DT/ln(0.78) ≈ 0.20 s

    # 2. LiDAR  (cached every 3 frames — vectorised cast is fast; 3-frame cadence is imperceptible)
    if frame % 3 == 0 or _lidar_cache[0] is None:
        _lidar_cache[0] = cast_lidar(pose)
    ranges = _lidar_cache[0]

    # 3. DWA  (cached every 3 frames — cuts rollout cost by 2/3)
    thdg=pose[2]
    if _waypoint[0] is not None:
        dx,dy=_waypoint[0][0]-pose[0],_waypoint[0][1]-pose[1]
        thdg=np.arctan2(dy,dx)
    if frame % 3 == 0 or _dwa_cache[2] < 0:
        _dwa_cache[0],_dwa_cache[1] = dwa(pose,velocity,ranges,thdg)
        _dwa_cache[2] = frame
    v_sc,om_del = _dwa_cache[0],_dwa_cache[1]
    eff_v=velocity[0]*v_sc; eff_omg=velocity[1]+om_del
    dwa_active=(v_sc<0.99)
    if dwa_active and not _prev_dwa:
        _fire_alert("DWA OBSTACLE AVOIDANCE ENGAGED",AMBER)
    _prev_dwa=dwa_active

    # 4. IMU
    iv,iw=imu_reading(eff_v,eff_omg,t)

    # 5. EKF  (IMU drives predict only — see EKF.update_imu comment)
    ekf.predict(iv,iw,DT)

    # 6. GPS
    gps_timer+=1; gps_locked=False
    if gps_timer>=GPS_EVERY:
        gps_timer=0
        gps=gps_reading(pose)
        if gps is not None:
            gx,gy,is_mp=gps
            ekf.update_gps(gx,gy)
            if is_mp:
                gps_mp_xs.append(gx); gps_mp_ys.append(gy)
                _fire_alert("GPS MULTIPATH DETECTED",AMBER)
            else:
                gps_xs.append(gx); gps_ys.append(gy)
            gps_locked=True

    # 7. Integrate pose
    x,y,th=pose
    pose[0]=float(np.clip(x+eff_v*np.cos(th)*DT,0.5,MAP_W-0.5))
    pose[1]=float(np.clip(y+eff_v*np.sin(th)*DT,0.5,MAP_H-0.5))
    pose[2]=th+eff_omg*DT
    act_x.append(pose[0]); act_y.append(pose[1])
    ep=ekf.pose; est_x.append(ep[0]); est_y.append(ep[1])
    wake.append((pose[0],pose[1]))

    # 8. SLAM  (every 4 frames — map builds smoothly, saves Bresenham trace cost)
    if frame % 4 == 0:
        update_slam(ekf.pose,ranges)

    # 9. AIS vessel motion
    for obs in DYNAMIC:
        obs['pos']+=obs['vel']*DT
        if not(1.5<obs['pos'][0]<MAP_W-1.5): obs['vel'][0]*=-1
        if not(1.5<obs['pos'][1]<MAP_H-1.5): obs['vel'][1]*=-1
        obs['theta']=float(np.arctan2(obs['vel'][1],obs['vel'][0]))

    # 10. CPA / COLREGS
    own_vx=eff_v*np.cos(pose[2]); own_vy=eff_v*np.sin(pose[2])
    cpa_data=[]
    for obs in DYNAMIC:
        cd,tc=compute_cpa(pose[0],pose[1],own_vx,own_vy,
                           obs['pos'][0],obs['pos'][1],obs['vel'][0],obs['vel'][1])
        lv=threat_level(cd,tc); ru=colregs_rule(pose[2],pose[:2],obs['pos'])
        if lv=='red' and tc>0:
            _fire_alert(f"CPA ALERT  {obs['name']}  {cd:.1f}m  T-{tc:.0f}s",RED_HI)
        cpa_data.append((cd,tc,lv,ru))

    # 11. Anti-grounding
    depth=sonar_at(pose[0],pose[1])
    ag_alarm=depth<3.5
    if ag_alarm and not _prev_ag:
        _fire_alert(f"ANTI-GROUNDING  DEPTH {depth:.1f}m",RED_HI)
    _prev_ag=ag_alarm

    # 12. Alert toast
    if _alert_timer[0]>0:
        _alert_timer[0]-=1
        alpha=min(1.0,_alert_timer[0]/12.0)
        col=_alert_colors[0] if _alert_colors else AMBER
        alert_txt.set_alpha(alpha); alert_txt.get_bbox_patch().set_alpha(alpha*0.90)
        alert_txt.set_color(col); alert_txt.get_bbox_patch().set_edgecolor(col)
    elif _alert_queue:
        msg,col=_alert_queue.pop(0)
        _alert_colors[:]=[];  _alert_colors.append(col)
        alert_txt.set_text(msg); alert_txt.set_alpha(1.0)
        alert_txt.set_color(col)
        alert_txt.get_bbox_patch().set_alpha(0.92); alert_txt.get_bbox_patch().set_edgecolor(col)
        _alert_timer[0]=70
    else:
        alert_txt.set_alpha(0.0); alert_txt.get_bbox_patch().set_alpha(0.0)

    # ══════════ VISUALS ═══════════════════════════════════════════

    wave_phase+=0.055
    if frame % 3 == 0:
        ocean_im.set_data(ocean_frame(wave_phase,SP['sea_state']))

    # Own vessel
    bx, by, bth = pose[0], pose[1], pose[2]
    if _HAS_PIL:
        _rot = _cached_rot(_own_spr, _own_rot_cache, bth)
        boat_im.set_data(_rot)
        boat_im.set_extent([bx-_sw, bx+_sw, by-_sw, by+_sw])
    else:
        bv = boat_verts(*pose)
        boat_patch.set_xy(bv); boat_glow.set_xy(bv)
    hx = bx + 2.2*np.cos(bth); hy = by + 2.2*np.sin(bth)
    hdg_line.set_data([bx, hx], [by, hy])
    hdg_glow.set_data([bx, hx], [by, hy])

    # V-shaped Kelvin wake (proportional to speed)
    spd_frac = float(np.clip(velocity[0]/MAX_SPD, 0, 1))
    wake_alpha = spd_frac * 0.55
    wake_len   = spd_frac * 5.5 + 0.5
    wake_ang   = bth + np.pi           # stern direction
    half_v     = np.arcsin(1.0/3.0)    # Kelvin wake universal constant: arcsin(1/3) ≈ 19.47°
    _t         = np.linspace(0, wake_len, 28)
    # Outer arms (spread)
    wlx = bx + _t*np.cos(wake_ang + half_v)
    wly = by + _t*np.sin(wake_ang + half_v)
    wrx = bx + _t*np.cos(wake_ang - half_v)
    wry = by + _t*np.sin(wake_ang - half_v)
    # Inner (narrower) arms for trough detail
    wlx2 = bx + _t*np.cos(wake_ang + half_v*0.45)
    wly2 = by + _t*np.sin(wake_ang + half_v*0.45)
    wrx2 = bx + _t*np.cos(wake_ang - half_v*0.45)
    wry2 = by + _t*np.sin(wake_ang - half_v*0.45)
    for ln, xs, ys, a in ((wake_v_l, wlx, wly, wake_alpha),
                           (wake_v_r, wrx, wry, wake_alpha),
                           (wake_v_l2, wlx2, wly2, wake_alpha*0.55),
                           (wake_v_r2, wrx2, wry2, wake_alpha*0.55)):
        ln.set_data(xs, ys); ln.set_alpha(a)

    # Nav lights
    pp,sp_,mp=nav_light_pos(*pose)
    lp.center=pp; lpg.center=pp; ls.center=sp_; lsg.center=sp_; lm.center=mp

    # AIS vessels
    for i,obs in enumerate(DYNAMIC):
        cd,tc,lv,ru=cpa_data[i]; col=THREAT_CLR[lv]
        ox, oy, oth = obs['pos'][0], obs['pos'][1], obs['theta']
        if _HAS_PIL:
            _arot = _cached_rot(_ais_sprs[i], _ais_rot_caches[i], oth)
            ais_ims[i].set_data(_arot)
            ais_ims[i].set_extent([ox-_sw, ox+_sw, oy-_sw, oy+_sw])
            # Tint the image alpha by threat level
            ais_ims[i].set_alpha(0.90 if lv=='green' else 1.0)
        else:
            bv2 = boat_verts(ox, oy, oth, obs['l'], obs['w'])
            dyn_patches[i].set_xy(bv2); dyn_patches[i].set_edgecolor(col)
            dyn_patches[i].set_linewidth(2.6 if lv!='green' else 1.5)
            dyn_glows[i].set_xy(bv2); dyn_glows[i].set_facecolor(col)
            dyn_glows[i].set_alpha(0.28 if lv=='red' else(0.16 if lv=='amber' else 0.10))
        # Heading vector 8s
        px2=obs['pos'][0]+obs['vel'][0]*8; py2=obs['pos'][1]+obs['vel'][1]*8
        ais_hdg_lines[i].set_data([obs['pos'][0],px2],[obs['pos'][1],py2])
        ais_hdg_lines[i].set_color(col)
        # CPA marker
        if 0<tc<TCPA_WARN and lv in('amber','red'):
            cpa_markers[i].set_data([obs['pos'][0]+obs['vel'][0]*tc],
                                     [obs['pos'][1]+obs['vel'][1]*tc])
            cpa_markers[i].set_color(col)
        else: cpa_markers[i].set_data([],[])
        # Label
        ts=f"T{tc:.0f}s" if 0<tc<99 else "CLEAR"
        dyn_labels[i].set_position((obs['pos'][0],obs['pos'][1]+obs['w']+0.28))
        dyn_labels[i].set_text(f"{obs['name']} {ru}"); dyn_labels[i].set_color(col)
        ais_td[i].set_facecolor(col)
        ais_tt[i].set_text(f"CPA:{cd:.1f}m  {ts}  {ru}"); ais_tt[i].set_color(col)

    # Wake
    if len(wake)>1:
        wx,wy=zip(*wake); wake_sc.set_offsets(np.c_[wx,wy])

    # LiDAR
    ga=pose[2]+LDR_ANGLES
    beam_ex=pose[0]+ranges*np.cos(ga); beam_ey=pose[1]+ranges*np.sin(ga)
    for i in range(LDR_BEAMS):
        r=ranges[i]
        lidar_lines[i].set_data([pose[0],beam_ex[i]],[pose[1],beam_ey[i]])
        if   r<LDR_RANGE*0.35: lidar_lines[i].set_color(RED_HI);  lidar_lines[i].set_alpha(0.40)
        elif r<LDR_RANGE*0.65: lidar_lines[i].set_color(AMBER);   lidar_lines[i].set_alpha(0.24)
        else:                   lidar_lines[i].set_color(PHOSPHOR);lidar_lines[i].set_alpha(0.10)

    # Radar sweep
    radar_ang=(radar_ang+8.5*DT)%(2*np.pi)
    for rl in (radar_line,radar_glow):
        rl.set_data([pose[0],pose[0]+LDR_RANGE*np.cos(radar_ang)],
                    [pose[1],pose[1]+LDR_RANGE*np.sin(radar_ang)])
    s_angs=np.linspace(radar_ang-0.45,radar_ang,28)
    radar_sweep.set_xy([(pose[0],pose[1])]+list(zip(
        pose[0]+LDR_RANGE*0.95*np.cos(s_angs),pose[1]+LDR_RANGE*0.95*np.sin(s_angs))))

    # GPS
    if gps_xs:    gps_sc.set_offsets(np.c_[gps_xs[-30:],gps_ys[-30:]])
    if gps_mp_xs: gps_mp_sc.set_offsets(np.c_[gps_mp_xs[-15:],gps_mp_ys[-15:]])

    traj_act.set_data(act_x[-400:],act_y[-400:])
    traj_est.set_data(est_x[-400:],est_y[-400:])

    ew,eh,ea=ekf.ellipse_params()
    for ell in (cov_ell,cov_glow):
        ell.set_center((ep[0],ep[1]))
        ell.width=max(ew,0.12); ell.height=max(eh,0.12); ell.angle=ea

    # Rotating reticles on detected obstacles
    reticle_angle=(reticle_angle+0.04)%(2*np.pi)
    for ring,ox,oy,r_ret in det_rings:
        det=bool(np.any(np.hypot(beam_ex[::5]-ox,beam_ey[::5]-oy)<0.95))
        ring.set_visible(det)
        if det:
            ring.set_xy(reticle_verts(ox,oy,r_ret,reticle_angle))

    # Waypoint
    if _waypoint[0] is not None:
        wp_dot.set_data([_waypoint[0][0]],[_waypoint[0][1]])
        wp_line.set_data([pose[0],_waypoint[0][0]],[pose[1],_waypoint[0][1]])
        d=np.hypot(_waypoint[0][0]-pose[0],_waypoint[0][1]-pose[1])
        brg=np.degrees(np.arctan2(_waypoint[0][1]-pose[1],_waypoint[0][0]-pose[0]))%360
        wp_txt.set_text(f"WPT  {d:.1f}m  BRG {brg:.0f}°")
        if d<2.0:
            _fire_alert("WAYPOINT REACHED",PHOSPHOR)
            _waypoint[0]=None
    else:
        wp_dot.set_data([],[]); wp_line.set_data([],[])
        wp_txt.set_text("DBL-CLICK: SET WAYPOINT")

    # Gauges
    sf=float(np.clip(abs(eff_v)/MAX_SPD,0,1))
    sa=np.radians(-210+sf*240)
    for n in (spd_needle,spd_glow): n.set_data([0,0.72*np.cos(sa)],[0,0.72*np.sin(sa)])
    spd_val.set_text(f"{abs(eff_v):.2f}")
    hdg_deg=float(np.degrees(pose[2]))%360
    ha=np.radians(hdg_deg)
    for n in (hdg_needle,hdg_glow): n.set_data([0,0.72*np.cos(ha)],[0,0.72*np.sin(ha)])
    hdg_val.set_text(f"{hdg_deg:.0f}°")

    # HUD
    kts=abs(eff_v)*1.944
    hud1.set_text(f"SPD  {abs(eff_v):.2f} m/s   {kts:.1f} kts")
    hud2.set_text(f"POS ({pose[0]:.1f},{pose[1]:.1f})  HDG {hdg_deg:05.1f}°  t={t:.0f}s")
    hud3.set_text(f"SONAR  {depth:.1f}m" + ("  [SHALLOW]" if depth<3.5 else ""))
    dwa_txt.set_text("[ DWA AVOIDANCE ACTIVE ]" if dwa_active else "")

    # Anti-grounding
    if ag_alarm:
        ag_txt.set_text("ANTI-GROUNDING  SHALLOW WATER"); ag_txt.get_bbox_patch().set_alpha(0.92)
    else:
        ag_txt.set_text(""); ag_txt.get_bbox_patch().set_alpha(0.0)

    # Badges
    def _badge(txt,active,label,col):
        txt.set_text(label if active else ''); txt.get_bbox_patch().set_alpha(0.90 if active else 0.0)
    _badge(badge_gps,SP['gps_offline'], "GPS OFF",  RED_HI)
    _badge(badge_ldr,SP['lidar_deg'],   "LDR 30%",  ORANGE_HI)
    _badge(badge_ap, SP['autopilot'],   "AUTOPLT",  TEXT_HI)

    param_txt.set_text(
        f"SEA: B{SP['sea_state']}   "
        f"GPS: {'OFF' if SP['gps_offline'] else 'ON'}   "
        f"LIDAR: {'30%' if SP['lidar_deg'] else '100%'}   "
        f"IMU: {'SPIKE' if SP['imu_spike'] else 'OK'}   "
        f"AUTOPILOT: {'ON' if SP['autopilot'] else 'OFF'}   "
        f"COVERAGE: {np.mean(slam_map>0.12)*100:.0f}%"
    )

    # PPI + ARPA
    update_ppi(pose,ranges,radar_ang)
    ppi_im.set_data(ppi_img)
    for sl in (ppi_sweep_line,ppi_sweep_glow):
        sl.set_data([PPI_R,PPI_R+PPI_R*np.cos(radar_ang)],
                    [PPI_R,PPI_R+PPI_R*np.sin(radar_ang)])
    for i,obs in enumerate(DYNAMIC):
        rx=obs['pos'][0]-pose[0]; ry=obs['pos'][1]-pose[1]
        ppx=PPI_R+rx*PPI_SCALE; ppy=PPI_R+ry*PPI_SCALE
        col=THREAT_CLR[cpa_data[i][2]]
        if 1<ppx<PPI_N-1 and 1<ppy<PPI_N-1:
            arpa_dots[i].set_data([ppx],[ppy]); arpa_dots[i].set_color(col)
            t6x=ppx+obs['vel'][0]*360*PPI_SCALE; t6y=ppy+obs['vel'][1]*360*PPI_SCALE
            arpa_vecs[i].set_data([ppx,t6x],[ppy,t6y]); arpa_vecs[i].set_color(col)
        else:
            arpa_dots[i].set_data([],[]); arpa_vecs[i].set_data([],[])

    # SLAM — blend occupancy with visited mask
    # Unvisited cells → 0.5 (unknown/gray), visited → actual slam_map value
    _sd = np.where(_slam_visited, slam_map, 0.5)
    if _HAS_SCIPY: _sd = gaussian_filter(_sd, sigma=0.35)
    slam_im.set_data(_sd.T)
    slam_dot.set_data([pose[0]],[pose[1]])
    slam_traj.set_data(act_x[-250:],act_y[-250:])
    cov=float(np.mean(_slam_visited))
    slam_cov.set_text(f"MAPPED  {cov*100:.0f}%   |   OBS {np.mean(slam_map>0.25)*100:.0f}%")

    # Sensor status
    def _upd(dot,bar,vtxt,active,q,lbl):
        col=LP_ACC if active else RED_HI
        dot.set_facecolor(col); bar.set_width(float(np.clip(q,0,1))*0.37)
        bar.set_facecolor(col); vtxt.set_text(lbl)
    gps_q=1.0 if gps_locked else max(0.0,1.0-gps_timer/GPS_EVERY)
    _upd(gps_dot,gps_bar,gps_val,gps_locked and not SP['gps_offline'],gps_q,
         "LOCK" if gps_locked else("OFF" if SP['gps_offline'] else "SRCH"))
    ie=abs(iv-eff_v)
    _upd(imu_dot,imu_bar,imu_val,True,max(0,1-ie*12),f"d{ie:.3f}")
    lhf=float(np.mean(ranges<LDR_RANGE*0.97*(0.30 if SP['lidar_deg'] else 1.0)))
    _upd(lid_dot,lid_bar,lid_val,True,lhf,f"{lhf*100:.0f}%")
    pos_err=float(np.hypot(pose[0]-ep[0],pose[1]-ep[1]))
    _upd(ekf_dot,ekf_bar,ekf_val,True,max(0,1-pos_err/3),f"d{pos_err:.2f}m")

    # NIS + RMSE  (update plot data every 4 frames — deque collection still every frame)
    rmse_log.append(pos_err)
    if frame % 4 == 0 and ekf.nis_log:
        nd = list(ekf.nis_log); nx = list(range(len(nd)))
        nis_line.set_data(nx, nd)
        rmse_line.set_data(list(range(len(rmse_log))), [r*2 for r in rmse_log])
        ln = nd[-1]
        nis_stat.set_text(f"NIS={ln:.2f}  thresh={CHI2_95:.2f}  "
                          f"[{'CONSISTENT' if ln<CHI2_95 else 'INCONSISTENT'}]")
        nis_stat.set_color(LP_ACC if ln < CHI2_95 else RED_HI)

    # State bars
    norms=[ep[0]/MAP_W,ep[1]/MAP_H,(ep[2]%(2*np.pi))/(2*np.pi),abs(eff_v)/MAX_SPD]
    raws=[ep[0],ep[1],np.degrees(ep[2])%360,eff_v]
    fmts=['{:.1f}','{:.1f}','{:.0f}','{:.2f}']
    for i,(sb,st) in enumerate(zip(s_bars,s_txts)):
        sb.set_width(float(np.clip(norms[i],0,1))*0.56)
        st.set_text(fmts[i].format(raws[i]))
    ekf_err.set_text(f"POS ERR: {pos_err:.3f} m")
    ekf_err.set_color(LP_ACC if pos_err<0.8 else(AMBER if pos_err<2.0 else RED_HI))

    return [
        ocean_im, boat_im, boat_patch, boat_glow, hdg_line, hdg_glow,
        wake_v_l, wake_v_r, wake_v_l2, wake_v_r2,
        lp,ls,lm,lpg,lsg,
        traj_act, traj_est, wake_sc, gps_sc, gps_mp_sc,
        cov_ell, cov_glow, radar_line, radar_glow, radar_sweep,
        wp_dot, wp_line,
        *ais_ims,
        slam_im, slam_dot, slam_traj, slam_cov,
        ppi_im, ppi_sweep_line, ppi_sweep_glow, ppi_vessel,
        hud1, hud2, hud3, dwa_txt, wp_txt,
        ag_txt, alert_txt,
        badge_gps, badge_ldr, badge_ap,
        spd_needle, spd_glow, spd_val,
        hdg_needle, hdg_glow, hdg_val,
        nis_line, rmse_line, nis_stat, ekf_err,
        *lidar_lines,
        *[r for r,_,_,_ in det_rings],
        *dyn_patches, *dyn_glows, *dyn_labels,
        *ais_hdg_lines, *cpa_markers,
        *arpa_dots, *arpa_vecs,
        gps_dot,imu_dot,lid_dot,ekf_dot,
        gps_bar,imu_bar,lid_bar,ekf_bar,
        gps_val,imu_val,lid_val,ekf_val,
        *ais_td,*ais_tt,*s_bars,*s_txts,
    ]

# ═══════════════════════════════════════════════════════════════════
#  MANUAL POPUP  (startup — opaque white panel, no axis('off'))
# ═══════════════════════════════════════════════════════════════════
_WH='#F4F7FA'; _WH2='#E2E8F0'; _NV='#1C2A3A'; _BL='#1060B8'; _OR='#C84800'

def _pop_ax(rect, zorder=200, bg=_WH):
    """Create an opaque axes with hidden decorations (NOT axis('off'))."""
    ax=fig.add_axes(rect)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_zorder(zorder)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    ax.set_facecolor(bg)
    ax.patch.set_facecolor(bg); ax.patch.set_alpha(1.0); ax.patch.set_visible(True)
    return ax

ax_popup=_pop_ax([0.10,0.07,0.80,0.87])

# Outer border
ax_popup.add_patch(Rectangle((0,0),1,1,transform=ax_popup.transAxes,
    fill=False,ec='#7A90A8',lw=2.0,clip_on=False))
# Title bar
ax_popup.add_patch(Rectangle((0,0.885),1,0.115,transform=ax_popup.transAxes,
    facecolor='#1C2A3A',edgecolor='none',clip_on=False))
ax_popup.text(0.5,0.942,'MARINE ADAS SIMULATOR  —  USER GUIDE',
    transform=ax_popup.transAxes,color='#FFFFFF',fontsize=13,
    ha='center',va='center',family='monospace',fontweight='bold')
ax_popup.text(0.5,0.910,'Real-time maritime vessel simulator demonstrating ADAS '
    'applied to autonomous navigation.',
    transform=ax_popup.transAxes,color='#A8C0D8',fontsize=7.5,
    ha='center',va='center',family='monospace')

# Divider
ax_popup.add_patch(Rectangle((0.02,0.875),0.96,0.001,
    transform=ax_popup.transAxes,facecolor=_BL,edgecolor='none',clip_on=False))

# ── helpers ───────────────────────────────────────────────────
def _ph2(ax,x,y,txt):
    ax.add_patch(Rectangle((x-0.01,y-0.018),0.45,0.038,
        transform=ax.transAxes,facecolor=_WH2,edgecolor='none',clip_on=False))
    ax.text(x,y,txt,transform=ax.transAxes,color=_BL,fontsize=8.5,
            va='center',family='monospace',fontweight='bold')
def _pt2(ax,x,y,txt,col=_NV,fs=7.5):
    ax.text(x,y,txt,transform=ax.transAxes,color=col,fontsize=fs,
            va='top',family='monospace')

C1,C2=0.025,0.515

# ──────────────────────────────────────────────────────────────
# LEFT COLUMN
# ──────────────────────────────────────────────────────────────
_ph2(ax_popup,C1,0.852,'  WHAT IS THIS?')
_pt2(ax_popup,C1,0.826,
    'A Python / matplotlib simulator that runs five ADAS algorithms simultaneously\n'
    'on a virtual vessel — sensing, localising, mapping and autonomously avoiding\n'
    'obstacles in real time. Designed as an interactive portfolio demonstration.')

_ph2(ax_popup,C1,0.720,'  ACTIVE ALGORITHMS')
algos2=[
    ('EKF ','Extended Kalman Filter — fuses GPS + IMU → pose estimate with uncertainty'),
    ('DWA ','Dynamic Window Approach — real-time trajectory rollout obstacle avoidance'),
    ('SLAM','Simultaneous Localisation & Mapping — builds live occupancy grid'),
    ('CPA ','Closest Point of Approach — predicts collision risk with nearby vessels'),
    ('COL ','COLREGS rules 13-16 — HEAD-ON / GIVE WAY / STAND ON logic'),
]
for i,(tag,desc) in enumerate(algos2):
    y=0.688-i*0.060
    ax_popup.text(C1,y,f'[{tag}]',transform=ax_popup.transAxes,
        color=_BL,fontsize=7.5,va='top',family='monospace',fontweight='bold')
    _pt2(ax_popup,C1+0.075,y,desc,fs=7.2)

_ph2(ax_popup,C1,0.358,'  READING THE DISPLAYS')
displays2=[
    ('Nav view (left) ','Real-time position, 360° LiDAR beams, vessel wake & sprites'),
    ('Radar sweep     ','PPI 360° phosphor display with ARPA vessel tracking'),
    ('Occupancy map   ','SLAM grid  —  white=free · grey=unknown · black=obstacle'),
    ('Sensor status   ','GPS / IMU / LiDAR / EKF quality bars + AIS CPA data'),
    ('EKF monitor     ','NIS (χ²) health plot + position RMSE over time'),
]
for i,(lbl,desc) in enumerate(displays2):
    y=0.325-i*0.049
    ax_popup.text(C1,y,lbl,transform=ax_popup.transAxes,
        color=_BL,fontsize=7.2,va='top',family='monospace',fontweight='bold')
    _pt2(ax_popup,C1+0.148,y,'—  '+desc,fs=7.2)

# ──────────────────────────────────────────────────────────────
# RIGHT COLUMN
# ──────────────────────────────────────────────────────────────
_ph2(ax_popup,C2,0.852,'  CONTROLS')
ctrl2=[
    ('↑ / ↓ ','Throttle forward  /  Brake'),
    ('← / → ','Helm port  /  Helm starboard'),
    ('P      ','Autopilot toggle  (double-click map first to set waypoint)'),
    ('G      ','Toggle GPS offline'),
    ('L      ','Toggle LiDAR degraded mode  (30% range)'),
    ('I      ','Inject IMU spike noise'),
    ('[ / ]  ','Sea state  Beaufort 0 – 5'),
    ('Dbl-clk','Set navigation waypoint on map'),
]
for i,(k,d) in enumerate(ctrl2):
    y=0.820-i*0.067
    ax_popup.text(C2,y,k,transform=ax_popup.transAxes,
        color=_OR,fontsize=8.5,va='top',family='monospace',fontweight='bold')
    _pt2(ax_popup,C2+0.115,y,d,fs=7.5)

_ph2(ax_popup,C2,0.295,'  QUICK TIPS')
tips2=[
    'Drive into grey areas — watch the SLAM map grow',
    'Approach rocks to see DWA avoidance auto-engage',
    'Double-click map, then press P for autopilot routing',
    'Press G or L to simulate sensor failure',
]
for i,tip in enumerate(tips2):
    _pt2(ax_popup,C2,0.266-i*0.050,f'→  {tip}',fs=7.2)

# Footer strip — clearly below all content
ax_popup.add_patch(Rectangle((0,0),1,0.100,transform=ax_popup.transAxes,
    facecolor=_WH2,edgecolor='none',clip_on=False))
ax_popup.add_patch(Rectangle((0,0.098),1,0.002,transform=ax_popup.transAxes,
    facecolor='#BFC8D8',edgecolor='none',clip_on=False))
ax_popup.text(0.5,0.028,'Press  ENTER  or  ESC  to begin',
    transform=ax_popup.transAxes,color='#6080A0',fontsize=7.5,
    ha='center',va='center',family='monospace')

# ── START button — centred in footer ──────────────────────────
ax_btn_close=_pop_ax([0.370,0.108,0.165,0.043],zorder=210,bg='#1C2A3A')
for sp in ax_btn_close.spines.values(): sp.set_edgecolor(_BL); sp.set_linewidth(2.0); sp.set_visible(True)
btn_close=Button(ax_btn_close,'  START SIMULATION  ',color='#1C2A3A',hovercolor='#2A4060')
btn_close.label.set_color('#FFFFFF'); btn_close.label.set_family('monospace')
btn_close.label.set_fontsize(9.5); btn_close.label.set_fontweight('bold')

# ═══════════════════════════════════════════════════════════════════
#  QUICK-REFERENCE CARD  (corner card, toggled by ? button)
# ═══════════════════════════════════════════════════════════════════
ax_qref=_pop_ax([0.032,0.052,0.190,0.370],zorder=190,bg='#F8FAFB')
ax_qref.add_patch(Rectangle((0,0),1,1,transform=ax_qref.transAxes,
    fill=False,ec='#7A90A8',lw=1.4,clip_on=False))
ax_qref.add_patch(Rectangle((0,0.888),1,0.112,transform=ax_qref.transAxes,
    facecolor='#1C2A3A',edgecolor='none',clip_on=False))
ax_qref.text(0.5,0.944,'QUICK REFERENCE',transform=ax_qref.transAxes,
    color='white',fontsize=8,ha='center',va='center',family='monospace',fontweight='bold')

qkeys=[
    ('↑ / ↓','Throttle / Brake'),
    ('← / →','Port / Starboard'),
    ('P     ','Autopilot toggle'),
    ('G     ','GPS offline'),
    ('L     ','LiDAR degraded'),
    ('I     ','IMU spike'),
    ('[ / ]','Sea state ±'),
    ('Dbl   ','Set waypoint'),
    ('ENTER ','Close manual'),
]
for i,(k,d) in enumerate(qkeys):
    y=0.855-i*0.093
    ax_qref.text(0.04,y,k,transform=ax_qref.transAxes,
        color=_OR,fontsize=7,va='top',family='monospace',fontweight='bold')
    ax_qref.text(0.44,y,d,transform=ax_qref.transAxes,
        color=_NV,fontsize=7,va='top',family='monospace')
ax_qref.set_visible(False)   # hidden until ? pressed

# ── ? button (bottom-right corner of info strip) ───────────────
ax_btn_help=_pop_ax([0.948,0.009,0.043,0.030],zorder=190,bg='#1C2A3A')
for sp in ax_btn_help.spines.values(): sp.set_edgecolor('#7A90A8'); sp.set_linewidth(1.0); sp.set_visible(True)
btn_help=Button(ax_btn_help,'?',color='#1C2A3A',hovercolor='#2A4060')
btn_help.label.set_color('#A8C8E0'); btn_help.label.set_family('monospace')
btn_help.label.set_fontsize(11); btn_help.label.set_fontweight('bold')
ax_btn_help.set_visible(False)   # shown once main popup is closed

# ── event handlers ─────────────────────────────────────────────
_qref_open=[False]

def _close_popup(event=None):
    _popup_open[0]=False
    ax_popup.set_visible(False)
    ax_btn_close.set_visible(False)
    ax_btn_help.set_visible(True)
    ani.event_source.start()

def _open_popup(event=None):
    _popup_open[0]=True
    _qref_open[0]=False
    ax_qref.set_visible(False)
    ani.event_source.stop()
    ax_popup.set_visible(True)
    ax_btn_close.set_visible(True)
    ax_btn_help.set_visible(False)
    fig.canvas.draw_idle()

def _toggle_qref(event=None):
    _qref_open[0]=not _qref_open[0]
    ax_qref.set_visible(_qref_open[0])
    fig.canvas.draw_idle()

btn_close.on_clicked(_close_popup)
btn_help.on_clicked(_toggle_qref)

def _on_key_popup(event):
    if event.key in ('enter','escape'):
        if _popup_open[0]:   _close_popup()
        elif _qref_open[0]:  _toggle_qref()
fig.canvas.mpl_connect('key_press_event',_on_key_popup)

# ═══════════════════════════════════════════════════════════════════
#  LAUNCH
# ═══════════════════════════════════════════════════════════════════
ani = animation.FuncAnimation(
    fig, animate,
    frames=int(SIM_TIME/DT),
    interval=int(DT*1000),
    blit=False,         # popup overlay requires full redraw per frame
)
ani.event_source.stop() # start paused — popup is showing; _close_popup starts it
plt.show()
