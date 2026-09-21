"""
Shared style for every figure in the manuscript.

Palette follows a 60-30-10 split:

    60   warm grey    #918E8E   ground, structure, anything not being pointed at
    30   deep petrol  #1D5C6E   the second voice: models, cool side of diverging
    10   burnt rust   #C8552B   reserved for the thing the panel is about

Fixed semantics override the aesthetic where the two disagree, so that a colour
means the same thing in every figure of the paper:

    m=0  grey        the majority class, deliberately recessive
    m=1  rust        the paper's subject
    m=2  petrol
    ON   rust        warm side of the diverging map
    OFF  petrol      cool side
    data grey fill   model petrol line
    binocular petrol / monocular grey

Import and call `apply()` once at the top of a figure script.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- 60 -------------------------------------------------------------------
GREY      = '#918E8E'
GREY_MID  = '#B7B4B4'
GREY_PALE = '#D8D5D4'
GREY_WASH = '#EFEDEC'
INK       = '#3A3838'      # text and rules; not black, to sit with the greys

# --- 30 -------------------------------------------------------------------
PETROL      = '#1D5C6E'
PETROL_MID  = '#3E8095'
PETROL_PALE = '#A9C6D0'

# --- 10 -------------------------------------------------------------------
RUST      = '#C8552B'
RUST_MID  = '#DD7F5C'
RUST_PALE = '#EFC3B1'

PAPER = '#FFFFFF'

# --- fixed semantics ------------------------------------------------------
ORDER = {0: GREY, 1: RUST, 2: PETROL}
ORDER_LABEL = {0: r'$m{=}0$', 1: r'$m{=}1$', 2: r'$m{=}2$'}

ON, OFF = RUST, PETROL
DATA, MODEL, RESIDUAL = GREY, PETROL, RUST
BINOCULAR, MONOCULAR = PETROL, GREY

# Diverging map for signed receptive fields, built from the two accents so it
# belongs to the palette rather than being imported from elsewhere.
RF_CMAP = LinearSegmentedColormap.from_list(
    'rf_signed',
    [PETROL, PETROL_MID, PETROL_PALE, PAPER, RUST_PALE, RUST_MID, RUST],
    N=256,
)

# --- type scale -----------------------------------------------------------
# Sized for legibility at print, not for fitting the most panels on a page.
FS_TICK, FS_LABEL, FS_TITLE, FS_PANEL, FS_ANNOT = 10, 11, 11.5, 13, 9.5

# --- page widths (mm -> inches) -------------------------------------------
MM = 1 / 25.4
W_SINGLE, W_THREEQ, W_FULL = 90 * MM, 140 * MM, 183 * MM


def apply():
    """Set rcParams. Call once per figure script."""
    mpl.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': FS_TICK,
        'axes.labelsize': FS_LABEL, 'axes.titlesize': FS_TITLE,
        'xtick.labelsize': FS_TICK, 'ytick.labelsize': FS_TICK,
        'legend.fontsize': FS_ANNOT,

        'axes.edgecolor': INK, 'axes.labelcolor': INK,
        'text.color': INK, 'xtick.color': INK, 'ytick.color': INK,
        'axes.linewidth': 0.9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': False,
        'axes.facecolor': 'none', 'figure.facecolor': PAPER,
        'axes.titlelocation': 'left', 'axes.titlepad': 6,

        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
        'xtick.major.width': 0.9, 'ytick.major.width': 0.9,

        'lines.linewidth': 1.6, 'lines.solid_capstyle': 'round',
        'patch.linewidth': 0.9,

        # no idle whitespace
        'figure.constrained_layout.use': False,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
        'savefig.transparent': False,
    })


def panel_label(ax, letter, dx=-0.085, dy=1.06):
    """Bold panel letter, positioned in axes coordinates."""
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=FS_PANEL, fontweight='bold', va='top', ha='left', color=INK)


def strip(ax):
    """Remove all axis furniture; for schematic panels."""
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def save(fig, stem, outdir='.'):
    """Write vector PDF and SVG plus a PNG for drafts."""
    from pathlib import Path
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(out / f'{stem}.{ext}', dpi=400 if ext == 'png' else None)
    print(f'wrote {stem}.pdf / .svg / .png to {out}')
