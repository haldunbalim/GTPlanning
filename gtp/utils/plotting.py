
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib as mpl

colors = ['#5DBDF3', '#FAB84E', 'tab:green', 'tab:red',
          'tab:purple', 'tab:brown', '#CF9BE7', 'tab:cyan']

color_dict = {
    'GTP-H1': colors[1],
    'GTP-H4': colors[1],
    'GTP-H8': colors[1],
    'GTP': colors[1],
    'TD3': colors[5],
    'SAC': colors[6],
    # Ablations
    'Best-of-N': colors[0],
    'H1': colors[5],
    'H2': colors[3],
    'H4': colors[1],
    'H8': colors[0],
    'H12': colors[2],
    'H16': colors[4],
    'Pessimistic': colors[1],
    'Optimistic': colors[0],
}

def set_mpl_params():
    mpl.rcParams.update({
        # ---- FONT ----
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times"],
        "mathtext.fontset": "stix",   # Times-like math
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",

        # ---- aesthetics ----
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
    })

def make_rounded_subplots(
    nrows=1, ncols=1, *,
    figsize=None,
    panel_color="#EBEBEB",
    grid_color="white",
    frame_color="#2B2B2B",
    frame_lw=2.2,
    rounding=0.025,
    grid=True,
    **subplots_kwargs,
):
    """Create a figure with ggplot-style rounded gray panels.
 
    Returns (fig, axes) just like plt.subplots. Plot whatever you want on
    the axes — lines, scatter, fills, images — and everything is clipped
    to the rounded shape automatically (via a draw-time hook).
 
    Parameters
    ----------
    nrows, ncols, figsize, **subplots_kwargs : forwarded to plt.subplots
    panel_color : background color of the panel (default ggplot gray)
    grid_color  : color of the gridlines
    frame_color : color of the rounded frame stroke
    frame_lw    : frame line width
    rounding    : corner radius in axes-fraction units (0.0 = square,
                  0.05 = quite round)
    grid        : whether to show the white gridlines
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **subplots_kwargs)
    axes_arr = np.atleast_1d(axes).ravel()

    bgs = {}  # ax -> background patch (used as the clip path)

    for ax in axes_arr:
        # transparent default background; we'll draw our own rounded one
        ax.set_facecolor("none")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)

        bg = FancyBboxPatch(
            (0, 0), 1, 1,
            boxstyle=f"round,pad=0,rounding_size={rounding}",
            transform=ax.transAxes,
            facecolor=panel_color, edgecolor="none",
            zorder=0, clip_on=False,
        )
        ax.add_patch(bg)

        if grid:
            ax.grid(True, color=grid_color, linewidth=1.0, zorder=1)
            ax.set_axisbelow(True)

        frame = FancyBboxPatch(
            (0, 0), 1, 1,
            boxstyle=f"round,pad=0,rounding_size={rounding}",
            transform=ax.transAxes,
            fill=False, edgecolor=frame_color, linewidth=frame_lw,
            clip_on=False, zorder=10,
        )
        ax.add_patch(frame)

        bgs[ax] = (bg, frame)

    # auto-clip artists added later (so the user can plot freely afterwards)
    def _clip_on_draw(event):
        for ax, (bg, frame) in bgs.items():
            for artist in ax.lines + ax.collections + ax.patches + ax.images:
                if artist is bg or artist is frame:
                    continue
                if artist.get_clip_path() is None:
                    artist.set_clip_path(bg)
            for gl in ax.get_xgridlines() + ax.get_ygridlines():
                if gl.get_clip_path() is None:
                    gl.set_clip_path(bg)

    fig.canvas.mpl_connect("draw_event", _clip_on_draw)
    return fig, axes
