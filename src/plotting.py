"""Shared figure style and colours so every chart in the report looks consistent."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from config import FIG_DIR  # noqa: E402

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
NEUTRAL = "#8a8985"

MODEL_COLORS = {
    "Logistic Regression": "#6250d6",
    "SVM (RBF)": "#eb6834",
    "Random Forest": "#1baf7a",
    "Majority baseline": NEUTRAL,
}
CLASS_COLORS = {0: "#2a78d6", 1: "#e34948"}
CLASS_NAMES = {0: "No disease", 1: "Disease"}
SITE_COLORS = {
    "Cleveland": "#2a78d6",
    "Hungary": "#eb6834",
    "Switzerland": "#1baf7a",
    "Long Beach VA": "#eda100",
}

# Diverging (blue <-> grey <-> red) for signed values such as correlations/coefficients.
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "blue_grey_red", ["#104281", "#2a78d6", "#f0efec", "#e34948", "#8c1d1d"]
)
# Sequential single-hue blue for magnitudes such as confusion-matrix counts.
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "blues", ["#f5f9fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def set_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.edgecolor": TEXT_SECONDARY,
        "axes.labelcolor": TEXT_PRIMARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "text.color": TEXT_PRIMARY,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "patch.edgecolor": "white",
        "patch.linewidth": 1.0,
    })


def savefig(fig, name):
    """Save to figures/<name>.png (300 dpi) and close the figure."""
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


set_style()
