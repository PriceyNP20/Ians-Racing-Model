"""Ian Racing Model MVP."""

__all__ = ["__version__"]

__version__ = "0.1.0"

try:
    from . import ui as _ui
    from .table_styles import install_streamlit_table_styles, picks_tracker_style as _picks_tracker_style

    _ui.picks_tracker_style = _picks_tracker_style
    install_streamlit_table_styles()
except Exception:
    pass
