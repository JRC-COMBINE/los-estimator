"""Test-wide setup.

Forces the non-interactive Agg matplotlib backend before any test imports
`matplotlib.pyplot`. Every test run here has `show_figures = False`, so no
test needs a GUI backend -- but if one is picked up implicitly (e.g. TkAgg on
Windows), a broken/incomplete local Tcl/Tk install can crash figure creation
with `_tkinter.TclError`, intermittently, deep inside plotting code that has
nothing to do with the test's actual assertions. This must run before any
other import in the test session pulls in `matplotlib.pyplot`.
"""

import matplotlib

matplotlib.use("Agg")
