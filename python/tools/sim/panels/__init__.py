"""
GUI panels of the GPU simulator (ui.py), one module per feature.

Convention: a feature adds ``python/tools/sim/panels/<feature>.py`` exposing
``draw_<feature>_panel(state, ui_state)``, where ``state`` is the ``state.SimulationState`` and
``ui_state`` is ``ui.UIState``. It then adds one call in ``ui.make_callback`` under the
"Feature panels" comment. Panels typically open their own ``imgui.TreeNode``. ``panels.common``
holds shared widget helpers, e.g. ``draw_params``, which renders every public, settable
float/int/bool/Enum attribute of a parameter object with its docstring as tooltip.
"""
