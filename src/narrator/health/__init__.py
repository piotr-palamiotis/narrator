"""Setup checks and the problems the dashboard shows (todo C8).

`checks` holds the checks (pure functions over the config, a fakeable
Environment and a RuntimeState snapshot), `monitor.HealthMonitor` runs them
on its own thread, `problems` defines the items. Nothing here stops the app
or changes the config: a problem offers fixes, the operator applies them
(AppController.apply_fix).
"""
from narrator.health.checks import Environment, RuntimeState
from narrator.health.monitor import HealthMonitor
from narrator.health.problems import Fix, Problem, Severity, Subsystem

__all__ = ["Environment", "Fix", "HealthMonitor", "Problem", "RuntimeState", "Severity", "Subsystem"]
