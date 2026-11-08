"""HealthMonitor: runs the static checks on its own thread -- at start, when
asked (config change, an operator's re-check, a part of the app failing) and
every `interval` seconds -- and merges them with the live runtime problems.

Never on the ASR, render or event-drain path: the checks run here only, and
readers (`problems()`, called from the web UI's 10 Hz state snapshot) get the
last result list, swapped atomically, plus runtime problems built from plain
attribute reads.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from narrator.config.schema import RootConfig
from narrator.health.checks import Environment, RuntimeState, runtime_problems, static_problems
from narrator.health.problems import Problem, Severity, sort_problems

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_SECONDS = 30.0


class HealthMonitor:
    def __init__(
        self,
        get_config: Callable[[], RootConfig | None],
        get_runtime: Callable[[], RuntimeState],
        env: Environment | None = None,
        interval: float = DEFAULT_INTERVAL_SECONDS,
    ) -> None:
        self._get_config = get_config
        self._get_runtime = get_runtime
        self.env = env if env is not None else Environment()
        self._interval = interval
        self._static: list[Problem] = []
        self._wake = threading.Event()
        self._invalidate = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._logged: set[str] = set()
        self._check_lock = threading.Lock()
        self.checks_run = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="health", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def request_check(self, invalidate: bool = False) -> None:
        """Re-run the static checks soon (on the monitor thread).
        `invalidate` also drops the cached probes (Vulkan devices, build info)."""
        if invalidate:
            self._invalidate = True
        self._wake.set()

    def check_now(self, invalidate: bool = True) -> list[Problem]:
        """Run the static checks in the caller's thread (tests, the operator's
        re-check from the web executor) and return every problem."""
        if invalidate:
            self.env.invalidate()
        self._run_checks()
        return self.problems()

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._invalidate:
                self._invalidate = False
                self.env.invalidate()
            self._run_checks()
            self._wake.wait(self._interval)
            self._wake.clear()

    def _run_checks(self) -> None:
        # Serialized (the monitor thread and check_now from a fix can run at
        # once), and a result is kept only if the config did not change
        # while it was computed -- otherwise a slow pass on the old config
        # (the Vulkan probe takes ~1 s) could overwrite a newer one.
        with self._check_lock:
            for _ in range(3):
                config = self._get_config()
                if config is None:
                    return
                try:
                    runtime = self._get_runtime()
                except Exception:
                    runtime = None
                try:
                    found = static_problems(config, self.env, runtime)
                except Exception:
                    logger.exception("setup checks failed")
                    return
                if self._get_config() is config:
                    break
            self._static = found  # one reference swap: readers never see a partial list
            self.checks_run += 1
            self._log_new(found)

    def _log_new(self, found: list[Problem]) -> None:
        """Log each problem once when it appears (the journal is the record
        on a headless box), and again if it comes back after clearing."""
        ids = {p.id for p in found}
        for problem in found:
            if problem.id not in self._logged and problem.severity != Severity.INFO:
                level = logging.ERROR if problem.severity == Severity.ERROR else logging.WARNING
                logger.log(level, "setup check: %s", problem.message)
        self._logged = ids

    def problems(self) -> list[Problem]:
        """Runtime problems (live) + the last static results, sorted."""
        config = self._get_config()
        live: list[Problem] = []
        if config is not None:
            try:
                live = runtime_problems(self._get_runtime(), config, self.env)
            except Exception:
                logger.debug("runtime problems failed", exc_info=True)
        return sort_problems(live + list(self._static))
