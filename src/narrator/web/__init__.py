"""Browser admin UI for narrator (replaces the Tk GUI).

An aiohttp server on its own thread and asyncio loop inside the app process
(server.py), loopback-only by default and guarded by Host/Origin checks plus a
a password login (security.py). It reads controller state and calls the
operator actions (actions.py, shared with the OSC adapter in osc.py); it never
runs on, or blocks, the ASR/render/NDI threads. The frontend is plain
HTML/JS/CSS in static/, bundled as package data (no build step).
"""
