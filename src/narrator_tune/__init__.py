"""narrator-tune: offline tuning for narrator (not part of the app runtime).

Searches the whisper model and decode settings under which narrator's LIVE
captioner best matches a library of reference recordings, by replaying them
through narrator's real decode path (`narrator.asr.streaming_decoder`,
`backend`, `vad`, `stream_sim`, `bench`), which this package imports and
never copies. narrator itself never imports this package: the app, its GUI
and the PyInstaller build run without it.

Entry point: `narrator-tune` (narrator_tune.cli), installed with the
`[tune]` extra (`make configure`).
"""
