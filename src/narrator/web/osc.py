"""Optional OSC (UDP) control input: a thin adapter onto web.actions.

Off by default (`web.osc.enabled`), bound to loopback by default. OSC has no
authentication, so anything that can send UDP to the bound address can drive
the captions -- keep it on 127.0.0.1 unless the network is trusted.

Addresses (arguments optional unless noted):

    /narrator/clear                    clear the screen
    /narrator/mode        [s]          "auto" | "manual" | "toggle" (no arg: toggle)
    /narrator/manual-text  s           append an operator caption line
    /narrator/pause       [i|f|T|F|s]  1/0, true/false, "toggle" (no arg: toggle)
    /narrator/recording   [i|f|T|F|s]  same, for session recording

A lone numeric 0 argument on /narrator/clear or a no-arg toggle address is
treated as a button *release* (TouchOSC/vMix buttons send 1 on press, 0 on
release) and ignored, so one press is one action.

The parser is a minimal OSC 1.0 decoder (messages and bundles; i, f, s, T, F,
N, h, d argument types) -- no python-osc dependency needed for this.
"""
from __future__ import annotations

import asyncio
import logging
import struct

from narrator.web import actions

logger = logging.getLogger(__name__)

PREFIX = "/narrator/"


class OscParseError(ValueError):
    pass


def _read_string(data: bytes, offset: int) -> tuple[str, int]:
    end = data.find(b"\0", offset)
    if end < 0:
        raise OscParseError("unterminated OSC string")
    text = data[offset:end].decode("utf-8", errors="replace")
    offset = end + 1
    offset += (-offset) % 4  # strings are null-padded to a multiple of 4
    return text, offset


def parse_packet(data: bytes) -> list[tuple[str, list]]:
    """[(address, args), ...] for one UDP datagram (a message or a bundle)."""
    if data.startswith(b"#bundle\0"):
        messages: list[tuple[str, list]] = []
        offset = 16  # "#bundle\0" + 8-byte time tag (ignored: act on arrival)
        while offset + 4 <= len(data):
            (size,) = struct.unpack_from(">i", data, offset)
            offset += 4
            if size < 0 or offset + size > len(data):
                raise OscParseError("bad bundle element size")
            messages.extend(parse_packet(data[offset : offset + size]))
            offset += size
        return messages
    address, offset = _read_string(data, 0)
    if not address.startswith("/"):
        raise OscParseError(f"not an OSC address: {address!r}")
    if offset >= len(data):
        return [(address, [])]
    tags, offset = _read_string(data, offset)
    if not tags.startswith(","):
        raise OscParseError("missing OSC type tag string")
    args: list = []
    for tag in tags[1:]:
        if tag == "i":
            args.append(struct.unpack_from(">i", data, offset)[0])
            offset += 4
        elif tag == "f":
            args.append(struct.unpack_from(">f", data, offset)[0])
            offset += 4
        elif tag == "h":
            args.append(struct.unpack_from(">q", data, offset)[0])
            offset += 8
        elif tag == "d":
            args.append(struct.unpack_from(">d", data, offset)[0])
            offset += 8
        elif tag == "s":
            text, offset = _read_string(data, offset)
            args.append(text)
        elif tag == "T":
            args.append(True)
        elif tag == "F":
            args.append(False)
        elif tag == "N":
            args.append(None)
        else:
            raise OscParseError(f"unsupported OSC argument type {tag!r}")
    return [(address, args)]


def _is_release(args: list) -> bool:
    return len(args) == 1 and isinstance(args[0], (int, float)) and not isinstance(args[0], bool) and args[0] == 0


def dispatch(controller, address: str, args: list) -> str | None:
    """Run the action for one OSC message; returns its name, or None when
    the address is not ours or the message was a button release."""
    if not address.startswith(PREFIX):
        return None
    name = address[len(PREFIX):].strip("/")
    first = args[0] if args else None
    if name == "clear":
        if _is_release(args):
            return None
        actions.clear(controller)
    elif name == "mode":
        if _is_release(args):
            return None
        actions.set_mode(controller, first if isinstance(first, str) else None)
    elif name == "manual-text":
        actions.manual_text(controller, first if isinstance(first, str) else "")
    elif name in ("pause", "recording"):
        action = actions.pause if name == "pause" else actions.recording
        action(controller, first)
    else:
        logger.debug("OSC: unknown address %s", address)
        return None
    return name


class OscProtocol(asyncio.DatagramProtocol):
    def __init__(self, controller) -> None:
        self._controller = controller

    def datagram_received(self, data: bytes, addr) -> None:
        try:
            messages = parse_packet(data)
        except (OscParseError, struct.error) as exc:
            logger.debug("OSC: ignoring malformed packet from %s: %s", addr, exc)
            return
        for address, args in messages:
            try:
                name = dispatch(self._controller, address, args)
            except actions.ActionError as exc:
                logger.warning("OSC %s from %s: %s", address, addr, exc)
                continue
            except Exception:
                logger.exception("OSC %s from %s failed", address, addr)
                continue
            if name:
                logger.info("OSC %s %s from %s", address, args, addr[0] if addr else "?")


def encode_message(address: str, *args) -> bytes:
    """Build an OSC message (tests and the README's examples use it)."""

    def pad(raw: bytes) -> bytes:
        raw += b"\0"
        return raw + b"\0" * ((-len(raw)) % 4)

    tags, payload = ",", b""
    for arg in args:
        if isinstance(arg, bool):
            tags += "T" if arg else "F"
        elif isinstance(arg, int):
            tags += "i"
            payload += struct.pack(">i", arg)
        elif isinstance(arg, float):
            tags += "f"
            payload += struct.pack(">f", arg)
        elif isinstance(arg, str):
            tags += "s"
            payload += pad(arg.encode("utf-8"))
        else:
            raise TypeError(f"unsupported OSC argument {arg!r}")
    return pad(address.encode("utf-8")) + pad(tags.encode("ascii")) + payload
