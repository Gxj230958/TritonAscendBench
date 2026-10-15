"""Bounded JSON control messages and immutable, data-only tensor payloads."""
from __future__ import annotations

import array
import contextlib
import fcntl
import json
import os
import socket
import stat
import tempfile

from safetensors.torch import load_file, save, save_file

VERSION = 1
MAX_MESSAGE = 65536
MAX_PAYLOAD = 256_000_000_000
SEALS = (fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK |
         fcntl.F_SEAL_SEAL)


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def send(sock, message, fd=None):
    data = json.dumps(message, allow_nan=False, separators=(",", ":")).encode()
    if len(data) > MAX_MESSAGE:
        raise ValueError("control message too large")
    rights = [] if fd is None else [
        (socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))]
    if sock.sendmsg([data], rights) != len(data):
        raise RuntimeError("short control write")


def receive(sock, *, payload=False):
    data, ancillary, flags, _ = sock.recvmsg(
        MAX_MESSAGE, socket.CMSG_SPACE(4 * array.array("i").itemsize),
        socket.MSG_CMSG_CLOEXEC)
    fds = []
    try:
        unexpected = False
        for level, kind, raw in ancillary:
            if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS:
                unexpected = True
                continue
            values = array.array("i")
            values.frombytes(raw[:len(raw) - len(raw) % values.itemsize])
            fds.extend(values)
        if unexpected:
            raise ValueError("unexpected ancillary message")
        if not data or flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
            raise ValueError("closed or truncated control channel")
        if len(fds) != int(payload):
            raise ValueError("unexpected descriptor count")
        msg = json.loads(data, object_pairs_hook=_json_object,
                         parse_constant=lambda s: (_ for _ in ()).throw(
            ValueError("non-finite JSON")))
        if not isinstance(msg, dict) or msg.get("v") != VERSION:
            raise ValueError("invalid protocol version/message")
        return msg, fds.pop() if fds else None
    finally:
        for fd in fds:
            os.close(fd)


@contextlib.contextmanager
def payload_file(tensors, directory, transport="memfd"):
    """The fallback file is unlinked before transfer and reclaimed on close."""
    path = None
    fd = None
    try:
        if transport == "memfd" and hasattr(os, "memfd_create"):
            fd = os.memfd_create("tab-trial", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
            # Path writers may use atomic rename, which procfs cannot support.
            encoded = save(tensors)
            if not 8 <= len(encoded) <= MAX_PAYLOAD:
                raise ValueError("payload size outside bounds")
            view = memoryview(encoded)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("short payload write")
                view = view[written:]
            del view, encoded
            fcntl.fcntl(fd, fcntl.F_ADD_SEALS, SEALS)
            kind = "memfd"
        else:
            fd, path = tempfile.mkstemp(prefix="trial-", dir=directory)
            save_file(tensors, path)
            os.close(fd)
            fd = None
            fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
            os.unlink(path)
            path = None
            kind = "file"
        size = os.fstat(fd).st_size
        if not 8 <= size <= MAX_PAYLOAD:
            raise ValueError("payload size outside bounds")
        yield fd, {"bytes": size, "storage": kind}
    finally:
        if fd is not None:
            os.close(fd)
        if path is not None:
            os.unlink(path)


def load_payload(fd, msg):
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or type(msg.get("bytes")) is not int
                or info.st_size != msg["bytes"] or not 8 <= info.st_size <= MAX_PAYLOAD):
            raise ValueError("invalid payload descriptor/size")
        if msg.get("storage") == "memfd":
            if fcntl.fcntl(fd, fcntl.F_GET_SEALS) & SEALS != SEALS:
                raise ValueError("unsealed payload")
        elif msg.get("storage") == "file":
            if fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY:
                raise ValueError("fallback descriptor must be read-only")
        else:
            raise ValueError("invalid payload storage")
        # Clone detaches mappings and permits candidates to mutate their inputs.
        return {k: v.clone() for k, v in load_file(f"/proc/self/fd/{fd}").items()}
    finally:
        os.close(fd)


def message(kind, **kwargs):
    return {"v": VERSION, "kind": kind, **kwargs}


def expect(sock, kind, identity=None, *, payload=False):
    msg, fd = receive(sock, payload=payload)
    if msg.get("kind") != kind or (identity is not None and (
            type(msg.get("id")) is not int or msg.get("id") != identity)):
        if fd is not None:
            os.close(fd)
        raise ValueError(f"unexpected message; expected {kind}")
    return msg, fd


def child_loop(g):
    """Trusted shim; inputs and outputs die before the release acknowledgement."""
    args, fn = g["args"], g["fn"]
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET, fileno=args["stream_fd"])
    compile_needed = (g["_BACKEND"].needs_compile_warmup() and g["_uses_triton"])
    send(sock, message("ready", compile=compile_needed, error=g["status"]["load_error"]))
    while True:
        msg, fd = receive(sock, payload=True)
        identity = msg.get("id")
        if msg.get("kind") != "trial" or type(identity) is not int:
            os.close(fd)
            raise ValueError("expected trial")
        inp = load_payload(fd, msg)
        inp.update(msg["scalars"])
        inp = g["_to_device"](inp)
        send(sock, message("loaded", id=identity))
        expect(sock, "execute", identity)
        rec = g["_new_rec"]()
        flat, out, timer, bracket = {}, None, None, None
        try:
            phase = msg["phase"]
            if phase == "compile":
                g["set_audit_phase"]("compile")
                try:
                    out = fn(inp)
                    g["_sync_device"]()
                finally:
                    g["set_audit_phase"]("graded")
            elif phase == "warmup":
                out = fn(inp)
                g["_sync_device"]()
                flat, rec["error"] = g["_flatten"](out)
            elif phase == "correctness":
                flat, rec["error"] = g["_record_diag"](rec, inp)
            elif phase == "measured":
                timer = g["_make_event_timer"]()
                bracket = timer if timer is not None else contextlib.nullcontext()
                g["_sync_device"]()
                h0 = g["_host_ns"]()
                try:
                    with bracket:
                        if msg["profile"]:
                            out, diag = g["_profile"](fn, inp)
                            rec.update(diag)
                        else:
                            out = fn(inp)
                finally:
                    g["_sync_device"]()
                h1 = g["_host_ns"]()
                flat, rec["error"] = g["_flatten"](out)
                if not msg["profile"]:
                    rec["durations"] = [h1 - h0]
                    rec["event_ns"] = [timer.elapsed_ns()] if timer is not None else None
            else:
                raise ValueError("invalid execution phase")
            rec["ran"] = rec["error"] is None
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
        if not msg["output"] or flat is None:
            flat = {}
        send(sock, message("executed", id=identity))
        with payload_file(flat, args["out_dir"], args["transport"]) as (out_fd, meta):
            send(sock, message("result", id=identity, record=rec, **meta), out_fd)
            expect(sock, "consumed", identity)
        del inp, flat, out, timer, bracket
        send(sock, message("released", id=identity))
        if msg["last"]:
            break
    sock.close()
