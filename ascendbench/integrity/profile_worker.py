"""Trusted, separately killable profiler parser. Never imports a candidate."""
import socket
import sys

from ascendbench.integrity.process_isolation import _parse_npu_kernel_event
from ascendbench.integrity.stream_transport import expect, message, send


def main():
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET, fileno=int(sys.argv[1]))
    send(sock, message("ready"))
    while True:
        request, _ = expect(sock, "parse")
        kernel, fallback = _parse_npu_kernel_event(request["raw"], request["root"], synchronous=True)
        send(sock, message("parsed", id=request["id"], kernel=kernel, fallback=fallback))


if __name__ == "__main__":
    main()
