"""Small JSON RPC over a Unix socket, with bounded messages."""

import json
import os
from pathlib import Path
import socket
import socketserver
import threading

MAX_MESSAGE = 4 * 1024 * 1024


def request(path, payload, timeout=120):
    data = json.dumps(payload, allow_nan=False).encode() + b"\n"
    if len(data) > MAX_MESSAGE:
        raise ValueError("Request is too large")
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(timeout)
        connection.connect(str(path))
        connection.sendall(data)
        with connection.makefile("rb") as stream:
            line = stream.readline(MAX_MESSAGE + 1)
    if len(line) > MAX_MESSAGE or not line.endswith(b"\n"):
        raise RuntimeError("Invalid service response")
    result = json.loads(line)
    if not result["ok"]:
        raise RuntimeError(result["error"])
    return result["result"]


class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, path, dispatch):
        self.path = Path(path)
        self.dispatch = dispatch
        self.slots = threading.BoundedSemaphore(8)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # A short /proc path avoids Unix socket's 108-byte address limit while
        # keeping the socket within the directory mounted into the container.
        fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            address = f"/proc/self/fd/{fd}/{self.path.name}" if Path('/proc/self/fd').is_dir() else str(self.path)
            super().__init__(address, Handler)
        finally:
            os.close(fd)
        self.path.chmod(0o600)
        self.thread = threading.Thread(target=self.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            request.close()
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    def close(self):
        self.shutdown()
        self.server_close()
        self.thread.join()
        self.path.unlink(missing_ok=True)


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        self.request.settimeout(10)
        try:
            line = self.rfile.readline(MAX_MESSAGE + 1)
            if len(line) > MAX_MESSAGE or not line.endswith(b"\n"):
                raise ValueError("Invalid request size")
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("Request must be an object")
            result = {"ok": True, "result": self.server.dispatch(payload)}
        except Exception as error:
            result = {"ok": False, "error": str(error)[:4000]}
        try:
            data = json.dumps(result, allow_nan=False).encode() + b"\n"
            if len(data) > MAX_MESSAGE:
                data = b'{"ok":false,"error":"Response is too large"}\n'
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
