"""Host-owned HTTPS CONNECT allowlist reached over a Unix socket from offline Docker.

TLS stays end-to-end. Request contents, responses, and credentials are not logged.
"""

import ipaddress
import os
from pathlib import Path
import re
import select
import socket
import socketserver
import threading

PROXY_URL = 'http://127.0.0.1:18080'
PROXY_SOCKET = '/run/autoastra/model-proxy.sock'


def validate_hosts(hosts):
    if not isinstance(hosts, tuple) or any(
        not isinstance(host, str) or len(host) > 253 or not re.fullmatch(
            r'[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)+', host)
        for host in hosts
    ):
        raise ValueError('model_hosts must be a tuple of exact lowercase DNS names')
    for host in hosts:
        try:
            ipaddress.ip_address(host)
        except ValueError:
            continue
        raise ValueError('model_hosts cannot contain IP addresses')


def _connect_public(host):
    addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError('Provider resolved to a non-public address')
    last_error = None
    for family, kind, protocol, _, address in addresses:
        connection = socket.socket(family, kind, protocol)
        connection.settimeout(10)
        try:
            connection.connect(address)
            return connection
        except OSError as error:
            connection.close()
            last_error = error
    raise OSError('Cannot connect to provider') from last_error


def _tunnel(left, right, closed):
    left.settimeout(10)
    right.settimeout(10)
    while not closed.is_set():
        readable, _, _ = select.select([left, right], [], [], 1)
        for source in readable:
            data = source.recv(65536)
            if not data:
                return
            (right if source is left else left).sendall(data)


class _Threaded:
    daemon_threads = True
    block_on_close = False

    def configure(self):
        self.closed = threading.Event()
        self.slots = threading.BoundedSemaphore(32)
        self.connections = set()
        self.lock = threading.Lock()

    def process_request(self, request, address):
        if not self.slots.acquire(blocking=False):
            request.close()
            return
        with self.lock:
            self.connections.add(request)
        try:
            super().process_request(request, address)
        except BaseException:
            self.finish_connection(request)
            raise

    def finish_connection(self, request):
        with self.lock:
            self.connections.discard(request)
        self.slots.release()

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.finish_connection(request)

    def handle_error(self, request, address):
        pass  # Disconnects must not print potentially sensitive request data.


class _UnixServer(_Threaded, socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    pass


class _TCPServer(_Threaded, socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True


class _ConnectHandler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(10)
        header = b''
        while b'\r\n\r\n' not in header:
            data = self.request.recv(4096)
            if not data:
                return
            header += data
            if len(header) > 16384:
                return self.reject(400)
        try:
            method, authority, version = header.split(b'\r\n', 1)[0].decode('ascii').split(' ')
            if method != 'CONNECT' or version not in ('HTTP/1.0', 'HTTP/1.1'):
                return self.reject(403)
            host, port = authority.rsplit(':', 1)
            # Check before DNS lookup: arbitrary DNS requests are blocked too.
            if host not in self.server.hosts or port != '443':
                return self.reject(403)
            if not header.endswith(b'\r\n\r\n'):
                return self.reject(400)
        except (ValueError, UnicodeError):
            return self.reject(400)
        try:
            upstream = _connect_public(host)
        except (OSError, ValueError):
            return self.reject(502)
        with upstream:
            self.request.sendall(b'HTTP/1.1 200 Connection Established\r\n\r\n')
            _tunnel(self.request, upstream, self.server.closed)

    def reject(self, status):
        self.request.sendall(f'HTTP/1.1 {status} Blocked\r\nContent-Length: 0\r\nConnection: close\r\n\r\n'.encode())


class ModelGateway:
    """Host policy and listener; not exposed as a host TCP service."""

    def __init__(self, path, hosts):
        validate_hosts(hosts)
        self.path = Path(path)
        # Keep the socket in the mounted directory without the Unix address
        # length limit breaking deeply nested multi-seed output paths.
        fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            address = (f'/proc/self/fd/{fd}/{self.path.name}'
                       if Path('/proc/self/fd').is_dir() else str(self.path))
            self.server = _UnixServer(address, _ConnectHandler)
        finally:
            os.close(fd)
        self.server.configure()
        self.server.hosts = frozenset(hosts)
        self.path.chmod(0o600)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.closed.set()
        self.server.shutdown()
        with self.server.lock:
            for connection in self.server.connections:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        self.server.server_close()
        self.thread.join(timeout=2)
        self.path.unlink(missing_ok=True)


class _RelayHandler(socketserver.BaseRequestHandler):
    def handle(self):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as upstream:
            upstream.connect(PROXY_SOCKET)
            _tunnel(self.request, upstream, self.server.closed)


def proxy_environment():
    return [item for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
                             'http_proxy', 'https_proxy', 'all_proxy')
            for item in ('--env', f'{name}={PROXY_URL}')] + [
                '--env', 'NO_PROXY=localhost,127.0.0.1,::1',
                '--env', 'no_proxy=localhost,127.0.0.1,::1']


if __name__ == '__main__':
    # Development container's main process; never mounted in the evaluator.
    with _TCPServer(('127.0.0.1', 18080), _RelayHandler) as server:
        server.configure()
        server.serve_forever()
