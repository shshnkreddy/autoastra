"""Network policy and real Docker checks; run on the Linux execution host."""

import json
from pathlib import Path
import shutil
import socket
import tempfile
import unittest
from unittest.mock import patch

from autoastra.autoresearch.config import RunnerConfig
from autoastra.autoresearch.network import model_hosts
from autoastra.sandbox import SandboxConfig, create_sandbox
from autoastra.sandbox.network import ModelGateway, _connect_public
from autoastra.sandbox.sandbox import _container_options, _docker
from tasks.affine import task


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='gateway-')
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'proxy.sock'
        self.gateway = ModelGateway(self.path, ('api.openai.com',))
        self.addCleanup(self.gateway.close)

    def request(self, authority, method='CONNECT'):
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.addCleanup(client.close)
        client.settimeout(5)
        client.connect(str(self.path))
        client.sendall(f'{method} {authority} HTTP/1.1\r\n\r\n'.encode())
        return client, client.recv(4096)

    def test_denies_other_hosts_ports_methods_before_dns(self):
        with patch('autoastra.sandbox.network.socket.getaddrinfo') as dns:
            for authority, method in [('example.com:443', 'CONNECT'), ('api.openai.com:80', 'CONNECT'),
                                      ('api.openai.com.evil.test:443', 'CONNECT'), ('127.0.0.1:443', 'CONNECT'),
                                      ('api.openai.com:443', 'GET')]:
                _, response = self.request(authority, method)
                self.assertIn(b' 403 ', response)
            dns.assert_not_called()

    def test_allowed_connection_transports_bytes(self):
        upstream, peer = socket.socketpair()
        self.addCleanup(upstream.close)
        self.addCleanup(peer.close)
        peer.settimeout(5)
        with patch('autoastra.sandbox.network._connect_public', return_value=upstream) as connect:
            client, response = self.request('api.openai.com:443')
            self.assertIn(b' 200 ', response)
            client.sendall(b'client bytes')
            self.assertEqual(peer.recv(64), b'client bytes')
            peer.sendall(b'provider bytes')
            self.assertEqual(client.recv(64), b'provider bytes')
            connect.assert_called_once_with('api.openai.com')

    def test_private_resolution_is_rejected(self):
        for address in ('127.0.0.1', '10.0.0.1', '169.254.169.254', '::1'):
            with patch('autoastra.sandbox.network.socket.getaddrinfo', return_value=[
                (socket.AF_INET, socket.SOCK_STREAM, 6, '', (address, 443))]):
                with self.assertRaisesRegex(ValueError, 'non-public'):
                    _connect_public('api.openai.com')

    def test_provider_policy_and_evaluator_isolation(self):
        self.assertEqual(model_hosts(RunnerConfig(provider='openrouter', model='test/model')), ('openrouter.ai',))
        self.assertEqual(model_hosts(RunnerConfig(provider='opencode', model='openai/test')), ('api.openai.com',))
        options = _container_options(SandboxConfig(model_hosts=('api.openai.com',)), '/usr/bin')
        self.assertIn('--network=none', options)
        self.assertFalse(any('PROXY' in option for option in options))
        for hosts in (('*',), ('127.0.0.1',), ('https://api.openai.com',), ['api.openai.com']):
            with self.assertRaises(ValueError):
                SandboxConfig(model_hosts=hosts)


class DockerNetworkTests(unittest.TestCase):
    def test_gateway_allows_provider_and_blocks_bypass(self):
        root = Path(tempfile.mkdtemp(prefix='network-integration-'))
        try:
            destination = root / ('long-run-name-' * 8) / 'sandbox'
            with create_sandbox(task, destination, config=SandboxConfig(
                    model_hosts=('api.openai.com',), pids_limit=128)) as sandbox:
                info = json.loads(_docker('inspect', sandbox.container_id).stdout)[0]
                self.assertEqual(info['HostConfig']['NetworkMode'], 'none')
                script = '''
import json, socket, time, urllib.request, urllib.error
for attempt in range(50):
    try:
        with socket.create_connection(('127.0.0.1',18080), timeout=1): pass
        break
    except OSError:
        time.sleep(.1)
else: raise AssertionError('Proxy did not start')
try:
    urllib.request.urlopen('https://api.openai.com/v1/models', timeout=25)
except urllib.error.HTTPError as error:
    assert error.code in (401,403), error
try:
    urllib.request.urlopen('https://example.com', timeout=5)
except urllib.error.URLError as error:
    assert '403' in str(error), error
else: raise AssertionError('General website was reachable')
for address in ('1.1.1.1','8.8.8.8'):
    try:
        socket.create_connection((address,443), timeout=1)
    except OSError: pass
    else: raise AssertionError('Direct IP connection bypassed proxy')
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
    connection.connect('/run/autoastra/model-proxy.sock')
    connection.sendall(b'CONNECT example.com:443 HTTP/1.1\\r\\n\\r\\n')
    assert b' 403 ' in connection.recv(4096)
print(json.dumps({'provider_https': True, 'general_internet': False, 'direct_ip': False}))
'''
                result = sandbox.run(['python', '-c', script], timeout=45)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                print(result.stdout.strip())
            self.assertFalse((destination/'.runtime/ipc/model-proxy.sock').exists())
        finally:
            for directory in root.rglob('*'):
                if directory.is_dir(): directory.chmod(0o700)
            shutil.rmtree(root)


if __name__ == '__main__':
    unittest.main()
