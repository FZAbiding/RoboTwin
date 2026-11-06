"""Actual TCP lifecycle regression; no model, simulator or other service used."""
import errno
import socket
import unittest
from scripts.run_policy_experiments import check_port_available


class PortTest(unittest.TestCase):
    def test_closed_server_port_can_be_reused(self):
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            server.bind(('127.0.0.1',0));port=server.getsockname()[1];server.listen()
            with socket.create_connection(('127.0.0.1',port),timeout=2) as client:
                peer,_=server.accept();peer.close()
                self.assertEqual(client.recv(1),b'')
        with socket.socket() as old_probe:
            with self.assertRaises(OSError) as error:old_probe.bind(('127.0.0.1',port))
            self.assertEqual(error.exception.errno,errno.EADDRINUSE)
        check_port_available('127.0.0.1',port)

    def test_active_listener_remains_protected(self):
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            server.bind(('127.0.0.1',0));port=server.getsockname()[1];server.listen()
            with self.assertRaises(OSError) as error:check_port_available('127.0.0.1',port)
            self.assertEqual(error.exception.errno,errno.EADDRINUSE)


if __name__=='__main__':unittest.main()
