import unittest
import socket
import threading
import time
import sys
import os

# Mock the function to test specific logic from shared code
def send_socket_message(host, port, message, timeout=5):
    """Helper to send a message to the Blender socket server."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((host, port))
            s.sendall(message.encode('utf-8'))
        return True
    except Exception as e:
        print(f"Socket Error: {e}")
        return False

class TestSocketCommunication(unittest.TestCase):
    def setUp(self):
        self.host = "127.0.0.1"
        self.port = 8129 # Use a different port than default to avoid conflict
        self.server_thread = None
        self.stop_event = threading.Event()
        self.received_messages = []
        
        # Start Mock Server
        self.server_thread = threading.Thread(target=self.mock_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        time.sleep(0.1) # Wait for server startup

    def tearDown(self):
        self.stop_event.set()
        # Trigger a connection to unblock accept() if needed
        try:
             with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                 s.connect((self.host, self.port))
        except: pass
        self.server_thread.join(timeout=1)

    def mock_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((self.host, self.port))
            s.listen(1)
            s.settimeout(1.0)
            
            while not self.stop_event.is_set():
                try:
                    conn, addr = s.accept()
                    with conn:
                        data = conn.recv(4096)
                        if data:
                            self.received_messages.append(data.decode('utf-8'))
                except socket.timeout:
                    continue
                except Exception as e:
                    if not self.stop_event.is_set():
                        print(f"Server error: {e}")

    def test_send_basic_message(self):
        msg = "TEST_MESSAGE"
        result = send_socket_message(self.host, self.port, msg)
        self.assertTrue(result, "Socket send should return True")
        
        time.sleep(0.1) # Wait for processing
        self.assertIn(msg, self.received_messages)

    def test_send_complex_message(self):
        msg = "TEXTURE_UPDATE:ALBEDO:C:\\Path\\To\\File.png|ROUGHNESS:C:\\Path.png"
        result = send_socket_message(self.host, self.port, msg)
        self.assertTrue(result)
        
        time.sleep(0.1)
        self.assertIn(msg, self.received_messages)

    def test_connection_refused(self):
        # Test connecting to a closed port
        result = send_socket_message(self.host, self.port + 1, "FAIL")
        self.assertFalse(result, "Should return False when connection refused")

if __name__ == '__main__':
    unittest.main()
