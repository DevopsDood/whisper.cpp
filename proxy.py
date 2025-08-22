#!/usr/bin/env python3

import subprocess
import threading
import time
import os
import signal
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import requests
import json

# Configuration
SERVER1_PORT = 8081
SERVER2_PORT = 8082
PROXY_PORT = 8080

# Detect if we're on NVIDIA system (Linux) and use appropriate model
def get_model_path():
    # Check if we're on Linux (likely NVIDIA)
    import platform
    if platform.system() == "Linux":
        # Check for NVIDIA GPU by looking for nvidia-smi command
        try:
            import subprocess
            subprocess.run(["nvidia-smi"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # If nvidia-smi is available, we're on NVIDIA system
            return "models/ggml-large-v3-turbo.bin"
        except (subprocess.CalledProcessError, FileNotFoundError):
            # No NVIDIA GPU detected or nvidia-smi not available
            pass
    
    # Default to regular ggml model for non-NVIDIA systems or if unsure
    return "models/ggml-large-v3-turbo.bin"

MODEL_PATH = get_model_path()
SERVER1_CMD = ["./build/bin/whisper-server", "--host", "0.0.0.0", "--port", str(SERVER1_PORT), "-l", "auto", "-m", MODEL_PATH]
SERVER2_CMD = ["./build/bin/whisper-server", "--host", "0.0.0.0", "--port", str(SERVER2_PORT), "-l", "auto", "-m", MODEL_PATH]

# Global variables for tracking servers and request counts
server1_process = None
server2_process = None
request_count = 0
server1_request_count = 0
server2_request_count = 0
current_server = 1  # Start with server 1

# Lock for thread safety
lock = threading.Lock()

def start_server(cmd, port):
    """Start a whisper server process"""
    try:
        print(f"Starting server on port {port} with command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd)
        # Wait a bit for server to start
        time.sleep(2)
        
        # Check if it's running
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                print(f"Server on port {port} started successfully")
                return process
        except Exception as e:
            print(f"Server on port {port} failed to start: {e}")
        return None
    except Exception as e:
        print(f"Failed to start server on port {port}: {e}")
        return None

def stop_server(process, port):
    """Stop a whisper server process"""
    if process and process.poll() is None:
        print(f"Stopping server on port {port}")
        try:
            process.terminate()
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print(f"Force killing server on port {port}")
            process.kill()
            process.wait()

def rotate_server():
    """Rotate between servers"""
    global current_server
    with lock:
        if current_server == 1:
            current_server = 2
        else:
            current_server = 1

def get_current_server():
    """Get the currently selected server"""
    global current_server
    with lock:
        return current_server

def get_next_server():
    """Get the next server to use"""
    global request_count, server1_request_count, server2_request_count
    with lock:
        # Check if we need to restart servers (every 5 requests)
        request_count += 1
        current = get_current_server()
        
        # Increment the count for the currently active server 
        if current == 1:
            server1_request_count += 1
        else:
            server2_request_count += 1
            
        # If the current server has reached 5 requests, restart it
        if (current == 1 and server1_request_count >= 5) or \
           (current == 2 and server2_request_count >= 5):
            print(f"Server {current} reached 5 requests, restarting it")
            return current, True  # Restart needed
        else:
            return current, False

def restart_server(port):
    """Restart the specified server"""
    global server1_process, server2_process
    with lock:
        print(f"Restarting server on port {port}")
        
        if port == SERVER1_PORT:
            stop_server(server1_process, SERVER1_PORT)
            server1_process = start_server(SERVER1_CMD, SERVER1_PORT)
        elif port == SERVER2_PORT:
            stop_server(server2_process, SERVER2_PORT)
            server2_process = start_server(SERVER2_CMD, SERVER2_PORT)

def handle_request(request_handler):
    """Handle a request and route to appropriate server"""
    
    # Determine which server to use
    current_server, restart_needed = get_next_server()
    
    if restart_needed:
        # Restart the appropriate server
        restart_server(SERVER1_PORT if current_server == 1 else SERVER2_PORT)
        
    # Reset the request counter for that server after restart
    global server1_request_count, server2_request_count
    with lock:
        if current_server == 1 and restart_needed:
            server1_request_count = 0
        elif current_server == 2 and restart_needed:
            server2_request_count = 0
    
    # Get the target server endpoint
    server_port = SERVER1_PORT if current_server == 1 else SERVER2_PORT
    
    # Forward the request
    try:
        # Handle multipart/form-data correctly
        content_type = request_handler.headers.get('Content-Type', '')
        
        # Read the raw data
        content_length = int(request_handler.headers.get('Content-Length', 0))
        post_data = request_handler.rfile.read(content_length)
        
        # Forward to the selected server
        url = f"http://localhost:{server_port}/inference"
        
        headers = {
            'Content-Type': content_type,
            'Content-Length': str(len(post_data))
        }
        
        response = requests.post(url, data=post_data, headers=headers)
        
        # Return the response
        request_handler.send_response(response.status_code)
        for key, value in response.headers.items():
            if key.lower() != 'connection':
                request_handler.send_header(key, value)
        request_handler.end_headers()
        
        if response.content:
            request_handler.wfile.write(response.content)
            
    except Exception as e:
        print(f"Error handling request: {e}")
        request_handler.send_response(500)
        request_handler.end_headers()
        request_handler.wfile.write(b"Internal Server Error")

class ProxyRequestHandler(BaseHTTPRequestHandler):
    """Handle HTTP requests for the proxy server"""
    
    def do_POST(self):
        handle_request(self)
        
    def do_GET(self):
        # For health check or other GET requests
        if self.path == "/health":
            try:
                # Check both servers are running
                response1 = requests.get(f"http://localhost:{SERVER1_PORT}/health", timeout=2)
                response2 = requests.get(f"http://localhost:{SERVER2_PORT}/health", timeout=2)
                
                if response1.status_code == 200 and response2.status_code == 200:
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"status":"ok"}')
                else:
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b'{"status":"unhealthy"}')
            except Exception:
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b'{"status":"unhealthy"}')
        else:
            # For other GET requests, forward them to a server
            try:
                response = requests.get(f"http://localhost:{SERVER1_PORT}{self.path}")
                
                self.send_response(response.status_code)
                for key, value in response.headers.items():
                    if key.lower() != 'connection':
                        self.send_header(key, value)
                self.end_headers()
                
                if response.content:
                    self.wfile.write(response.content)
            except Exception as e:
                print(f"Error handling GET request: {e}")
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"Internal Server Error")

def main():
    global server1_process, server2_process
    
    print("Starting whisper.cpp proxy service...")
    
    # Start both servers
    server1_process = start_server(SERVER1_CMD, SERVER1_PORT)
    server2_process = start_server(SERVER2_CMD, SERVER2_PORT)
    
    if not server1_process or not server2_process:
        print("Failed to start one or both servers")
        return
    
    # Start proxy server
    try:
        print(f"Starting proxy server on port {PROXY_PORT}")
        httpd = HTTPServer(('0.0.0.0', PROXY_PORT), ProxyRequestHandler)
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down proxy server...")
    finally:
        # Cleanup
        stop_server(server1_process, SERVER1_PORT)
        stop_server(server2_process, SERVER2_PORT)

if __name__ == "__main__":
    main()
