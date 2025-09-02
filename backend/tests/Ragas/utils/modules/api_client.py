"""
API Client for interacting with the OpenInferESG API.
"""
import requests
import time
import socket
from typing import Dict, Optional, Any, Tuple

class OpenInferESGClient:
    def __init__(self, api_url: str = "http://localhost:8250"):
        """
        Initialize the OpenInferESG API client.
        
        Args:
            api_url: The base URL of the OpenInferESG API
        """
        self.api_url = api_url
        self.chat_endpoint = f"{api_url}/chat"
        self.upload_endpoint = f"{api_url}/report"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "OpenInferESG-Script/1.0",
            "Accept": "application/json",
            "Connection": "keep-alive"
        })
    
    def check_availability(self) -> bool:
        """
        Check if the OpenInferESG server is available
        
        Returns:
            True if the server is available, False otherwise
        """
        print(f"\nChecking server availability at {self.api_url}...")
        
        # Check if the host is reachable
        try:
            base_url_parts = self.api_url.split("://")
            host = base_url_parts[1].split(":")[0] if len(base_url_parts) > 1 else self.api_url
            port = self.api_url.split(":")[-1] if ":" in self.api_url.split("://")[-1] else "80"
            
            print(f"Testing connection to {host}:{port}...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex((host, int(port)))
            sock.close()
            
            if result != 0:
                print(f"✗ Connection error: Cannot connect to {host} on port {port}")
                return False
                
            print(f"✓ Connection to {host}:{port} successful")
            
        except Exception as e:
            print(f"✗ Error testing connection: {str(e)}")
            return False
        
        # Test API endpoints
        health_endpoints = ["/health", "/", "/info", "/version", "/chat?utterance=hello"]
        
        for endpoint in health_endpoints:
            try:
                response = requests.get(f"{self.api_url}{endpoint}", timeout=5)
                if response.status_code == 200:
                    print(f"✓ Server API is available")
                    return True
            except requests.exceptions.RequestException:
                pass
        
        print("✗ Server API does not appear to be responding")
        return False
    
    def upload_file(self, file_path: str) -> Optional[str]:
        """
        Upload a file to the OpenInferESG API
        
        Args:
            file_path: Path to the file to upload
            
        Returns:
            File ID if successful, None otherwise
        """
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (file_path.split('/')[-1], f)}
                upload_response = self.session.post(
                    self.upload_endpoint,
                    files=files,
                    timeout=300  # 5 minute timeout
                )
            
            if upload_response.status_code == 200:
                upload_data = upload_response.json()
                return upload_data.get('id')
            else:
                print(f"Upload failed with status {upload_response.status_code}: {upload_response.text}")
                return None
                
        except Exception as e:
            print(f"Upload error: {str(e)}")
            return None
    
    def wait_for_report(self, file_id: str, max_wait_time: int = 300) -> bool:
        """
        Wait for report generation to complete
        
        Args:
            file_id: The ID of the uploaded file
            max_wait_time: Maximum time to wait in seconds
            
        Returns:
            True if the report is ready, False otherwise
        """
        report_endpoint = f"{self.api_url}/report/{file_id}"
        check_interval = 10
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                response = self.session.get(
                    report_endpoint,
                    headers={"Accept": "text/markdown"},
                    timeout=10
                )
                
                if response.status_code == 200:
                    print(f"Report generation completed after {time.time() - start_time:.1f} seconds")
                    return True
                
            except Exception as e:
                print(f"Error checking report status: {str(e)}")
            
            time.sleep(check_interval)
        
        print("Timed out waiting for report generation")
        return False
    
    def get_answer(self, question: str, timeout: int = 120, retries: int = 3) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Get an answer from the OpenInferESG API
        
        Args:
            question: The question to ask
            timeout: Timeout in seconds
            retries: Number of retry attempts
            
        Returns:
            Tuple of (response_data, error_message)
        """
        for attempt in range(retries):
            try:
                params = {"utterance": question}
                response = self.session.get(
                    self.chat_endpoint,
                    params=params,
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    return response.json(), None
                else:
                    error_msg = f"API returned status {response.status_code}: {response.text}"
                    if attempt == retries - 1:  # Last attempt
                        return None, error_msg
                    time.sleep(10)  # Wait before retrying
                    
            except requests.exceptions.Timeout:
                error_msg = f"API request timed out after {timeout} seconds"
                if attempt == retries - 1:  # Last attempt
                    return None, error_msg
                time.sleep(20)  # Wait longer before retrying after a timeout
                
            except Exception as ex:
                error_msg = f"Exception during API call: {str(ex)}"
                if attempt == retries - 1:  # Last attempt
                    return None, error_msg
                time.sleep(10)  # Wait before retrying
        
        # This should never happen due to the returns in the loop
        return None, "Unknown error"
