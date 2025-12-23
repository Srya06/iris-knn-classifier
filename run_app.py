import subprocess
import time
import socket

print("🚀 Starting Streamlit server...")
process = subprocess.Popen(["streamlit", "run", "streamlit_app.py", "--server.headless", "true"])

time.sleep(3)  # Wait for server to start

# Get local IP
hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)

print(f"\n✅ Streamlit is running!")
print(f"🌐 Local URL: http://localhost:8501")
print(f"🔗 Network URL: http://{local_ip}:8501")
print(f"\nPress Ctrl+C to stop the server\n")

try:
    process.wait()
except KeyboardInterrupt:
    process.terminate()
    print("\n🛑 Server stopped")
