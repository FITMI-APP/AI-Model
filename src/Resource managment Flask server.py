import time
import subprocess
from flask import Flask, jsonify
import psutil

app = Flask(__name__)
process = None

import subprocess
import time

def start_server():
    global process
    if not process:
        try:
            starter_script = "C:/Users/Ziad Mazhar/Desktop/project/ladi-vton_Dress_code - Copy/src/backend.py"
            command = f'cmd.exe /k "conda activate tastoot && python \"{starter_script}\""'
            
            # Start the process
            process = subprocess.Popen(command, creationflags=subprocess.CREATE_NEW_CONSOLE)
            time.sleep(2)  # Give some time for the server to start
            process.wait(timeout=0)  # Wait without blocking
            
            return "Server started"
        except FileNotFoundError:
            return "Backend script not found. Please check the path."
        except PermissionError:
            return "Permission denied. Make sure you have the necessary permissions."
        except subprocess.TimeoutExpired:
            return "Server started"
        except Exception as e:
            return f"Error starting server: {e}"
    else:
        return "Server already running"

# Initialize the global process variable
process = None

@app.route('/start-server', methods=['GET'])
def start_server_endpoint():
    message = start_server()
    return jsonify({"message": message})

@app.route('/stop-server', methods=['GET'])
def stop_server_endpoint():
    message = stop_backend_server()
    return jsonify({"message": message})


def stop_backend_server():
    global process 
    # Find the process running the backend.py script
    for proc in psutil.process_iter(['pid', 'cmdline']):
        cmdline = proc.info['cmdline']
        if cmdline and 'backend.py' in ' '.join(cmdline):
            pid = proc.pid
            break
    else:
        print("Server is not running")
        return None

    # Terminate the process
    try:
        process = psutil.Process(pid)
        process.terminate()

        # Wait for the process to terminate, with repeated checks
        timeout = 5  # seconds
        interval = 0.5  # check every 0.5 seconds
        total_wait_time = 0

        while total_wait_time < timeout:
            if not process.is_running():
                print("Server stopped successfully")
                process = None
                return pid
            time.sleep(interval)
            total_wait_time += interval

        # Final check
        if process.is_running():
            print("Failed to stop the server: Process is still running")
            return None
        else:
            print("Server stopped successfully")
            process = None
            return pid
    except psutil.NoSuchProcess:
        print("The process does not exist")
        return None
    except Exception as e:
        print(f"Failed to stop the server: {e}")
        return None

@app.route('/', methods=['GET'])
def health():
    return jsonify("hello from model api backend")

if __name__ == "__main__":
    app.run(host='192.168.125.111', port=6000, debug=False, threaded=False)
