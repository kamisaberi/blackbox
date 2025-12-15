import os
import time
import random
import json
import socket
import datetime

# --- CONFIGURATION ---
# These are injected by the Orchestrator via Docker ENV
BEHAVIOR_FILE = os.getenv("BEHAVIOR_PATH", "/opt/behaviors/iot_normal.json")
CONTAINER_ID = os.getenv("CONTAINER_ID", "unknown-device")
TARGET_IP = os.getenv("CORE_IP", "172.17.0.1") # Default to Docker Gateway
TARGET_PORT = int(os.getenv("CORE_PORT", "514"))

def load_behavior():
    try:
        with open(BEHAVIOR_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[ACTOR] Error loading behavior {BEHAVIOR_FILE}: {e}")
        return {"eps": 1, "patterns": [{"weight": 1, "log": "error loading behavior"}]}

def send_udp(sock, message):
    try:
        bytes_msg = message.encode('utf-8')
        sock.sendto(bytes_msg, (TARGET_IP, TARGET_PORT))
    except Exception as e:
        print(f"[ACTOR] UDP Send Error: {e}")

def main():
    print(f"[ACTOR] Online: {CONTAINER_ID}")
    print(f"[ACTOR] Target: {TARGET_IP}:{TARGET_PORT}")
    print(f"[ACTOR] Profile: {BEHAVIOR_FILE}")

    config = load_behavior()
    
    # Setup UDP Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Calculate sleep time based on EPS (Events Per Second)
    eps = config.get('eps', 1.0)
    sleep_time = 1.0 / eps

    patterns = config.get('patterns', [])
    weights = [p['weight'] for p in patterns]
    
    targets = config.get('targets', ["10.0.0.5", "192.168.1.50", "8.8.8.8"])

    while True:
        # 1. Select a pattern
        selected = random.choices(patterns, weights=weights, k=1)[0]
        template = selected['log']

        # 2. Dynamic Replacement
        # Generate a timestamp: <PRI>Mon DD HH:MM:SS Host App: Msg
        # Syslog format (RFC 3164)
        pri = random.randint(1, 190)
        timestamp = datetime.datetime.now().strftime("%b %d %H:%M:%S")
        
        # Replace variables
        log_msg = template.replace("{target_ip}", random.choice(targets))
        log_msg = log_msg.replace("{random_port}", str(random.randint(1024, 65535)))
        log_msg = log_msg.replace("{random_user}", random.choice(["root", "admin", "guest", "user"]))
        
        # 3. Construct Final Packet
        # Format: <PRI>TIMESTAMP HOSTNAME APP: MESSAGE
        full_packet = f"<{pri}>{timestamp} {CONTAINER_ID} sim_app: {log_msg}"

        # 4. Fire
        send_udp(sock, full_packet)

        # 5. Wait
        # Add some jitter to make it look realistic
        jitter = random.uniform(0.9, 1.1)
        time.sleep(sleep_time * jitter)

if __name__ == "__main__":
    main()