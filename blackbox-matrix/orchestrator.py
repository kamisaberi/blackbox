import yaml
import os
import subprocess
import time
import sys
import threading

def build_images():
    print(">>> Building Matrix Images...")
    subprocess.run("docker build -t blackbox-sim-iot ./virtual-devices/iot-arm32", shell=True)
    subprocess.run("docker build -t blackbox-sim-jetson ./virtual-devices/jetson-arm64", shell=True)
    subprocess.run("docker build -t blackbox-sim-server ./virtual-devices/server-linux", shell=True)

def spawn_node(config, group, idx):
    image = f"blackbox-sim-{group['image']}"
    name = f"{group['name']}-{idx}"
    behavior_file = group['behavior'] + ".json"
    
    # Mount behavior file from host to container
    abs_behavior_path = os.path.abspath(f"./behaviors/{behavior_file}")
    
    cmd = [
        "docker", "run", "-d", "--rm",
        "--name", name,
        # Pass Config
        "-e", f"CONTAINER_ID={name}",
        "-e", f"CORE_IP={config.get('core_ip', '172.17.0.1')}",
        "-e", "CORE_PORT=514",
        # Mount Behavior
        "-v", f"{abs_behavior_path}:/opt/behaviors/profile.json",
        "-e", "BEHAVIOR_PATH=/opt/behaviors/profile.json",
        # Image
        image
    ]
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL) # Silence output

def run_group(config, group):
    delay = group.get('start_delay', 0)
    if delay > 0:
        time.sleep(delay)
        print(f"   >>> Activating Group: {group['name']} ({group['count']} nodes)")
    
    for i in range(group['count']):
        spawn_node(config, group, i)

def run_scenario(filename):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)

    print(f"==========================================")
    print(f" SCENARIO: {config['name']}")
    print(f" NODES:    {sum(g['count'] for g in config['nodes'])}")
    print(f"==========================================")

    # Clean previous run
    os.system("docker rm -f $(docker ps -aq --filter name=thermostat-)")
    os.system("docker rm -f $(docker ps -aq --filter name=jetson-)")
    os.system("docker rm -f $(docker ps -aq --filter name=ip-camera-)")

    threads = []
    for group in config['nodes']:
        t = threading.Thread(target=run_group, args=(config, group))
        threads.append(t)
        t.start()

    try:
        time.sleep(config['duration_seconds'])
    except KeyboardInterrupt:
        print("\nStopping...")

    print(">>> TEARDOWN MATRIX...")
    # Cleanup logic (adjust filters as needed)
    os.system("docker rm -f $(docker ps -aq --filter ancestor=blackbox-sim-iot)")
    os.system("docker rm -f $(docker ps -aq --filter ancestor=blackbox-sim-jetson)")
    os.system("docker rm -f $(docker ps -aq --filter ancestor=blackbox-sim-server)")

if __name__ == "__main__":
    # Check dependencies
    if not os.path.exists("./virtual-devices"):
        print("Error: Run this script from the blackbox-matrix directory")
        exit(1)

    # 1. Build
    # build_images() # Uncomment to rebuild on start

    # 2. Run
    scenario = "scenarios/02_mirai_botnet.yaml"
    if len(sys.argv) > 1:
        scenario = sys.argv[1]
    
    run_scenario(scenario)