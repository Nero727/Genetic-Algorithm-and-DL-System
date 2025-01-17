import subprocess
import os
import sys

def launch_instances(num_instances, log_dir="logs"):
    """
    Launch multiple instances of the main script with separate logging in new terminal windows.

    Args:
        num_instances (int): Number of instances to launch.
        log_dir (str): Directory to store logs for each instance.
    """
    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    processes = []

    for instance_id in range(1, num_instances + 1):
        log_file = os.path.join(log_dir, f"instance_{instance_id}.log")
        command = [
            sys.executable,  # Path to the current Python interpreter
            "Strategy_Generator_V10.py",
            "--instance", str(instance_id)
        ]

        if os.name == "nt":  # Windows
            process = subprocess.Popen(
                ["start", "cmd", "/k"] + command,
                stdout=open(log_file, 'w'),
                stderr=subprocess.STDOUT,
                shell=True
            )
        elif os.name == "posix":  # macOS/Linux
            process = subprocess.Popen(
                ["gnome-terminal", "--"] + command,
                stdout=open(log_file, 'w'),
                stderr=subprocess.STDOUT
            )
        else:
            raise OSError("Unsupported operating system.")

        processes.append(process)
        print(f"Launched instance {instance_id} in a new terminal with log file: {log_file}")

    print("All instances launched. Monitoring...")
    for process in processes:
        process.wait()

if __name__ == "__main__":
    num_instances = 1  # Adjust the number of instances as needed
    launch_instances(num_instances)
