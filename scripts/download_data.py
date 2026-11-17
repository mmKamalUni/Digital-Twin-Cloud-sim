"""Download the public datasets into data/raw/ (Intel Berkeley Lab and NAB).

    python scripts/download_data.py
"""
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")

INTEL = "http://db.csail.mit.edu/labdata/data.txt.gz"
NAB = "https://raw.githubusercontent.com/numenta/NAB/master/"
NAB_FILES = {
    "realKnownCause": ["ambient_temperature_system_failure", "cpu_utilization_asg_misconfiguration",
                       "ec2_request_latency_system_failure", "machine_temperature_system_failure",
                       "nyc_taxi", "rogue_agent_key_hold"],
    "realAWSCloudwatch": ["ec2_cpu_utilization_24ae8d", "ec2_cpu_utilization_53ea38",
                          "ec2_cpu_utilization_5f5533", "ec2_cpu_utilization_77c1ca",
                          "ec2_cpu_utilization_825cc2", "ec2_cpu_utilization_ac20cd",
                          "ec2_cpu_utilization_c6585a", "ec2_cpu_utilization_fe7f93",
                          "ec2_disk_write_bytes_1ef3de", "ec2_disk_write_bytes_c0d644",
                          "ec2_network_in_257a54", "ec2_network_in_5abac7", "elb_request_count_8c0756",
                          "grok_asg_anomaly", "iio_us-east-1_i-a2eb1cd9_NetworkIn",
                          "rds_cpu_utilization_cc0c53", "rds_cpu_utilization_e47b3b"],
}


def fetch(url, path):
    if os.path.exists(path):
        print("exists ", path)
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("get    ", url)
    urllib.request.urlretrieve(url, path)


def main():
    fetch(INTEL, os.path.join(RAW, "intel", "data.txt.gz"))
    for cat, names in NAB_FILES.items():
        for n in names:
            fetch(f"{NAB}data/{cat}/{n}.csv", os.path.join(RAW, "nab", f"{n}.csv"))
    fetch(NAB + "labels/combined_windows.json", os.path.join(RAW, "nab", "combined_windows.json"))
    fetch(NAB + "LICENSE.txt", os.path.join(RAW, "nab", "LICENSE.txt"))


if __name__ == "__main__":
    main()
