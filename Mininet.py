import sys
import pickle
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.log import setLogLevel
from mininet.cli import CLI
from mininet.node import RemoteController
import random
from randomPoissonDistribution import randomPoissonNumber_rand as randomPoissonNumber
import argparse
import numpy as np
import subprocess

class MyCustomTopo(Topo):
    "Custom topology example."

    def __init__(self, num_spine_switches, num_leaf_switches, num_hosts, min_cpu_range, max_cpu_range, bandwidth_range_host_leaf, bandwidth_range_leaf_spine, ch):
        "Create custom topo."
        # Initialize topology
        Topo.__init__(self)

        self.num_spine_switches = num_spine_switches
        self.num_leaf_switches = num_leaf_switches
        self.num_hosts = num_hosts
        # Add spine switches
        spines = [self.addSwitch(f's{i}') for i in range(1, num_spine_switches + 1)]
        # Add leaf switches
        leaves = [self.addSwitch(f'l{i}') for i in range(1, num_leaf_switches + 1)]

        if ch == 1:
            # Add hosts with CPU limits
            hosts = [self.addHost(f'h{i}', cpu=random.randint(min_cpu_range, max_cpu_range)) for i in range(1, num_hosts + 1)]

            # Keep track of created links
            created_links = []

            # Connect leaf switches to all spine switches
            for leaf in leaves:
                for spine in spines:
                    bw = int(random.randint(bandwidth_range_leaf_spine[0], bandwidth_range_leaf_spine[1]))
                    link = self.addLink(leaf, spine, bw=bw)
                    print(leaf, spine)
                    created_links.append((leaf, spine, bw))

            # Connect hosts to leaf switches in pairs
            for i in range(num_hosts):
                host = hosts[i]
                leaf_index = i // 5 % num_leaf_switches
                leaf_switch = leaves[leaf_index]
                bw = int(random.randint(bandwidth_range_host_leaf[0], bandwidth_range_host_leaf[1]))
                link = self.addLink(leaf_switch, host, bw=bw)
                created_links.append((leaf_switch, host, bw))
            self.created_links = created_links
        elif ch == 2:
            # Add hosts with CPU limits
            hosts = [self.addHost(f'h{i}', cpu=random.uniform(min_cpu_range, max_cpu_range)) for i in range(1, num_hosts + 1)]

            # Keep track of created links
            created_links = []

            # Connect leaf switches to all spine switches
            for leaf in leaves:
                for spine in spines:
                    bw = int(random.uniform(bandwidth_range_leaf_spine[0], bandwidth_range_leaf_spine[1]))
                    link = self.addLink(leaf, spine, bw=bw)
                    print(leaf, spine)
                    created_links.append((leaf, spine, bw))

            # Connect hosts to leaf switches in pairs
            for i in range(num_hosts):
                host = hosts[i]
                leaf_index = i // 5 % num_leaf_switches
                leaf_switch = leaves[leaf_index]
                bw = int(random.uniform(bandwidth_range_host_leaf[0], bandwidth_range_host_leaf[1]))
                link = self.addLink(leaf_switch, host, bw=bw)
                created_links.append((leaf_switch, host, bw))
            self.created_links = created_links
        elif ch == 3:
            # Add hosts with CPU limits
            hosts = [self.addHost(f'h{i}', cpu=max(1, int(np.random.normal(np.mean(min_cpu_range), np.std(max_cpu_range))))) for i in range(1, num_hosts + 1)]

            # Keep track of created links
            created_links = []

            # Connect leaf switches to all spine switches
            for leaf in leaves:
                for spine in spines:
                    bw = max(1, int(np.random.normal(np.mean(bandwidth_range_leaf_spine[0]), np.std(bandwidth_range_leaf_spine[1]))))
                    link = self.addLink(leaf, spine, bw=bw)
                    print(leaf, spine)
                    created_links.append((leaf, spine, bw))

            # Connect hosts to leaf switches in pairs
            for i in range(num_hosts):
                host = hosts[i]
                leaf_index = i // 5 % num_leaf_switches
                leaf_switch = leaves[leaf_index]
                bw = max(1, int(np.random.normal(np.mean(bandwidth_range_host_leaf[0]), np.std(bandwidth_range_host_leaf[1]))))
                link = self.addLink(leaf_switch, host, bw=bw)
                created_links.append((leaf_switch, host, bw))
            self.created_links = created_links
        else:
            # Add hosts with CPU limits
            hosts = [self.addHost(f'h{i}', cpu=randomPoissonNumber(min_cpu_range, max_cpu_range, 0.4)) for i in range(1, num_hosts + 1)]

            # Keep track of created links
            created_links = []

            # Connect leaf switches to all spine switches
            for leaf in leaves:
                for spine in spines:
                    bw = int(randomPoissonNumber(bandwidth_range_leaf_spine[0], bandwidth_range_leaf_spine[1], 0.4))
                    link = self.addLink(leaf, spine, bw=bw)
                    print(leaf, spine)
                    created_links.append((leaf, spine, bw))

            # Connect hosts to leaf switches in pairs
            for i in range(num_hosts):
                host = hosts[i]
                leaf_index = i // 5 % num_leaf_switches
                leaf_switch = leaves[leaf_index]
                bw = int(randomPoissonNumber(bandwidth_range_host_leaf[0], bandwidth_range_host_leaf[1], 0.4))
                link = self.addLink(leaf_switch, host, bw=bw)
                created_links.append((leaf_switch, host, bw))
            self.created_links = created_links

    def print_link_details(self):
        print("\nSubstrate Network Physical Link details :")
        for link in self.created_links:
            node1, node2, assigned_bw = link
            node1_name = node1.name if hasattr(node1, 'name') else node1
            node2_name = node2.name if hasattr(node2, 'name') else node2
            print(f"Link {node1_name} - {node2_name} assigned bandwidth: {int(assigned_bw)}")

def dumpNodeConnectionsToPickle(hosts, topo, pickle_file):
    "Dump connections to/from all nodes to a pickle file."
    data = {
        'num_spine_switches': topo.num_spine_switches,
        'num_leaf_switches': topo.num_leaf_switches,
        'num_hosts': topo.num_hosts,
        'links_details': []
    }

    for link in topo.created_links:
        node1, node2, assigned_bw = link
        node1_name = node1.name if hasattr(node1, 'name') else node1
        node2_name = node2.name if hasattr(node2, 'name') else node2
        link_details = {
            'node1': node1_name,
            'node2': node2_name,
            'assigned_bandwidth': int(assigned_bw)
        }
        data['links_details'].append(link_details)

    print("\nSubstrate Network Host's CPU details :")
    for host in hosts:
        connections = host.connectionsTo(hosts)
        host_data = {
            'allocated_cores': round(host.params['cpu']),
            'connections': []
        }
        print(f"{host.name} allocated cores: {round(host.params['cpu'])}")
        for conn in connections:
            if conn[0].isSwitch():
                assigned_bw = int(conn[0].connectionsTo(conn[1])[0].status.bw)
                print(f"{host.name} connected to {conn[0].name} with bandwidth {assigned_bw if assigned_bw is not None else 'Not Assigned'}")
                link_data = {
                    'switch_name': conn[0].name,
                    'bandwidth': assigned_bw if assigned_bw is not None else 'Not Assigned'
                }
                host_data['connections'].append(link_data)

        data[host.name] = host_data

    # Print link details after dumping host connections
    topo.print_link_details()

    # Dump the data to the pickle file
    with open(pickle_file, 'wb') as f:
        pickle.dump(data, f)

def runExperimentToPickle(num_spine_switches, num_leaf_switches, num_hosts, min_cpu_range, max_cpu_range, bandwidth_range_host_leaf, bandwidth_range_leaf_spine, ch, pickle_file):
    print("Starting network setup...")  # Debug print
    topo = MyCustomTopo(num_spine_switches, num_leaf_switches, num_hosts, min_cpu_range, max_cpu_range, bandwidth_range_host_leaf, bandwidth_range_leaf_spine, ch)
    net = Mininet(topo=topo, controller=None)  # Set controller to None initially
    ryu_controller = RemoteController('c0', ip='127.0.0.1', port=6653)
    net.addController(ryu_controller)
    net.start()
    print("Network started, running pingAll...")  # Debug print
    net.pingAll()  # This should output the results of pingAll
    print("pingAll completed, dumping data...")  # Debug print
    dumpNodeConnectionsToPickle(net.hosts, topo, pickle_file)
    net.stop()
    print("Network stopped, experiment completed.")

if __name__ == '__main__':
    print(f"Received arguments: {sys.argv[1:]}")  # Debug: Print arguments received
    if len(sys.argv) != 12:  # 11 arguments + the script name
        print("Usage: python3 Mininet.py <num_spine_switches> <num_leaf_switches> <num_hosts> <min_cpu_range> <max_cpu_range> <bw_host_leaf_min> <bw_host_leaf_max> <bw_leaf_spine_min> <bw_leaf_spine_max> <ch> <pickle_file>")
        sys.exit(1)

    num_spine_switches = int(sys.argv[1])
    num_leaf_switches = int(sys.argv[2])
    num_hosts = int(sys.argv[3])  # Now num_hosts is directly taken from the input
    min_cpu_range = int(sys.argv[4])
    max_cpu_range = int(sys.argv[5])
    bw_host_leaf_min = int(sys.argv[6])
    bw_host_leaf_max = int(sys.argv[7])
    bw_leaf_spine_min = int(sys.argv[8])
    bw_leaf_spine_max = int(sys.argv[9])
    ch = int(sys.argv[10])
    pickle_file = sys.argv[11]
    bandwidth_range_host_leaf = (bw_host_leaf_min, bw_host_leaf_max)
    bandwidth_range_leaf_spine = (bw_leaf_spine_min, bw_leaf_spine_max)

    setLogLevel('info')
    runExperimentToPickle(num_spine_switches, num_leaf_switches, num_hosts, min_cpu_range, max_cpu_range, bandwidth_range_host_leaf, bandwidth_range_leaf_spine, ch, pickle_file)