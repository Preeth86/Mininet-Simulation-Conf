import pickle
import numpy as np
from scipy.stats import norm
from math import exp
import heapq
import os
import sys
import json
import sys

output = []

def calculate_total_bandwidth(graph):
    total_bandwidth = 0
    for node in graph:
        for neighbor in graph[node]:
            total_bandwidth += graph[node][neighbor]['bandwidth']
    return total_bandwidth / 2  # Since it's an undirected graph


def custom_print(*args):
    message = ' '.join(str(arg) for arg in args)
    print(message)
    output.append([message])

def load_topology_from_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def calculate_mean_and_std(servers):
    cpu_resources = [server['cpu'] for server in servers.values()]  # Use current available CPU
    mean_cpu, std_cpu = np.mean(cpu_resources), np.std(cpu_resources, ddof=1)
    std_cpu = max(std_cpu, 1e-6)  # Prevent division by zero
    return mean_cpu, std_cpu


def calculate_link_bandwidth_statistics(graph):
    bandwidths = []
    for node in graph:
        for neighbor, bandwidth in graph[node].items():
            bandwidths.append(bandwidth)  # Append bandwidth of each link

    mean_bw_available = np.mean(bandwidths)
    std_bw_available = np.std(bandwidths, ddof=1)
    std_bw_available = max(std_bw_available, 1e-6)  # Prevent division by zero
    return mean_bw_available, std_bw_available

def node_embedding_and_mapping(servers, vnr):
    custom_print(f"\nNode Embedding and Mapping of VMs for VNR ID: {vnr['vnr_id'] + 1}")
    vm_to_server_assignments = {}
    vnr_to_server_assignments = {}

    P_idle, P_full, alpha_1 = 150, 300, 1.0  # Power constants and scaling factor for cost function

    for vm_index, vm_cpu in enumerate(vnr['vm_cpu_cores'], start=1):
        if vm_cpu <= 0:
            continue

        # Calculate mean and standard deviation based on current available CPU
        mean_cpu, std_cpu = calculate_mean_and_std(servers)
        server_calculations = []

        for server_id, server_info in servers.items():
            if vm_cpu > server_info['cpu']:
                continue

            # Ensure no multiple VMs of the same VNR are mapped to the same server
            if server_id in vnr_to_server_assignments.get(vnr['vnr_id'], []):
                continue

            U_cpu = (vm_cpu / server_info['cpu']) * 100
            overloading_prob = 1 - norm.cdf((server_info['cpu'] - vm_cpu - mean_cpu) / std_cpu)
            cumulative_cpu = vm_cpu + sum(v['cpu'] for v in server_info['vms'])
            P_k_U_cpu = P_idle + (P_full - P_idle) * (cumulative_cpu / server_info['original_cpu'])
            node_mapping_objective = P_k_U_cpu * exp(alpha_1 * overloading_prob)

            server_calculations.append((server_id, node_mapping_objective))
            custom_print(
                f"Server {server_id}, VM{vm_index}: CPU Utilization = {U_cpu:.2f}%, Overloading Probability = {overloading_prob * 100:.2f}%, Energy consumption = {P_k_U_cpu:.2f} W, Node Objective = {node_mapping_objective:.2f}")

        if server_calculations:
            best_server, best_node_objective = min(server_calculations, key=lambda x: x[1])
            custom_print(
                f"Best server for VM{vm_index} is {best_server} with Node Objective = {best_node_objective:.2f}")
            servers[best_server]['cpu'] -= vm_cpu
            servers[best_server]['vms'].append({'vnr_id': vnr['vnr_id'], 'vm_index': vm_index, 'cpu': vm_cpu})
            vm_to_server_assignments[f"VM{vm_index}"] = best_server
            vnr_to_server_assignments.setdefault(vnr['vnr_id'], []).append(best_server)
        else:
            custom_print(f"No suitable server found for VM{vm_index}.")

    return vm_to_server_assignments, vnr_to_server_assignments, servers

def dijkstra(graph, src, dst, bandwidth, k=1):
    heap = [(0, [src])]
    paths = []
    visited = set()
    while heap:
        (cost, path) = heapq.heappop(heap)
        node = path[-1]
        if node == dst:
            paths.append(path)
            if len(paths) >= k:
                return paths[0]  # Return the shortest path
        if node not in visited:
            visited.add(node)
            for neighbor, edge_data in graph[node].items():
                neighbor_cost = edge_data['bandwidth']
                if neighbor not in visited and neighbor_cost >= bandwidth:
                    heapq.heappush(heap, (cost + neighbor_cost, path + [neighbor]))
    return None


def link_embedding_and_mapping(graph, vnr, vm_to_server_assignments, link_flags):
    custom_print(f"\nLink Embedding and Mapping of Virtual Links for VNR ID: {vnr['vnr_id'] + 1} using Dijkstra's Algorithm:")
    embedding_success = {vnr['vnr_id']: True}
    path_mappings = []

    for link_index, (vm_source, vm_target) in enumerate(vnr['vm_links'], start=1):
        bandwidth_demand = vnr['bandwidth_values'][link_index - 1]
        custom_print(f"VM Source: {vm_source}, VM Target: {vm_target}")
        custom_print(f"VM to Server Assignments: {vm_to_server_assignments}")

        if f"VM{vm_source + 1}" not in vm_to_server_assignments or f"VM{vm_target + 1}" not in vm_to_server_assignments:
            custom_print(f"Failed to find server assignments for VM{vm_source + 1} or VM{vm_target + 1}.")
            embedding_success[vnr['vnr_id']] = False
            break

        source_server = vm_to_server_assignments[f"VM{vm_source + 1}"]
        target_server = vm_to_server_assignments[f"VM{vm_target + 1}"]

        shortest_path = dijkstra(graph, source_server, target_server, bandwidth_demand)
        if shortest_path:
            path_mappings.append(((source_server, target_server, vnr['vnr_id']), shortest_path))
            for i in range(len(shortest_path) - 1):
                custom_print(f"Before reduction: Link {shortest_path[i]} <-> {shortest_path[i + 1]}, BW: {graph[shortest_path[i]][shortest_path[i + 1]]['bandwidth']}")

                graph[shortest_path[i]][shortest_path[i + 1]]['bandwidth'] -= bandwidth_demand
                graph[shortest_path[i + 1]][shortest_path[i]]['bandwidth'] -= bandwidth_demand

                custom_print(f"After reduction: Link {shortest_path[i]} <-> {shortest_path[i + 1]}, BW: {graph[shortest_path[i]][shortest_path[i + 1]]['bandwidth']}")

                link_flags[(shortest_path[i], shortest_path[i + 1])] = True
                link_flags[(shortest_path[i + 1], shortest_path[i])] = True

            custom_print(f"Successfully embedded link from VM{vm_source + 1} to VM{vm_target + 1} with path: {shortest_path}")
        else:
            custom_print(f"Failed to embed link from VM{vm_source + 1} to VM{vm_target + 1} due to insufficient bandwidth.")
            embedding_success[vnr['vnr_id']] = False
            break

    if embedding_success[vnr['vnr_id']]:
        custom_print(f"All links for VNR {vnr['vnr_id'] + 1} successfully embedded.")
    else:
        custom_print(f"Link embedding failed for VNR {vnr['vnr_id'] + 1}.")

    return embedding_success, graph, path_mappings


def rollback_failed_embeddings(vnr, vm_to_server_assignments, embedding_success, servers):
    vnr_id = vnr['vnr_id']
    custom_print(f"\nStarting the rollback process for VNR ID: {vnr_id + 1}...")
    if not embedding_success.get(vnr_id, True):  # If VNR embedding failed
        for vm_assignment in list(vm_to_server_assignments.items()):
            vm_id, server_id = vm_assignment
            if f"VNR{vnr_id + 1}" in vm_id:  # If VM belongs to the failed VNR
                vm_index = int(vm_id.split('_')[0][2:])  # Extract VM index from VM ID
                vm_cpu_demand = [v['cpu'] for v in servers[server_id]['vms'] if
                                 v['vnr_id'] == vnr_id and v['vm_index'] == vm_index]
                if vm_cpu_demand:
                    vm_cpu_demand = vm_cpu_demand[0]  # Assume only one match, get the CPU demand
                    servers[server_id]['cpu'] += vm_cpu_demand  # Release CPU resources on the server

                    # Correctly update VM list, removing VMs belonging to the failed VNR
                    servers[server_id]['vms'] = [v for v in servers[server_id]['vms'] if
                                                 not (v['vnr_id'] == vnr_id and v['vm_index'] == vm_index)]
                    del vm_to_server_assignments[vm_id]  # Remove this VM from assignments
                    custom_print(
                        f"Released {vm_cpu_demand} CPU units for {server_id}. New available CPU: {servers[server_id]['cpu']}")
    else:
        custom_print(f"No rollback needed for VNR ID: {vnr_id + 1}")

    custom_print("\nFinal Updated Server CPU Resources and VM Assignments:")
    for server_id, server_info in servers.items():
        assigned_vms_formatted = [(v['vnr_id'] + 1, v['vm_index']) for v in server_info['vms']]
        custom_print(f"{server_id}: CPU remaining {server_info['cpu']}, Assigned VMs: {assigned_vms_formatted}")
    custom_print("Rollback process completed.")

def initialize_structures(sn_topology):
    servers = {f'h{i + 1}': {'cpu': sn_topology[f'h{i + 1}']['allocated_cores'],
                             'original_cpu': sn_topology[f'h{i + 1}']['allocated_cores'], 'vms': []}
               for i in range(sn_topology['num_hosts'])}

    graph = {}
    for link in sn_topology['links_details']:
        node1, node2, bw = link['node1'], link['node2'], link['assigned_bandwidth']
        if node1 not in graph:
            graph[node1] = {}
        if node2 not in graph:
            graph[node2] = {}
        graph[node1][node2] = {'bandwidth': bw}
        graph[node2][node1] = {'bandwidth': bw}  # Assume undirected graph

    link_flags = {(node1, node2): False for link in sn_topology['links_details'] for node1, node2 in
                  [(link['node1'], link['node2']), (link['node2'], link['node1'])]}
    return servers, graph, link_flags


def main():
    vnr_info = json.loads(sys.argv[1])
    SN_data = json.loads(sys.argv[2])
    idx = int(sys.argv[3])
    vnr = json.loads(sys.argv[4])

    output_file_name = 'Node & Link Embedding Details.pickle'

    # Initialize the servers and the network graph using the helper function
    servers, graph, link_flags = initialize_structures(SN_data)

    all_embedding_success = True
    all_path_mappings = []
    all_embedding_results = []

    initial_total_bandwidth = calculate_total_bandwidth(graph)  # Calculate initial total bandwidth

    custom_print(f"\nProcessing Node and Link Embeddings for VNR ID: {vnr['vnr_id'] + 1}")
    vm_to_server_assignments, _, servers = node_embedding_and_mapping(servers, vnr)
    embedding_success, graph, path_mappings = link_embedding_and_mapping(graph, vnr, vm_to_server_assignments, link_flags)

    all_path_mappings.extend(path_mappings)
    all_embedding_results.append((vnr, embedding_success))

    if not all(embedding_success.values()):
        custom_print(f"Embedding failed for VNR ID: {vnr['vnr_id'] + 1}. Rolling back.")
        rollback_failed_embeddings(vnr, vm_to_server_assignments, embedding_success, servers)
        all_embedding_success = False

    final_total_bandwidth = calculate_total_bandwidth(graph)  # Calculate final total bandwidth

    # Save embedding results to a pickle file
    embedding_data = [list(vm_to_server_assignments.items()), all_path_mappings, list(link_flags.items()), all_embedding_success, graph, initial_total_bandwidth, final_total_bandwidth]

    with open(output_file_name, 'wb') as file:
        pickle.dump(embedding_data, file)

    custom_print(
        f"One or more VNR embeddings {'succeeded' if all_embedding_success else 'failed'}, check logs for details.")

if __name__ == "__main__":
    main()
