import pickle
import heapq
import os
import sys
import json

def k_shortest_paths(graph, start, end, k=1):
    paths = [(0, [start])]
    shortest_paths = []
    while paths and len(shortest_paths) < k:
        (cost, path) = heapq.heappop(paths)
        last_node = path[-1]
        if last_node == end:
            shortest_paths.append((cost, path))
            continue
        for next_node, weight in graph[last_node].items():
            if next_node not in path:
                heapq.heappush(paths, (cost + weight, path + [next_node]))
    return shortest_paths

output = []

def calculate_total_bandwidth(graph):
    total_bandwidth = 0
    for node in graph:
        for neighbor in graph[node]:
            total_bandwidth += graph[node][neighbor]
    return total_bandwidth / 2  # Since it's an undirected graph

def custom_print(*args):
    message = ' '.join(str(arg) for arg in args)
    print(message)
    output.append([message])

def load_topology_from_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def node_embedding_and_mapping(servers, vnr):
    custom_print(f"\nNode Embedding and Mapping of VMs for VNR ID: {vnr['vnr_id'] + 1}")
    vm_to_server_assignments = {}
    vnr_to_server_assignments = {}

    host_flag = {key: True for key in servers}

    for vm_index, vm_cpu in enumerate(vnr['vm_cpu_cores'], start=1):
        first_fit = None
        for server_id in servers:
            if host_flag[server_id] and servers[server_id]['cpu'] >= vm_cpu:
                first_fit = server_id
                break
        if first_fit is None:
            custom_print(f"No suitable server found for VM{vm_index}.")
            return {}, {}, servers
        else:
            host_flag[first_fit] = False
            servers[first_fit]['cpu'] -= vm_cpu
            servers[first_fit]['vms'].append({'vnr_id': vnr['vnr_id'], 'vm_index': vm_index, 'cpu': vm_cpu})
            vm_to_server_assignments[f"VM{vm_index}"] = first_fit
            vnr_to_server_assignments.setdefault(vnr['vnr_id'], []).append(first_fit)

    return vm_to_server_assignments, vnr_to_server_assignments, servers

def link_embedding_and_mapping(graph, vnr, vm_to_server_assignments, link_flags):
    custom_print(f"\nLink Embedding and Mapping of Virtual Links for VNR ID: {vnr['vnr_id'] + 1} using Shortest Paths:")

    embedding_success = {vnr['vnr_id']: True}
    path_mappings = {}

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

        shortest_paths = k_shortest_paths(graph, source_server, target_server, k=1)
        path_found = False
        for cost, path in shortest_paths:
            if all(graph[path[i]][path[i + 1]] >= bandwidth_demand for i in range(len(path) - 1)):
                path_found = True
                path_mappings[(source_server, target_server, vnr['vnr_id'])] = path
                for i in range(len(path) - 1):
                    # Debug print before bandwidth reduction
                    custom_print(f"Before reduction: Link {path[i]} <-> {path[i + 1]}, BW: {graph[path[i]][path[i + 1]]}")

                    graph[path[i]][path[i + 1]] -= bandwidth_demand
                    graph[path[i + 1]][path[i]] -= bandwidth_demand

                    # Debug print after bandwidth reduction
                    custom_print(f"After reduction: Link {path[i]} <-> {path[i + 1]}, BW: {graph[path[i]][path[i + 1]]}")

                    link_flags[(path[i], path[i + 1])] = True  # Update link flags
                    link_flags[(path[i + 1], path[i])] = True  # Since the graph is undirected

                custom_print(
                    f"Successfully embedded link from VM{vm_source + 1} to VM{vm_target + 1} with path: {path}")
                break

        if not path_found:
            custom_print(
                f"Failed to embed link from VM{vm_source + 1} to VM{vm_target + 1} due to insufficient bandwidth.")
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
                vm_index = int(vm_id.split('M')[1])  # Extract VM index from VM ID
                vm_cpu_demand = [v['cpu'] for v in servers[server_id]['vms'] if v['vnr_id'] == vnr_id and v['vm_index'] == vm_index]
                if vm_cpu_demand:
                    vm_cpu_demand = vm_cpu_demand[0]  # Assume only one match, get the CPU demand
                    servers[server_id]['cpu'] += vm_cpu_demand  # Release CPU resources on the server

                    # Correctly update VM list, removing VMs belonging to the failed VNR
                    servers[server_id]['vms'] = [v for v in servers[server_id]['vms'] if not (v['vnr_id'] == vnr_id and v['vm_index'] == vm_index)]
                    del vm_to_server_assignments[vm_id]  # Remove this VM from assignments
                    custom_print(f"Released {vm_cpu_demand} CPU units for {server_id}. New available CPU: {servers[server_id]['cpu']}")
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
        graph[node1][node2] = bw
        graph[node2][node1] = bw  # Assume undirected graph

    link_flags = {(link['node1'], link['node2']): False for link in sn_topology['links_details']}
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
    all_path_mappings = {}
    all_embedding_results = []

    initial_total_bandwidth = calculate_total_bandwidth(graph)  # Calculate initial total bandwidth

    custom_print(f"\nProcessing Node and Link Embeddings for VNR ID: {vnr['vnr_id'] + 1}")
    vm_to_server_assignments, _, servers = node_embedding_and_mapping(servers, vnr)
    embedding_success, graph, path_mappings = link_embedding_and_mapping(graph, vnr, vm_to_server_assignments, link_flags)

    all_path_mappings.update(path_mappings)
    all_embedding_results.append((vnr, embedding_success))

    if not all(embedding_success.values()):
        custom_print(f"Embedding failed for VNR ID: {vnr['vnr_id'] + 1}. Rolling back.")
        rollback_failed_embeddings(vnr, vm_to_server_assignments, embedding_success, servers)
        all_embedding_success = False

    final_total_bandwidth = calculate_total_bandwidth(graph)  # Calculate final total bandwidth

    # Save embedding results to a pickle file
    embedding_data = [list(vm_to_server_assignments.items()), list(path_mappings.items()), list(link_flags.items()), all_embedding_success, graph, initial_total_bandwidth, final_total_bandwidth]

    with open(output_file_name, 'wb') as file:
        pickle.dump(embedding_data, file)

    custom_print(
        f"One or more VNR embeddings {'succeeded' if all_embedding_success else 'failed'}, check logs for details.")

if __name__ == "__main__":
    main()
