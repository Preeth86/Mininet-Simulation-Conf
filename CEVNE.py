import pickle
from pulp import *
import networkx as nx
import json
import sys
import heapq

output = []

def k_shortest_paths(graph, start, end, k=1):
    paths = [(0, [start])]
    shortest_paths = []
    while paths and len(shortest_paths) < k:
        (cost, path) = heapq.heappop(paths)
        last_node = path[-1]
        if last_node == end:
            shortest_paths.append((cost, path))
            continue
        for next_node, data in graph[last_node].items():
            weight = data['bandwidth']
            if next_node not in path:
                heapq.heappush(paths, (cost + weight, path + [next_node]))
    return shortest_paths

def custom_print(*args):
    message = ' '.join(str(arg) for arg in args)
    print(message)
    output.append([message])

def process_subnet(substrate_network_data):
    substrate_info = {}
    for sn, host_info in substrate_network_data.items():
        if not sn.startswith('h'):
            continue
        substrate_info[sn] = {'cpu': host_info['allocated_cores']}
    return substrate_info

def create_clusters(vnr_info, data):
    custom_print("\nSubstrate Network CPU Available Details Before Clustering:")
    for node_name, info in data.items():
        if node_name.startswith('h'):
            custom_print(f"{node_name}: {info['allocated_cores']}")
    substrate_info = process_subnet(data)
    clusters = {}
    for vnr_name, vms_info in vnr_info.items():
        min_vm_cpu = min(vm_info['cpu'] for vm_info in vms_info.values())
        substrate_info_filtered = {sn: info for sn, info in substrate_info.items() if info['cpu'] > min_vm_cpu}
        if sum(len(vm) for vm in vms_info.values()) > len(substrate_info_filtered):
            custom_print(f"\nNumber of Virtual Machines in VNR {vnr_name} is greater than the number of available Substrate Nodes.")
            custom_print("Hence, mapping cannot happen.")
            return None
    for vnr_name, vms_info in vnr_info.items():
        clusters[vnr_name] = {vm_name: [] for vm_name in vms_info}
        for vm_name, vm_info in vms_info.items():
            for sn, sn_info in substrate_info.items():
                if sn_info['cpu'] >= vm_info['cpu']:
                    clusters[vnr_name][vm_name].append(sn)
    custom_print("\nCluster Information:")
    for vnr_name, vnr_clusters in clusters.items():
        custom_print(f"{vnr_name}:")
        for vm_name, substrate_list in vnr_clusters.items():
            custom_print(f"  {vm_name}: {substrate_list}")
    return clusters

def solve_optimization(idx, clusters):
    pairs = [(vm_name, substrate_node) for vnr_name, vnr_clusters in clusters.items()
             for vm_name, substrate_list in vnr_clusters.items()
             for substrate_node in substrate_list]
    x = LpVariable.dicts('x', pairs, 0, 1, LpBinary)
    prob = LpProblem(f"VNR{idx}_Substrate_Mapping", LpMinimize)

    prob += lpSum(x[vm, substrate] for vm, substrate in pairs)

    for vnr_name, vnr_clusters in clusters.items():
        for vm_name, substrate_list in vnr_clusters.items():
            prob += lpSum(x[vm_name, substrate] for substrate in substrate_list) == 1

    for substrate in set(substrate_node for vnr_clusters in clusters.values()
                         for substrate_list in vnr_clusters.values()
                         for substrate_node in substrate_list):
        prob += lpSum(x[vm, substrate] for vnr_clusters in clusters.values()
                      for vm, substrate_list in vnr_clusters.items()
                      if substrate in substrate_list) <= 1

    pulp.LpSolverDefault.msg = False
    prob.solve()

    return x, prob

def print_mapping(idx, clusters, x):
    custom_print("\nNode Mapping Details:")
    i = 0
    h = []
    for vm_name, substrate_list in clusters[f'VNR{idx}'].items():
        mapped = False
        for substrate in substrate_list:
            if value(x[vm_name, substrate]) == 1:
                custom_print(f"Virtual Node: {vm_name} is mapped to Substrate Node: {substrate}")
                h.append((vm_name, substrate))
                mapped = True
                break
        if not mapped:
            i = -1
            custom_print(f"\nVirtual Node: {vm_name} is not mapped to any Substrate Node.")
    return i, h

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
        graph[node2][node1] = {'bandwidth': bw}

    link_flags = {(link['node1'], link['node2']): False for link in sn_topology['links_details']}
    return servers, graph, link_flags

def calculate_total_bandwidth(graph):
    total_bandwidth = sum(data['bandwidth'] for node in graph for neighbor, data in graph[node].items()) / 2
    return total_bandwidth

def link_embedding_and_mapping(graph, vnr, vm_to_server_assignments, link_flags):
    custom_print(f"\nLink Embedding and Mapping of Virtual Links for VNR ID: {vnr['vnr_id'] + 1} using k-Shortest Paths:")
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

        shortest_paths = k_shortest_paths(graph, source_server, target_server, k=3)
        path_found = False
        for cost, path in shortest_paths:
            if all(graph[path[i]][path[i + 1]]['bandwidth'] >= bandwidth_demand for i in range(len(path) - 1)):
                path_found = True
                path_mappings.append(((source_server, target_server, vnr['vnr_id']), path))
                for i in range(len(path) - 1):
                    custom_print(f"Before reduction: Link {path[i]} <-> {path[i + 1]}, BW: {graph[path[i]][path[i + 1]]['bandwidth']}")
                    graph[path[i]][path[i + 1]]['bandwidth'] -= bandwidth_demand
                    graph[path[i + 1]][path[i]]['bandwidth'] -= bandwidth_demand
                    custom_print(f"After reduction: Link {path[i]} <-> {path[i + 1]}, BW: {graph[path[i]][path[i + 1]]['bandwidth']}")
                    link_flags[(path[i], path[i + 1])] = True
                    link_flags[(path[i + 1], path[i])] = True
                custom_print(f"Successfully embedded link from VM{vm_source + 1} to VM{vm_target + 1} with path: {path}")
                break

        if not path_found:
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
    if not embedding_success.get(vnr_id, True):
        for vm_assignment in list(vm_to_server_assignments.items()):
            vm_id, server_id = vm_assignment
            if f"VNR{vnr_id + 1}" in vm_id:
                vm_index = int(vm_id.split('M')[1])
                vm_cpu_demand = [v['cpu'] for v in servers[server_id]['vms'] if v['vnr_id'] == vnr_id and v['vm_index'] == vm_index]
                if vm_cpu_demand:
                    vm_cpu_demand = vm_cpu_demand[0]
                    servers[server_id]['cpu'] += vm_cpu_demand
                    servers[server_id]['vms'] = [v for v in servers[server_id]['vms'] if not (v['vnr_id'] == vnr_id and v['vm_index'] == vm_index)]
                    del vm_to_server_assignments[vm_id]
                    custom_print(f"Released {vm_cpu_demand} CPU units for {server_id}. New available CPU: {servers[server_id]['cpu']}")
    else:
        custom_print(f"No rollback needed for VNR ID: {vnr_id + 1}")

    custom_print("\nFinal Updated Server CPU Resources and VM Assignments:")
    for server_id, server_info in servers.items():
        assigned_vms_formatted = [(v['vnr_id'] + 1, v['vm_index']) for v in server_info['vms']]
        custom_print(f"{server_id}: CPU remaining {server_info['cpu']}, Assigned VMs: {assigned_vms_formatted}")
    custom_print("Rollback process completed.")

def main():
    vnr_info = json.loads(sys.argv[1])
    SN_data = json.loads(sys.argv[2])
    idx = int(sys.argv[3])
    vnr = json.loads(sys.argv[4])

    output_file_name = 'Node & Link Embedding Details.pickle'

    servers, graph, link_flags = initialize_structures(SN_data)

    all_embedding_success = True
    all_path_mappings = {}
    all_embedding_results = []

    initial_total_bandwidth = calculate_total_bandwidth(graph)

    custom_print(f"\nProcessing Node and Link Embeddings for VNR ID: {vnr['vnr_id'] + 1}")

    clusters = create_clusters(vnr_info, SN_data)
    if clusters is not None:
        x, prob = solve_optimization(idx, clusters)
        p, vm_to_server_assignments = print_mapping(idx, clusters, x)
        if p == 0:
            vm_to_server_assignments_dict = dict(vm_to_server_assignments)  # Ensure vm_to_server_assignments is a dictionary
            embedding_success, graph, path_mappings = link_embedding_and_mapping(graph, vnr, vm_to_server_assignments_dict, link_flags)
            for key, path in path_mappings:
                all_path_mappings[key] = path
            all_embedding_results.append((vnr, embedding_success))
            if not all(embedding_success.values()):
                custom_print(f"Embedding failed for VNR ID: {vnr['vnr_id'] + 1}. Rolling back.")
                rollback_failed_embeddings(vnr, vm_to_server_assignments_dict, embedding_success, servers)
                all_embedding_success = False
            final_total_bandwidth = calculate_total_bandwidth(graph)  # Calculate final total bandwidth
            embedding_data = [list(vm_to_server_assignments_dict.items()), list(all_path_mappings.items()), list(link_flags.items()), all_embedding_success, graph, initial_total_bandwidth, final_total_bandwidth]
            with open(output_file_name, 'wb') as file:
                pickle.dump(embedding_data, file)
            custom_print(f"One or more VNR embeddings {'succeeded' if all_embedding_success else 'failed'}, check logs for details.")
        else:
            custom_print(f"\nNode Embedding is failed. Hence VNR{idx} rejected.")
            with open(output_file_name, 'wb') as file:
                pickle.dump({}, file)
    else:
        custom_print(f"\nClusters could not be created. Hence VNR{idx} rejected.")
        with open(output_file_name, 'wb') as file:
            pickle.dump({}, file)

if __name__ == "__main__":
    main()
