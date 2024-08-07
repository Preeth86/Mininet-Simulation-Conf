import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
import networkx as nx
import json
import sys
import pickle
import heapq
def map_virtual_nodes_with_degree(weight_matrix, virtual_data, substrate_data, clusters, croi, node_list, graph, servers):
    node_mapping_info = {}
    allocated_hosts = {host: False for host in node_list}
    node_degrees = dict(graph.degree(node_list)) if graph else {node: 0 for node in node_list}  # Handle None graph

    for vnr_id, vnr in enumerate([virtual_data]):  # Adjusted to handle single VNR input correctly
        node_mapping_info[vnr_id] = {}
        vm_cores = vnr['vm_cpu_cores']
        if len(vm_cores) > len(node_list):
            print("Mapping unsuccessful: More VMs than substrate nodes")
            return None, None

        # Sort hosts based on degree (descending order) and region of interest match
        sorted_hosts = sorted(node_list, key=lambda x: (clusters[node_list.index(x)] == croi, node_degrees[x]),
                              reverse=True)

        for vm_index, vm_core in enumerate(vm_cores):
            mapped = False
            print(f"Mapping VM {vm_index} with {vm_core} cores")
            for host in sorted_hosts:
                print(f"Checking host {host} with {substrate_data[host]['allocated_cores']} allocated cores")
                if not allocated_hosts[host] and substrate_data[host]['allocated_cores'] >= vm_core:
                    node_mapping_info[vnr_id][vm_index] = host
                    allocated_hosts[host] = True
                    substrate_data[host]['allocated_cores'] -= vm_core
                    servers[host]['cpu'] -= vm_core
                    mapped = True
                    print(f"VM {vm_index} mapped to host {host}")
                    break
            if not mapped:
                print(f"Mapping unsuccessful: No suitable host found for VM {vm_index}")
                return None, None

        # Reset the allocated_hosts dictionary for the next VNR request
        allocated_hosts = {host: False for host in node_list}

    return node_mapping_info, servers


# Example usage in your main function:
# node_mapping_info = map_virtual_nodes_with_degree(None, virtual_requests, substrate_data, clusters, croi, nodes, graph)

def calculate_weight(data_i, data_j):
    weight = abs(data_i['allocated_cores'] - data_j['allocated_cores'])
    return weight


def construct_weight_matrix(substrate_data, node_list):
    num_nodes = len(node_list)
    weight_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = calculate_weight(substrate_data[node_list[i]], substrate_data[node_list[j]])
            weight_matrix[i, j] = weight
            weight_matrix[j, i] = weight

    return weight_matrix


def laplacian_matrix(weight_matrix):
    degree_matrix = np.diag(np.sum(weight_matrix, axis=1))
    laplacian = degree_matrix - weight_matrix
    return laplacian


def normalize_laplacian(laplacian, degree_matrix):
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(degree_matrix)))
    normalized_laplacian = np.dot(np.dot(d_inv_sqrt, laplacian), d_inv_sqrt)
    return normalized_laplacian


def partition_substrate_network(substrate_data, k1):
    node_list = [node for node in substrate_data.keys() if
                 isinstance(substrate_data[node], dict) and 'allocated_cores' in substrate_data[node]]
    weight_matrix = construct_weight_matrix(substrate_data, node_list)
    laplacian = laplacian_matrix(weight_matrix)
    degree_matrix = np.diag(np.sum(weight_matrix, axis=1))
    norm_laplacian = normalize_laplacian(laplacian, degree_matrix)

    eigvals, eigvecs = eigsh(norm_laplacian, k=k1, which='SM')

    kmeans = KMeans(n_clusters=k1)
    clusters = kmeans.fit_predict(eigvecs)

    resources = [
        np.sum([substrate_data[node_list[i]]['allocated_cores'] for i in range(len(node_list)) if clusters[i] == j])
        for j in range(k1)]
    croi = np.argmax(resources)

    return clusters, croi, node_list


def map_virtual_links(virtual_requests, node_mapping_info, substrate_data, graph):
    link_mapping_info = {}

    for vnr_id, vnr in enumerate(virtual_requests):
        link_mapping_info[vnr_id] = []
        for (vm1, vm2), bandwidth in zip(vnr['vm_links'], vnr['bandwidth_values']):
            substrate_node1 = node_mapping_info[vnr_id][vm1]
            substrate_node2 = node_mapping_info[vnr_id][vm2]

            # Find a path between the mapped substrate nodes
            try:
                path = nx.shortest_path(graph, source=substrate_node1, target=substrate_node2, weight='weight')
                print("")
                print(
                    f"Found path for VNR {vnr_id + 1} VM link {vm1} <-> {vm2} between {substrate_node1} and {substrate_node2}: {path}")
            except nx.NetworkXNoPath:
                print(
                    f"No path found for VNR {vnr_id + 1} VM link {vm1} <-> {vm2} between {substrate_node1} and {substrate_node2}")
                continue

            # Check if all links in the path have sufficient bandwidth
            can_allocate_bandwidth = True
            for u, v in zip(path[:-1], path[1:]):
                link = next((l for l in substrate_data['links_details'] if
                             (l['node1'] == u and l['node2'] == v) or (l['node1'] == v and l['node2'] == u)), None)
                if link is None or link['assigned_bandwidth'] < bandwidth:
                    can_allocate_bandwidth = False
                    break

            if can_allocate_bandwidth:
                for u, v in zip(path[:-1], path[1:]):
                    link = next((l for l in substrate_data['links_details'] if
                                 (l['node1'] == u and l['node2'] == v) or (l['node1'] == v and l['node2'] == u)), None)
                    if link:
                        link['assigned_bandwidth'] -= bandwidth
                        link_mapping_info[vnr_id].append((u, v, bandwidth))
                print(
                    f"VNR {vnr_id + 1} VM link {vm1} <-> {vm2} mapped on substrate path {path} with bandwidth {bandwidth}")
            else:
                print(
                    f"Insufficient bandwidth to map VNR {vnr_id + 1} VM link {vm1} <-> {vm2} with bandwidth {bandwidth}")

    return link_mapping_info


def construct_graph(sn):
    graph = nx.Graph()
    for link in sn['links_details']:
        graph.add_edge(link['node1'], link['node2'], weight=1 / link['assigned_bandwidth'])
    return graph


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

    link_flags = {(link['node1'], link['node2']): False for link in sn_topology['links_details']}
    return servers, graph, link_flags


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
    print(f"\nLink Embedding and Mapping of Virtual Links for VNR ID: {vnr['vnr_id'] + 1} using Dijkstra's Algorithm:")
    embedding_success = {vnr['vnr_id']: True}
    path_mappings = []

    for link_index, (vm_source, vm_target) in enumerate(vnr['vm_links'], start=1):
        bandwidth_demand = vnr['bandwidth_values'][link_index - 1]
        print(f"VM Source: {vm_source}, VM Target: {vm_target}")
        print(f"VM to Server Assignments: {vm_to_server_assignments}")

        if f"VM{vm_source + 1}" not in vm_to_server_assignments or f"VM{vm_target + 1}" not in vm_to_server_assignments:
            print(f"Failed to find server assignments for VM{vm_source + 1} or VM{vm_target + 1}.")
            embedding_success[vnr['vnr_id']] = False
            break

        source_server = vm_to_server_assignments[f"VM{vm_source + 1}"]
        target_server = vm_to_server_assignments[f"VM{vm_target + 1}"]

        shortest_path = dijkstra(graph, source_server, target_server, bandwidth_demand)
        if shortest_path:
            path_mappings.append(((source_server, target_server, vnr['vnr_id']), shortest_path))
            for i in range(len(shortest_path) - 1):
                print(f"Before reduction: Link {shortest_path[i]} <-> {shortest_path[i + 1]}, BW: {graph[shortest_path[i]][shortest_path[i + 1]]['bandwidth']}")

                graph[shortest_path[i]][shortest_path[i + 1]]['bandwidth'] -= bandwidth_demand
                graph[shortest_path[i + 1]][shortest_path[i]]['bandwidth'] -= bandwidth_demand

                print(f"After reduction: Link {shortest_path[i]} <-> {shortest_path[i + 1]}, BW: {graph[shortest_path[i]][shortest_path[i + 1]]['bandwidth']}")

                link_flags[(shortest_path[i], shortest_path[i + 1])] = True
                link_flags[(shortest_path[i + 1], shortest_path[i])] = True

            print(f"Successfully embedded link from VM{vm_source + 1} to VM{vm_target + 1} with path: {shortest_path}")
        else:
            print(f"Failed to embed link from VM{vm_source + 1} to VM{vm_target + 1} due to insufficient bandwidth.")
            embedding_success[vnr['vnr_id']] = False
            break

    if embedding_success[vnr['vnr_id']]:
        print(f"All links for VNR {vnr['vnr_id'] + 1} successfully embedded.")
    else:
        print(f"Link embedding failed for VNR {vnr['vnr_id'] + 1}.")

    return embedding_success, graph, path_mappings


def rollback_failed_embeddings(vnr, vm_to_server_assignments, embedding_success, servers):
    vnr_id = vnr['vnr_id']
    print(f"\nStarting the rollback process for VNR ID: {vnr_id + 1}...")
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
                    print(f"Released {vm_cpu_demand} CPU units for {server_id}. New available CPU: {servers[server_id]['cpu']}")
    else:
        print(f"No rollback needed for VNR ID: {vnr_id + 1}")

    print("\nFinal Updated Server CPU Resources and VM Assignments:")
    for server_id, server_info in servers.items():
        assigned_vms_formatted = [(v['vnr_id'] + 1, v['vm_index']) for v in server_info['vms']]
        print(f"{server_id}: CPU remaining {server_info['cpu']}, Assigned VMs: {assigned_vms_formatted}")
    print("Rollback process completed.")
    

def calculate_total_bandwidth(graph):
    total_bandwidth = 0
    for node in graph:
        for neighbor in graph[node]:
            total_bandwidth += graph[node][neighbor]['bandwidth']
    return total_bandwidth / 2  # Since it's an undirected graph


def main():
    vnr_info = json.loads(sys.argv[1])
    SN_data = json.loads(sys.argv[2])
    idx = int(sys.argv[3])
    vnr = json.loads(sys.argv[4])

    output_file_name = 'Node & Link Embedding Details.pickle'
    servers, graph, link_flags = initialize_structures(SN_data)
    initial_total_bandwidth = calculate_total_bandwidth(graph)
    k1 = 2
    clusters, croi, nodes = partition_substrate_network(SN_data, k1)

    graph1 = construct_graph(SN_data)

    node_mapping_info, servers = map_virtual_nodes_with_degree(None, vnr, SN_data, clusters, croi, nodes, graph1, servers)

    if node_mapping_info is None or servers is None:
        print("Node mapping failed.")
        return

    vm_to_server_assignments = {}
    all_embedding_success = True
    all_path_mappings = {}
    all_embedding_results = []
    for ind, ass in node_mapping_info[0].items():
        vm_to_server_assignments[f'VM{ind+1}'] = ass

    embedding_success, graph, path_mappings = link_embedding_and_mapping(graph, vnr, vm_to_server_assignments, link_flags)

    for key, path in path_mappings:
        all_path_mappings[key] = path
    all_embedding_results.append((vnr, embedding_success))
    if not all(embedding_success.values()):
        print(f"Embedding failed for VNR ID: {vnr['vnr_id'] + 1}. Rolling back.")
        rollback_failed_embeddings(vnr, vm_to_server_assignments, embedding_success, servers)
        all_embedding_success = False

    final_total_bandwidth = calculate_total_bandwidth(graph)  # Calculate final total bandwidth

    # Save embedding results to a pickle file
    embedding_data = [list(vm_to_server_assignments.items()), list(all_path_mappings.items()), list(link_flags.items()),
                      all_embedding_success, graph, initial_total_bandwidth, final_total_bandwidth]

    with open(output_file_name, 'wb') as file:
        pickle.dump(embedding_data, file)

    print(
        f"One or more VNR embeddings {'succeeded' if all_embedding_success else 'failed'}, check logs for details.")


if __name__ == "__main__":
    main()
