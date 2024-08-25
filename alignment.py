
import numpy as np


type_cats = {'Continent': 10,
 'Country': 9,
 'County': 8,
 'State': 8,
 'Region': 8,
 'District': 8,
 'Suburb': 7,
 'Spa Town': 7,
 'Village': 7,
 'Province': 7,
 'Town': 7,
 'City': 7,
 'Borough': 7,
 'ArmyCamp': 4,
 'Army Camp': 4,
 'DP Camp': 3,
 'Place': 5,
 'Location': 5,
 'Language': 5,
 'Significant Location': 5,
 'Common': 5,
 'Mass Grave Site': 5,
 'ConcentrationCamp': 0,
 'Concentration Camp': 0,
 'Camp': 0,
 'Ghetto': 1,
 'Death Camp': 2,
 'DeathCamp': 2,
 'School': 6,
 'Institution': 6,
 'Lake': 6,
 'River': 6,
 'Company': 6,
 'University': 6,
 'Street': 6,
 'Legislation': 6,
 'Facility': 6,
 'Ship': 6,
 'Mountain Range': 6,
 'SignificantFacility': 6,
 'Significant Facility': 6,
 'HistoricalPlace': 6}


def get_dist_func_paths(path="/cs/labs/oabend/eitan.wagner/events/testimonies/", get_paths=False, use_types=True):
    """
    Get the distance function for the graph G.
    :param path: The path to the graph file
    :return:
    """
    import networkx as nx
    import json
    G = nx.read_adjlist(path + "graph_gpt-4o-mini.adjlist", delimiter='*')
    index2node = list(G.nodes)
    node2index = {node: index for index, node in enumerate(index2node)}

    dists = dict(nx.all_pairs_shortest_path_length(nx.Graph(G)))
    with open("/cs/labs/oabend/eitan.wagner/events/testimonies/nodesgpt-4o-mini.json", 'r') as file:
        all_nodes = json.load(file)
    all_nodes = dict(all_nodes)

    def dist_func(node1, node2):
        """
        Get the distance between two nodes in the graph. with indices.
        """
        if use_types:
            types = all_nodes[index2node[node1]]["type"], all_nodes[index2node[node2]]["type"]
            if type_cats.get(types[0], 6) == type_cats.get(types[1], 6) and type_cats.get(types[0], 6) < 5:  # both Holocaust related and the same type
                return dists[index2node[node1]].get(index2node[node2], float('inf')) * 0.2
            elif (type_cats.get(types[0], 6) < 5) != (type_cats.get(types[1], 6) < 5):  # one Holocaust related and the other not
                return dists[index2node[node1]].get(index2node[node2], float('inf')) * 2
        return dists[index2node[node1]].get(index2node[node2], 10000) / 4

    if get_paths:
        import json
        with open(path + "graphs_gpt-4o-mini.json", 'r') as file:
            d = json.load(file)
        paths = {k: v["path"] for k, v in d.items()}
        return dist_func, paths, node2index

    return dist_func


def compress_sequence(seq):
    """
    Compress the sequence by combining consecutive subsequences with the same values.
    Args:
    seq (list): Input sequence.
    Returns:
    list: Compressed sequence and counts of consecutive values.
    """
    if not seq:
        return [], []
    compressed_seq = [seq[0]]
    counts = [1]
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            counts[-1] += 1
        else:
            compressed_seq.append(seq[i])
            counts.append(1)
    return compressed_seq, counts

def expand_alignment(alignment, counts1, counts2):
    """
    Expand the alignment to match the original sequences.
    Args:
    alignment (list): Compressed sequence alignment.
    counts1 (list): Counts of consecutive values in the first sequence.
    counts2 (list): Counts of consecutive values in the second sequence.
    Returns:
    list: Expanded alignment.
    """
    expanded_alignment = []
    added1 = 0
    added2 = 0
    for ai, (i, j) in enumerate(alignment):
        for k in range(max(counts1[i], counts2[j])):
            expanded_alignment.append((i + added1 + min(k, counts1[i]-1), j + added2 + min(k, counts2[j]-1)))
        if ai < len(alignment) - 1 and alignment[ai + 1][0] != i:
            added1 += counts1[i] - 1
        if ai < len(alignment) - 1 and alignment[ai + 1][1] != j:
            added2 += counts2[j] - 1
        # for _ in range(counts1[i]):
        #     for _ in range(counts2[j]):
        #         expanded_alignment.append((i, j))
    return expanded_alignment


def dynamic_time_warping(seq1, seq2, dist_func):
    """
    Perform Dynamic Time Warping on two sequences.
    Args:
    seq1 (list): First sequence.
    seq2 (list): Second sequence.
    dist_func (function): Point-wise distance function between elements of the sequences.
    Returns:
    float: The DTW distance between the two sequences.
    list: The expanded alignment path that achieves the lowest distance.
    """
    compressed_seq1, counts1 = compress_sequence(seq1)
    compressed_seq2, counts2 = compress_sequence(seq2)
    # Initialize the DTW matrix.
    n, m = len(compressed_seq1), len(compressed_seq2)
    dtw_matrix = np.full((n + 1, m + 1), float('inf'))
    dtw_matrix[0, 0] = 0

    # Fill in the DTW matrix.
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = dist_func(compressed_seq1[i - 1], compressed_seq2[j - 1])
            # Find the minimum cost path to the current cell.
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],    # insertion
                                          dtw_matrix[i, j - 1],    # deletion
                                          dtw_matrix[i - 1, j - 1])  # match

    # Backtrack to find the alignment
    alignment = []
    i, j = n, m
    while i > 0 and j > 0:
        alignment.append((i - 1, j - 1))
        min_cost = min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])
        if min_cost == dtw_matrix[i - 1, j]:
            i -= 1
        elif min_cost == dtw_matrix[i, j - 1]:
            j -= 1
        else:
            i -= 1
            j -= 1

    alignment.reverse()
    expanded_alignment = expand_alignment(alignment, counts1, counts2)
    return dtw_matrix[n, m], expanded_alignment

def dynamic_time_warping2(seq1, seq2, dist_func):
    """
    Perform Dynamic Time Warping on two sequences.
    Args:
    seq1 (list): First sequence.
    seq2 (list): Second sequence.
    dist_func (function): Point-wise distance function between elements of the sequences.
    Returns:
    float: The DTW distance between the two sequences.
    list: The expanded alignment path that achieves the lowest distance.
    """
    n, m = len(seq1), len(seq2)
    dtw_matrix = np.full((n + 1, m + 1), float('inf'))
    dtw_matrix[0, 0] = 0

    # Fill in the DTW matrix.
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = dist_func(seq1[i - 1], seq2[j - 1])
            # Find the minimum cost path to the current cell.
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],    # insertion
                                          dtw_matrix[i, j - 1],    # deletion
                                          dtw_matrix[i - 1, j - 1])  # match

    # Backtrack to find the alignment
    alignment = []
    i, j = n, m
    while i > 0 and j > 0:
        alignment.append((i - 1, j - 1))
        min_cost = min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])
        if min_cost == dtw_matrix[i - 1, j]:  # insertion
            i -= 1
        elif min_cost == dtw_matrix[i, j - 1]:  # deletion
            j -= 1
        else:  # match
            i -= 1
            j -= 1

    alignment.reverse()
    return dtw_matrix[n, m], alignment


# *************************************************************

def edit_distance_with_alignment(seq1, seq2, dist_func=None, repeat_cost=1., return_dist=False):
    """
    Compute the edit distance between two sequences and return the aligned sequences using a provided point-wise distance function for replacements.

    :param seq1: First sequence
    :param seq2: Second sequence
    :param repeat_cost:
    :param dist_func: Function to compute the distance between two elements (default is 1 for replacement, 0 for same)
    :return: (edit_distance, aligned_seq1, aligned_seq2)
    """

    # Default distance function
    if dist_func is None:
        dist_func = lambda x, y: 0 if x == y else 1

    len_seq1 = len(seq1)
    len_seq2 = len(seq2)

    # Create a distance matrix
    dp = [[0] * (len_seq2 + 1) for _ in range(len_seq1 + 1)]

    # Initialize distance matrix
    dp[0][0] = 0
    for i in range(1, len_seq1):
        dp[i][0] = dp[i-1][0] + (repeat_cost if seq1[i-1] == seq1[i] else 1)
    for j in range(1, len_seq2):
        dp[0][j] = dp[0][j-1] + (repeat_cost if seq2[j-1] == seq2[j] else 1)

    # Compute the edit distance
    for i in range(1, len_seq1 + 1):
        r_cost1 = repeat_cost if i > 1 and seq1[i-1] == seq1[i-2] else 1
        for j in range(1, len_seq2 + 1):
            # if i == j == 4:
            #     print("here")
            r_cost2 = repeat_cost if j > 1 and seq2[j-1] == seq2[j-2] else 1
            insertion = dp[i][j-1] + 1 * r_cost1
            deletion = dp[i-1][j] + 1 * r_cost2
            replacement = dp[i-1][j-1] + dist_func(seq1[i-1], seq2[j-1])
            dp[i][j] = min(insertion, deletion, replacement)

    if return_dist:
        return dp[len_seq1][len_seq2]

    # Backtracking to find the alignment
    i, j = len_seq1, len_seq2
    aligned_seq1 = []
    aligned_seq2 = []

    while i > 0 or j > 0:
        r_cost1 = repeat_cost if i > 1 and seq1[i-1] == seq1[i-2] else 1
        r_cost2 = repeat_cost if j > 1 and seq2[j - 1] == seq2[j - 2] else 1
        if i > 0 and dp[i][j] == dp[i-1][j] + 1 * r_cost2:
            # Deletion in seq1
            # aligned_seq1.append(seq1[i-1])
            aligned_seq1.append(i-1)
            aligned_seq2.append(-1)
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1 * r_cost1:
            # Insertion in seq1
            aligned_seq1.append(-1)
            # aligned_seq2.append(seq2[j-1])
            aligned_seq2.append(j-1)
            j -= 1
        elif i > 0 and j > 0:
            # Replacement or no-op
            aligned_seq1.append(i-1)
            aligned_seq2.append(j-1)
            # aligned_seq1.append(seq1[i-1])
            # aligned_seq2.append(seq2[j-1])
            i -= 1
            j -= 1

    # Reverse to get the correct alignment
    aligned_seq1.reverse()
    aligned_seq2.reverse()

    return dp[len_seq1][len_seq2], list(zip(aligned_seq1, aligned_seq2))




# *************************************************************

def testimony_distances():
    """
    Compute the distances between testimonies based on their paths in the graph.
    :return:
    """
    # import json
    # with open(f"/cs/labs/oabend/eitan.wagner/events/testimonies/graphs_gpt-4o-mini.json",
    #           "r") as file:
    #     d = json.load(file)

    dist_func, paths3, node2index = get_dist_func_paths(get_paths=True)
    index2node = {v: k for k, v in node2index.items()}
    paths2 = {k: [node2index.get(_v[0], -1) for _v in v["nodes"]] for k, v in paths3.items()}
    paths = {k: [v for v in paths2[k] if v != -1] for k in paths2}

    # compute pairwise distances
    distances = {}
    for k1, v1 in list(paths.items()):
        distances[k1] = {}
        for k2, v2 in paths.items():
            if k1 == k2:
                continue
                # distances[k1][k2] = 0
            else:
                dist = edit_distance_with_alignment(v1, v2, dist_func, repeat_cost=0.5, return_dist=True)
                # distances[k1][k2] = dist / max(len(v1), len(v2))
                distances[k1][k2] = max(dist / len(v1), dist / len(v2))


    # find the closest and farthest neighbors using the matrix
    # closest = {}
    # farthest = {}
    closest = []
    farthest = []
    for k1, v1 in distances.items():
        closest.append((k1, min(v1, key=v1.get)))
        farthest.append((k1, max(v1, key=v1.get)))
        # closest[k1] = min(v1, key=v1.get)
        # farthest[k1] = max(v1, key=v1.get)

    # find five most close and most far. save the pair of ids
    closest = sorted(closest, key=lambda x: distances[x[0]][x[1]])
    farthest = sorted(farthest, key=lambda x: distances[x[0]][x[1]])

    most_close = closest[:5]
    most_far = farthest[-5:]

    # most_close = min(closest, key=lambda x: distances[x[0]][x[1]])
    # most_far = max(farthest, key=lambda x: distances[x[0]][x[1]])
    # most_close = min(closest, key=lambda x: distances[x][closest[x]])
    # most_far = max(farthest, key=lambda x: distances[x][farthest[x]])

    # most_close = min(closest, key=lambda x: distances[x][closest[x]])
    # most_far = max(farthest, key=lambda x: distances[x][farthest[x]])

    print(f"\n\nMost close: {most_close}")
    for k1, k2 in most_close:
        print(paths2[k1], paths2[k2])
        print([n[0] for n in paths3[k1]["nodes"]], [n[0] for n in paths3[k2]["nodes"]])
        # print([index2node.get(p, None) for p in paths2[k1]], [index2node.get(p, None) for p in paths2[k2]])
        print("\n")
    # print(paths2[most_close[0]], paths2[most_close[1]])
    # print([index2node.get(p, None) for p in paths2[most_close[0]]], [index2node.get(p, None) for p in paths2[most_close[1]]])

    print(f"\n\nMost far: {most_close}")
    for k1, k2 in most_far:
        print(paths2[k1], paths2[k2])
        print([n[0] for n in paths3[k1]["nodes"]], [n[0] for n in paths3[k2]["nodes"]])
        # print([index2node.get(p, None) for p in paths2[k1]], [index2node.get(p, None) for p in paths2[k2]])
        print("\n")
    # print(f"Most far: {most_far}")
    # print([index2node.get(p, None) for p in paths2[most_far[0]]], [index2node.get(p, None) for p in paths2[most_far[1]]])

    return distances, closest, farthest, most_close, most_far

# *************************************************************


# Example usage:
if __name__ == "__main__":
    testimony_distances()


    del_cost = 1
    def distance_func(a, b):
        if a == -1 or b == -1:
            return min(del_cost, abs(a-b))
        return (a - b) ** 2
    euclidean_distance = lambda x, y: (x - y) ** 2

    seq1 = [1, 1, 2, 3, 3, 3, 4, 2, 2, 1]
    # seq2 = [1, 3, 3, 4, 2, 1, 1, 1]
    seq2 = [1, 3, 3, 7, 2, 1, 1, 1]
    # distance, alignment = dynamic_time_warping2(seq1, seq2, dist_func=euclidean_distance)

    # insert -1 every time there is a value change
    _seq1 = []
    _seq2 = []
    for i in range(len(seq1)):
        if i > 0 and seq1[i] != seq1[i-1]:
            _seq1.append(-1)
        _seq1.append(seq1[i])
    for i in range(len(seq2)):
        if i > 0 and seq2[i] != seq2[i-1]:
            _seq2.append(-1)
        _seq2.append(seq2[i])

    # distance, alignment = dynamic_time_warping2(seq1, seq2, dist_func=distance_func)
    distance, alignment = dynamic_time_warping2(_seq1, _seq2, dist_func=distance_func)
    print(f"DTW Distance: {distance}")
    print(f"Alignment: {alignment}")

    dist_func = get_dist_func_paths()

    # Example usage
    # seq1 = "kitten"
    # seq2 = "sitttting"
    seq1 = [1, 1, 2, 3, 3, 3, 4, 2, 2, 1]
    seq2 = [1, 3, 3, 4, 2, 1, 1, 1]
    # pointwise_dist_func = lambda a, b: 0 if a == b else 2  # A custom point-wise distance function
    pointwise_dist_func = lambda a, b: ((a-b) ** 2)/2  # A custom point-wise distance function

    dist, aligned_seq = edit_distance_with_alignment(seq1, seq2, pointwise_dist_func, repeat_cost=0.5)
    print(f"Edit distance between '{seq1}' and '{seq2}' is {dist}")
    print(f"Aligned sequences:")
    print(aligned_seq)