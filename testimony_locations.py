
import json
import random

import spacy
import networkx as nx
import openai
import numpy as np
from time import sleep
import sys
# from torch.cuda import graph

import argparse
from utils import parse_args
args = parse_args()
OUTPUT_TYPE = "testimonies" if not args.lake_district else "lds"

def exponential_backoff(client, messages, model='gpt-4-turbo-preview', max_retries=5, base_delay=1, id=0):
    retries = 0
    while retries < max_retries:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=1024
            )
            return completion
        except Exception as e:
            print(f"Request failed: {str(e)}")
            delay = base_delay * (2 ** retries)
            print(f"Retrying in {delay} seconds...")
            sleep(delay)
            retries += 1

    raise Exception(f"Request failed even after retries. id: {id}")


# ***********************************

def get_graphs_gpt4(i=43019, print_output=False, model="gpt-4o", save=True, revise=True):
    """

    :param i:
    :param print_output:
    :param model:
    :param save:
    :param revise:
    :return:
    """
    # from openai import OpenAI
    from openai_client import get_client
    client = get_client()
    testimony_text = make_numbered(i)
    # Set up OpenAI API credentials

    # Define the model and parameters
    # model = 'gpt-4-turbo-preview'
    prompt = """
    I'll give you a Holocaust testimony. 
    I want you to give me a JSON representing the graph of the mentioned locations (proper and common) and any known relations between them. Locations can be GPEs (like country or city) or significant facilities (like army camps , ghettos, concentration camps and death camps).  
    Some important points:
    1. Make sure the nodes contain locations only and not anything else (no nodes for events or people). 
    2. Give the nodes a type based on the type of location. The types should include: City, Country, Village, Ghetto, Army Camp, Concentration Camp, and Death Camp. 
    3. Keep the graph as full as possible,  so, for example, if a place in a city in country is mentioned, there should be nodes for the place, the city, and the country. Separate a district from a city description into two nodes.
    4. The graph should include relations between locations (i.e., A is in B).  Make sure that the direction of an edge is that of inclusion if relevant (that is, if A is in B then the edge should be from A to B).
    5. Make sure to avoid double entries.
    6. Give me the graph as JSON dictionary, with a the "nodes" field indicating a list of nodes  and "edges" indicating a list of edges. These nodes and edges should be in a format that can be create a python networkx graph. Make sure the nodes are given as a list of tuples, in which the first value is the name and the second is a dictionary with the type (as described above) The edges should be in a list of tuples, each containing two names (see example).
    
    Here is an example (from a different testimony):
    ```json
    {
    "nodes": [
        ["Forth", {"type": "Village"}],
        ["Nuremberg", {"type": "City"}],
        ["Dachau", {"type": "Camp"}],
        ["Bamberg", {"type": "City"}],
        ["Schnaittach", {"type": "City"}],
        ["WashingtonHeights", {"type": "District"}],
        ["FortDix", {"type": "Camp"}],
        ["LosAngeles", {"type": "City"}],
        ["Oakland", {"type": "City"}],
        ["CampRitchie", {"type": "Camp"}],
        ["Naples", {"type": "City"}],
        ["SouthernFrance", {"type": "Region"}],
        ["Aschaffenburg", {"type": "City"}],
        ["Frankfurt", {"type": "City"}],
        ["JewishCemetery", {"type": "Place"}],
        ["Germany", {"type": "Country"}],
        ["NewYork", {"type": "City"}],
        ["America", {"type": "Country"}],
        ["NewJersey", {"type": "State"}],
        ["California", {"type": "State"}],
        ["Italy", {"type": "Country"}],
        ["France", {"type": "Country"}]
    ], 
    "edges": [
        ["Forth", "Germany"],
        ["Nuremberg", "Germany"],
        ["Dachau", "Germany"],
        ["Bamberg", "Germany"],
        ["Schnaittach", "Germany"],
        ["WashingtonHeights", "NewYork"],
        ["NewYork", "America"],
        ["FortDix", "NewJersey"],
        ["NewJersey", "America"],
        ["LosAngeles", "America"],
        ["Oakland", "California"],
        ["California", "America"],
        ["CampRitchie", "America"],
        ["Naples", "Italy"],
        ["SouthernFrance", "France"],
        ["Aschaffenburg", "Germany"],
        ["Frankfurt", "Germany"],
        ["JewishCemetery", "Schnaittach"]
    ]
    }
    ```
    
    This should all be based on the text. 
    
    Testimony:
    """
    m3a = """
    Go over your answer and make sure that it is consistent. Check the types of the nodes and the direction of the edges. Make sure that the nodes are locations only and that there are no double entries.
    Give your (possibly) corrected answer in the same JSON format.
    """
    m3b = """
    Go over your answer and make sure that it is consistent. 
    Make sure that: (1) the sentence numbers are in ascending order; (2) a node does not repeat without other nodes between; 
    (3) there are edges between adjacent nodes; (4) a long description of a location is not repeated as a separate node (e.g., "Brooklyn, New York" should be one node and not two).
    
    Give your (possibly) corrected answer in the same JSON format.
    """

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.append({"role": "user", "content": prompt + testimony_text})
    completion = exponential_backoff(client, messages, id=i, model=model)
    if revise:
        messages.append({"role": completion.choices[0].message.role, "content": completion.choices[0].message.content})
        messages.append({"role": "user", "content": m3a})
        completion = exponential_backoff(client, messages, id=i, model=model)

    try:
        d = json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"JSON parsing failed: {str(e)}")
        return {"nodes": [], "edges": []}, {"nodes": [], "edges": []}

    # get path
    m2 = """
    Now, can you give a graph with the trajectory of the witness' movements? That is, give a list of location where he is. 
    All location nodes should be nodes from the networkx graph you gave before. The nodes should have a field noting the sentence number in the text in which the witness was in that location.
    The edges should be between each adjacent node by order of the testimony.  Make sure that the sentence number is sorted in ascending order.
    For each edge, add the method of transportation can be inferred from the text. Methods include: By foot, By car, By train, By plane. If the method is unknown give Unknown.
    Give me a graph in JSON format (like in the example).
    
    For example:
    ```json
    {
      "nodes": [
        ["Forth", {"sentence": 1}],
        ["Nuremberg", {"sentence": 2}],
    ],
      "edges": [
        ["Forth", "Nuremberg", {"type": "By foot"}],
    ]
    }
    ```
    """
    messages.append({"role": completion.choices[0].message.role, "content": completion.choices[0].message.content})
    messages.append({"role": "user", "content": m2})
    completion = exponential_backoff(client, messages, id=i, model=model)

    if revise:
        messages.append({"role": completion.choices[0].message.role, "content": completion.choices[0].message.content})
        messages.append({"role": "user", "content": m3b})
        completion = exponential_backoff(client, messages, id=i, model=model)

    try:
        d2 = json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"JSON parsing failed: {str(e)}")
        return d, {"nodes": [], "edges": []}
    # d2 = json.loads(completion.choices[0].message.content)
    messages.append({"role": completion.choices[0].message.role, "content": completion.choices[0].message.content})
    if print_output:
        print(messages)

    if save:
        with open(args.base_path + f"{OUTPUT_TYPE}/singles/graph_{i}_{model}.json", 'w') as file:
            json.dump(d, file)
        with open(args.base_path + f"{OUTPUT_TYPE}/singles/path_{i}_{model}.json", 'w') as file:
            json.dump(d2, file)
        # update created_ids
        with open(args.base_path + f"created_ids{'_e' if args.evaluate else ''}_{model}.json", 'r') as file:
            created_ids = json.load(file)
        created_ids.append(i)
        with open(args.base_path + f"created_ids{'_e' if args.evaluate else ''}_{model}.json", 'w') as file:
            json.dump(created_ids, file)
    return d, d2


def get_doubles_gpt4(locations, print_output=False, i=43019, model="gpt-4o"):
    """

    :param locations:
    :param print_output:
    :param i:
    :param model:
    :return:
    """
    from openai_client import get_client
    client = get_client()

    prompt = """
    I'll give you (in JSON format) a list of place names. I want you to see if there are any places that appear twice but with different names. 
    Give me a JSON with a list of lists, where the inner list is the multiple names that describe the same place (and both appear in the input). No need to return unique names (i.e., lists with one element). 
    Convert names only if you are positive that they are the same, e.g., different spellings or a longer description of the same place (like US, USA, America etc.). 
    Make sure to maintain the exact spelling that appeared, including special characters.
    Make sure to give only the JSON format with no additional text.
    
    For example, if the input is:
    ```json
    ["United States of America", "USA", "Lodz", "Lódz"]
    ```
    Then the output should be:
    ```json
    [["United States of America", "USA"], ["Lodz", "Lódz"]]
    ```
    
    Here is the input:
    """
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    input = f"""
    ```json
    {json.dumps(locations)}
    ```
    """
    messages.append({"role": "user", "content": prompt + input})
    completion = exponential_backoff(client, messages, id=i, model=model)
    try:
        l = json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"JSON parsing failed: {str(e)}")
        return {"nodes": [], "edges": []}, {"nodes": [], "edges": []}

    if print_output:
        print(messages)
    return l


def get_ld_graphs_gpt4(i="", print_output=False, model="gpt-4o-mini", save=True, revise=True):
    """

    :param i:
    :param print_output:
    :param model:
    :param save:
    :param revise:
    :return:
    """
    # from openai import OpenAI
    from openai_client import get_client
    # Set up OpenAI API credentials
    client = get_client()
    text = get_travel(i)

    # Define the model and parameters
    prompt = """
    I'll give you a travel description in the Lake District. 
    I want you to give me a JSON representing the graph of the mentioned locations (proper and common) and any known relations between them. Locations can be GPEs (like country or city) or important buildings or natural locations.  
    Some important points:
    1. Make sure the nodes contain locations only and not anything else (no nodes for events or people). 
    2. Give the nodes a type based on the type of location. The types should include: Country, City, Facility and Natural. 
    3. Keep the graph as full as possible, so, for example, if a place in a city in country is mentioned, there should be nodes for the place, the city, and the country.
    4. The graph should include relations between locations (i.e., A is in B).  Make sure that the direction of an edge is that of inclusion if relevant (that is, if A is in B then the edge should be from A to B).
    5. Make sure to avoid double entries.
    6. Give me the graph as JSON dictionary, with a the "nodes" field indicating a list of nodes  and "edges" indicating a list of edges. These nodes and edges should be in a format that can be create a python networkx graph. Make sure the nodes are given as a list of tuples, in which the first value is the name and the second is a dictionary with the type (as described above) The edges should be in a list of tuples, each containing two names (see example).
    
    Here is an example:
    ```json
    {
    "nodes": [
        ["lake", {"type": "Natural"}],
        ["Keswick", {"type": "City"}]
    ], 
    "edges": [
        ["lake", "Keswick"]
    ]
    }
    ```
    
    This should all be based on the text. 
    
    Description:
    """

    m3a = """
    Go over your answer and make sure that it is consistent. Check the types of the nodes and the direction of the edges. Make sure that the nodes are locations only and that there are no double entries.
    Give your (possibly) corrected answer in the same JSON format.
    """
    m3b = """
    Go over your answer and make sure that it is consistent. 
    Make sure that: (1) the sentence numbers are in ascending order; (2) a node does not repeat without other nodes between; 
    (3) there are edges between adjacent nodes; (4) a long description of a location is not repeated as a separate node (e.g., "Brooklyn, New York" should be one node and not two).
    
    Give your (possibly) corrected answer in the same JSON format.
    """

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.append({"role": "user", "content": prompt + text})
    completion = exponential_backoff(client, messages, id=i, model=model)
    if revise:
        messages.append({"role": completion.choices[0].message.role, "content": completion.choices[0].message.content})
        messages.append({"role": "user", "content": m3a})
        completion = exponential_backoff(client, messages, id=i, model=model)

    try:
        d = json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"JSON parsing failed: {str(e)}")
        return {"nodes": [], "edges": []}, {"nodes": [], "edges": []}

    # get path
    m2 = """
    Now, can you give a graph with a trajectory of the travel? That is, give a list of location where the traveler will be. 
    All location nodes should be nodes from the networkx graph you gave before. 
    The edges should be between each adjacent node by order of the testimony.
    Give me a graph in JSON format (like in the example).
    
    For example:
    ```json
    {
    "nodes": [
        ["Hagley"],
        ["Dovedale"]
    ], 
    "edges": [
        ["Hagley", "Dovedale"]
    ]
    }
    ```
    """
    messages.append({"role": completion.choices[0].message.role, "content": completion.choices[0].message.content})
    messages.append({"role": "user", "content": m2})
    completion = exponential_backoff(client, messages, id=i, model=model)

    if revise:
        messages.append({"role": completion.choices[0].message.role, "content": completion.choices[0].message.content})
        messages.append({"role": "user", "content": m3b})
        completion = exponential_backoff(client, messages, id=i, model=model)

    try:
        d2 = json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"JSON parsing failed: {str(e)}")
        return d, {"nodes": [], "edges": []}
    # d2 = json.loads(completion.choices[0].message.content)
    messages.append({"role": completion.choices[0].message.role, "content": completion.choices[0].message.content})
    if print_output:
        print(messages)

    if save:
        with open(args.base_path + f"{OUTPUT_TYPE}/singles/graph_{i}_{model}.json", 'w') as file:
            json.dump(d, file)
        with open(args.base_path + f"{OUTPUT_TYPE}/singles/path_{i}_{model}.json", 'w') as file:
            json.dump(d2, file)
        # update created_ids
        with open(args.base_path + f"lds/created_ids{'_e' if args.evaluate else ''}_{model}.json", 'r') as file:
            created_ids = json.load(file)
        created_ids.append(i)
        with open(args.base_path + f"lds/created_ids{'_e' if args.evaluate else ''}_{model}.json", 'w') as file:
            json.dump(created_ids, file)
    return d, d2


def get_doubles_gpt4(locations, print_output=False, i=43019, model="gpt-4o"):
    """

    :param locations:
    :param print_output:
    :param i:
    :param model:
    :return:
    """
    from openai_client import get_client
    client = get_client()

    prompt = """
    I'll give you (in JSON format) a list of place names. I want you to see if there are any places that appear twice but with different names. 
    Give me a JSON with a list of lists, where the inner list is the multiple names that describe the same place (and both appear in the input). No need to return unique names (i.e., lists with one element). 
    Convert names only if you are positive that they are the same, e.g., different spellings or a longer description of the same place (like US, USA, America etc.). 
    Make sure to maintain the exact spelling that appeared, including special characters.
    Make sure to give only the JSON format with no additional text.
    
    For example, if the input is:
    ```json
    ["United States of America", "USA", "Lodz", "Lódz"]
    ```
    Then the output should be:
    ```json
    [["United States of America", "USA"], ["Lodz", "Lódz"]]
    ```
    
    Here is the input:
    """
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    input = f"""
    ```json
    {json.dumps(locations)}
    ```
    """
    messages.append({"role": "user", "content": prompt + input})
    completion = exponential_backoff(client, messages, id=i, model=model)
    try:
        l = json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"JSON parsing failed: {str(e)}")
        return {"nodes": [], "edges": []}, {"nodes": [], "edges": []}

    if print_output:
        print(messages)
    return l


# ***********************************

def get_travel(i):
    """
    Get the travel path of the witness
    :param i:
    :return:
    """
    with open(args.base_path + f"data/lake_district.json", 'r') as file:
        d = json.load(file)
    return d[str(i)]

def make_numbered(i=43019, return_n=False):
    """
    Add numbers to the sentences in the testimony
    :param i:
    :return:
    """
    data_path = args.base_path + "data/"
    with open(data_path + 'sf_raw_text.json', 'r') as infile:
        texts = json.load(infile)

    nlp = spacy.load("en_core_web_md")
    doc = nlp(texts[str(i)])

    n_sents = []
    p = ""
    for s in doc.sents:
        if s.text[-1] == ":" or len(s.text + p) < 50:
            p = p + " " + s.text
        else:
            n_sents.append(p + s.text)
            p = ""

    numbered = "\n".join([f"({str(i)}) {s.strip()}" for i, s in enumerate(n_sents)])
    if return_n:
        return n_sents
    return numbered

# ***********************************

def plot_path(G, G_paths=None, i=""):
    """
    Plot the graph and a path on it
    :param G:
    :param G_path:
    :param i:
    :return:
    """
    import matplotlib.pyplot as plt
    # from matplotlib.colors import Colormap

    # Plot the graph
    # plt.figure(figsize=(100, 100))
    # plt.figure(figsize=(35, 35))

    if G_paths is None:
    # if True:
        G_paths = {i: None}
        path_nodes = []
    else:
        path_nodes = [n for Gp in G_paths.values() for n in Gp.nodes]

    for G_path in [G_paths[i]]:
        plt.figure(figsize=(60, 60))
        # pos = nx.spring_layout(G, k=0.1, seed=42)

        # pos = nx.nx_pydot.graphviz_layout(G, prog="twopi")
        types = list(nx.get_node_attributes(G, "type").values())
        size2type = {2500: ['Continent'],
                     1500: ["Country"],
                     200: ["County", "State", 'Region', 'District'],
                     50: ['Suburb', 'Spa Town', 'Village', 'Province', 'Town', 'City', 'Borough', 'Moshav', 'Kibbutz'],
                     15: ["ArmyCamp", "Army Camp", 'DP Camp', 'Place', 'Location', 'Language', 'Significant Location', 'Common', 'Military Base', 'Internment Camp',
                          'ConcentrationCamp', 'Concentration Camp', 'Displaced Persons Camp', 'Work Camp', 'Refugee Camp', 'Labor Camp', 'Sub-Camp', 'Transitional Camp',
                          'Ghetto', 'Death Camp', 'DeathCamp', 'Camp', 'Mass Grave Site', 'Forest', 'Island'],
                     5: ["School", 'Institution', 'Lake', 'River', 'Company', 'University', 'Street', 'Legislation',
                         'Facility', 'Ship', 'Mountain Range', 'SignificantFacility', 'Significant Facility', 'HistoricalPlace',
                         'Building', 'Organization']
                     }
        holocaust_types = ['DP Camp', 'ConcentrationCamp', 'Concentration Camp', 'Ghetto', 'Death Camp', 'DeathCamp', 'Camp', 'Mass Grave Site']
        node_labels = {n: (n if t in size2type[2500] + size2type[1500] + size2type[200] + size2type[50] + size2type[15] + holocaust_types else "") for n, t in zip(G.nodes, types)}
        type2size = {t: (s * 20 if t in holocaust_types else s * 2 if t in size2type[1500] else s) for s, ts in size2type.items() for t in ts}
        type2color = {t: "green" if t in holocaust_types else "brown" if t in ["Country", 'Continent'] else "blue" for t in types}

        random.seed(42)
        _G = G.copy()
        _G.remove_nodes_from([n for n, t in zip(_G.nodes, types) if t in size2type[5] and n not in path_nodes])

        # remove 60 percent the nodes of size 50 and 15
        # set numpy random seed
        np.random.seed(42)
        r = np.random.rand(len(_G.nodes))
        _G.remove_nodes_from([n for _i, (n, t) in enumerate(zip(_G.nodes, types)) if t in size2type[50] + size2type[15] and r[_i] < 0.6 and n not in path_nodes])

        # remove all isolated nodes
        _G.remove_nodes_from([n for n in nx.isolates(_G) if n not in path_nodes])

        types = list(nx.get_node_attributes(_G, "type").values())
        H = nx.convert_node_labels_to_integers(_G, label_attribute="node_label")
        H_layout = nx.nx_agraph.pygraphviz_layout(H, prog="twopi", args='-Granksep=2 -Gnormalize=0 -Gstart=42')
        pos = {H.nodes[n]["node_label"]: p for n, p in H_layout.items()}

        shape_list = ["o" if t in size2type[2500] + size2type[1500] else "s" if t in size2type[200] + size2type[50] else "^" if t in holocaust_types else "d" for t in types]

        # size_list = [type2size.get(t, 5) * 15 for t in types]
        size_list = [type2size.get(t, 5) * 15 for t in types]
        # make nodes with a small outdegree smaller
        size_list = [size_list[i] * 0.4 if len(list(_G.neighbors(n))) < 10 and shape_list[i] == 'o' else size_list[i] * 2.5 if len(list(_G.neighbors(n))) > 100 and shape_list[i] == 'o' else size_list[i] for i, n in enumerate(_G.nodes)]

        fontsize_list = [type2size.get(t, 5) // 70 + 4 for t in types]
        # adjust size but degree
        fontsize_list = [fontsize_list[i] - 4 if len(list(_G.neighbors(n))) < 10 and shape_list[i] == 'o' else fontsize_list[i] + 4 if len(list(_G.neighbors(n))) > 100 and shape_list[i] == 'o' else fontsize_list[i] for i, n in enumerate(_G.nodes)]

        # fontsize_list = [type2size.get(t, 5) // 5 + 5 for t in types]
        color_list = [type2color.get(t, "blue") for t in types]
        # make a list of shapes. we want a circle for countries and continents and squares for the rest besided holocaust types which are triangles

        nx.set_node_attributes(_G, color_list, "color")
        nx.set_node_attributes(_G, size_list, "size")
        for _i, (n, (x, y)) in enumerate(pos.items()):
            nx.draw_networkx_nodes(_G, pos, nodelist=[n], node_size=size_list[_i], node_color=color_list[_i], alpha=0.6, node_shape=shape_list[_i])

            # for _i, (node, (x, y)) in enumerate(pos.items()):
            plt.text(x, y, n, fontsize=fontsize_list[_i], ha='center', va='center')

        nx.draw_networkx_edges(_G, pos, edge_color="gray", width=2, arrows=True, alpha=0.6)

        if G_path is not None:
            print(i, "path:", G_path.edges)
            nx.set_edge_attributes(_G, range(1, len(_G.edges)+1), "n")
            path_edges = G_path.edges

            # Highlight the path
            # Extract nodes from the path_edges
            path_nodes = set([node for edge in path_edges for node in edge])
            print(path_nodes)
            # Draw the path edges in red
            # Create a colormap of reds
            cmap = plt.cm.get_cmap('Reds')
            # Normalize the colors based on the number of nodes
            import matplotlib.colors as mcolors
            norm = mcolors.Normalize(vmin=0, vmax=len(G_path.nodes))

            sizes = np.full(len(G_path.edges), 8) + np.arange(len(G_path.edges))
            # Draw the edges with the specified colors or sizes
            for _i, (edge, size) in enumerate(zip(path_edges, sizes)):
                nx.draw_networkx_edges(_G, pos, edgelist=[edge], edge_color=cmap(norm(_i)), width=size, arrows=True, arrowsize=150)

            e_labels = {e: i+1 for i, e in enumerate(path_edges)}
            nx.draw_networkx_edge_labels(_G, pos, edge_labels=e_labels)
            # Draw the path nodes in red
            nx.draw_networkx_nodes(_G, pos, nodelist=path_nodes, node_color="r", node_size=700, alpha=0.9)

        plt.savefig(args.base_path + f"{OUTPUT_TYPE}/plot{'_e' if args.evaluate else ''}{args.model}-{i}.png", dpi=150)
    # plt.show()
    print("Done")


# ***********************************
# for evaluation

def get_gold_xlsx():
    """
    :param data_path:
    :return:
    """
    import pandas as pd
    sheets = pd.read_excel(args.base_path + "data/gold_loc_xlsx/test_set-derived_from_testimonies1_full - 13.4.xlsx", sheet_name=None, usecols="B:D")
    d = {}

    for t, df in sheets.items():
        df.dropna(inplace=True)
        locs = df['location'].str.rstrip().to_list()
        d[t] = [df['text'].to_list(), locs]
    return d

def modified_edit_distance(s1, s2, modified=True):
    m = len(s1)
    n = len(s2)
    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]

    # Initialize the DP table
    for i in range(m+1):
        dp[i][0] = i if not modified else 0
    for j in range(n+1):
        dp[0][j] = j

    # Fill the DP table
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                if modified:
                    # No penalty for deletion from s1
                    dp[i][j] = min(dp[i-1][j], 1 + dp[i][j-1], 1 + dp[i-1][j-1])
                else:
                    dp[i][j] = min(1 + dp[i-1][j], 1 + dp[i][j-1], 1 + dp[i-1][j-1])

    return dp[m][n]


def align_locations(path, gold_path, model="gpt-4o"):
    """

    :param path:
    :param gold_path:
    :return:
    """
    from openai_client import get_client
    client = get_client()

    m1 = """
    I have a list of predicted locations and a list of locations from the gold standard.
    For each location in the predicted list, I want you to find a corresponding location in the gold standard list if it exists (even if it's written differently). 
    In the case it exists, give me the id of the corresponding location in the gold standard list. If it doesn't exist, give me -1.
    
    Here is an example:
    For predicted locations: ["Warsaw (Ghetto)", "Luck", "Warsaw", "New York"],
    and gold-standard locations: ["Lutsk", "The Warsaw ghetto"]]
    
    The output should be the JSON: 
    {"ids": [1, 0, -1, -1]}
    
    Make sure to follow the instructions and give the output in the correct format.
    
    """

    m1a = f"""
    Predicted locations: {path},
    Gold-standard locations: {gold_path}
    """

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.append({"role": "user", "content": m1 + m1a})
    completion = exponential_backoff(client, messages, model=model)
    try:
        d = json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"JSON parsing failed: {str(e)}")
        return {"ids": []}
    return d["ids"]

def evaluate_gold(path, gold_path, d):
    """
    Compare the path to the gold path
    :param path:
    :param gold_path:
    :return:
    """
    # get types of locations
    loc_to_type = {n[0]: n[1]["type"] for _d in d.values() for n in _d["graph"]["nodes"]}

    # add type to each location in the path
    _path = [p + ", " + loc_to_type.get(p, "") for p in path]
    _path = align_locations(_path, gold_path)

    # use the modified edit distance to compare the paths
    return modified_edit_distance([i for i, p in enumerate(gold_path)], _path), modified_edit_distance([i for i, p in enumerate(gold_path)], _path, modified=False)


def evaluate(path_d, gold_path, d, i):
    print("Evaluation:")
    eval = evaluate_gold([p[0] for p in path_d["nodes"]], gold_path, d)

    # save evaluation
    e = {}
    with open(args.base_path + f"testimonies/eval_{args.model}.json", 'r') as file:
        e = json.load(file)
    e[i] = eval
    with open(args.base_path + f"testimonies/eval_{args.model}.json", 'w') as file:
        json.dump(e, file)

    print(eval)

def evaluate_models():
    """
    Evaluate the models
    :return:
    """
    gold_d = get_gold_xlsx()

    es = {}
    for model in ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"]:
        with open(args.base_path + f"testimonies/eval_{model}.json", 'r') as file:
            e = json.load(file)
        es[model] = e
        es[model]["total"] = [0, 0]

    # use only the ids that are in all models
    ids = list(es["gpt-4o"].keys())
    es["gpt-4o-mini"] = {k: v for k, v in es["gpt-4o-mini"].items() if k in ids}
    es["max"] = {"total": [0, 0]}
    es["random"] = {"total": [0, 0]}

    for i in ids[:-1]:
        print(i)
        gold_path = gold_d[i][1]
        gold_path = [gold_path[i] for i in range(1, len(gold_path) - 1) if
                     gold_path[i] != gold_path[i - 1]]  # and remove start and end

        for model in ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"]:
            es[model][i] = [es[model][i][0] / len(gold_path), es[model][i][1] / len(gold_path)]
            es[model]["total"] = [es[model]["total"][0] + es[model][i][0] / len(ids), es[model]["total"][1] + es[model][i][1]/len(ids)]

        # convert gold path to indices
        locs = list(set(gold_path))
        _gold_path = [locs.index(l) for l in gold_path]
        # find most common index in gold_path
        _max_path = [max(set(_gold_path), key=_gold_path.count) for _ in _gold_path]
        _rand_path = np.random.randint(0, 2533, len(_gold_path))

        es["max"][i] = [modified_edit_distance(_max_path, _gold_path) / len(gold_path), modified_edit_distance(_max_path, _gold_path, modified=False) / len(gold_path)]
        es["max"]["total"] = [es["max"]["total"][0] + es["max"][i][0] / len(ids), es["max"]["total"][1] + es["max"][i][1] / len(ids)]

        es["random"][i] = [modified_edit_distance(_rand_path, _gold_path) / len(gold_path), modified_edit_distance(_rand_path, _gold_path, modified=False) / len(gold_path)]
        es["random"]["total"] = [es["random"]["total"][0] + es["random"][i][0] / len(ids), es["random"]["total"][1] + es["random"][i][1] / len(ids)]

    print(es)

# ***********************************

def combine_singles(model="gpt-4o-mini"):
    with open(args.base_path + f"created_ids{'_e' if args.evaluate else ''}_{model}.json", "r") as file:
        created_ids = json.load(file)

    d = {}
    for i in created_ids:
        d[i] = {}
        # load graph
        with open(args.base_path + f"{OUTPUT_TYPE}/singles/graph_{i}_{model}.json", "r") as file:
            d[i]["graph"] = json.load(file)
            # if "Danube" in d[i]["graph"]:
            #     print("here")

        # load path
        with open(args.base_path + f"{OUTPUT_TYPE}/singles/path_{i}_{model}.json", "r") as file:
            d[i]["path"] = json.load(file)

    with open(args.base_path + f"{OUTPUT_TYPE}/graphs{'_e' if args.evaluate else ''}_{model}.json", "w") as file:
        json.dump(d, file)
    return d

def get_graphs(testimony_ids, model="gpt-4o-mini", load=True):
    """
    :param testimony_ids:
    """
    if load:
        with open(args.base_path + f"{OUTPUT_TYPE}/graphs{'_e' if args.evaluate else ''}_{model}.json", "r") as file:
            d = json.load(file)
        return d

    with open(args.base_path + f"{OUTPUT_TYPE}/created_ids{'_e' if args.evaluate else ''}_{model}.json", "r") as file:
        created_ids = json.load(file)

    test_ids = []
    ignore = created_ids
    if not args.lake_district:
        test_d = get_gold_xlsx()
        test_ids = list(test_d.keys())
        ignore = ['45064', '29550'] + ignore
    nontest_ids = [t for t in testimony_ids if t not in test_ids]

    d = {}
    s = slice(0, args.n if args.n >= 0 else 10)
    print("Slice:")
    print(s)
    if args.set == "test":
        ids = [t for t in test_ids[s] if t not in ignore]
    else:
        ids = [t for t in nontest_ids[s] if t not in ignore]
    # ids = [t for t in testimony_ids[s] if t not in ignore]
    # ids = ['31487']
    for i in ids:
        print(i)
        if args.lake_district:
            graph_d, path_d = get_ld_graphs_gpt4(i=i, print_output=True, model=model)
        else:
            graph_d, path_d = get_graphs_gpt4(i=i, print_output=True, model=model)
        d[i] = {"graph": graph_d, "path": path_d}

    print("Ids:")
    print(ids)
    print("Graphs:")
    print(d)
    return d


def get_conversion_d(d, load=False, model="gpt-4o", save=True):
    """

    :param d:
    :param load:
    :return:
    """
    if load:
        with open(args.base_path + f"{OUTPUT_TYPE}/duplicates{'_e' if args.evaluate else ''}_{args.model}.json", "r") as file:
            conversion_d = json.load(file)
        return conversion_d

    all_locs = list(set([n[0] for _d in d.values() for n in _d["graph"]["nodes"]]))
    all_locs.sort()
    print("All locs:")
    print(all_locs)
    conversion_lists = get_doubles_gpt4(all_locs, print_output=True, model=model)
    print(conversion_lists)
    conversion_d = {}
    if len(conversion_lists) > 0:
        for l in list(conversion_lists.values())[0]:
            for _l in l[1:]:
                conversion_d[_l] = l[0]

    if save:
        with open(args.base_path + f"{OUTPUT_TYPE}/duplicates{'_e' if args.evaluate else ''}_{model}.json", "w") as file:
            json.dump(conversion_d, file)
    return conversion_d

def get_joint_graph(d, conversion_d):
    """

    """
    loc_to_type = {n[0]: n[1]["type"] for _d in d.values() for n in _d["graph"]["nodes"]}
    print("\nlocations:")
    l = list(loc_to_type.keys())
    l.sort()
    print(l)

    def impossible_edge(n1, n2):
        # check that n1 and n2 are both strings
        if not isinstance(n1, str) or not isinstance(n2, str):
            return True
        type1 = loc_to_type.get(n1, "")
        type2 = loc_to_type.get(n2, "")
        if type1 == type2:
            return True
        if type1 == "City" and type2 == "Village":
            return True
        if type1 == "Country" and type2 == "City":
            return True
        if type1 == "Continent" and type2 == "Country":
            return True
        return False

    new_graphs = {i: {"nodes": [n if n[0] not in conversion_d else [conversion_d[n[0]], {"type": loc_to_type[n[0]]}]
                                for n in _d["graph"]["nodes"]],
                      "edges": [[conversion_d.get(e[0], e[0]), conversion_d.get(e[1], e[1])]
                                for e in _d["graph"]["edges"] if (len(e) == 2 and not impossible_edge(e[0], e[1]))]} for i, _d in d.items()}
    print("new graphs:")
    print(new_graphs)
    all_nodes = set([(n[0], n[1]["type"]) for g in new_graphs.values() for n in g["nodes"]])
    # all_nodes = set([(n[0], n[1]["type"]) for g in d.values() for n in g["graph"]["nodes"]])
    node_set = set([n[0] for g in new_graphs.values() for n in g["nodes"]])
    # node_set = set([n[0] for g in d.values() for n in g["graph"]["nodes"]])
    all_nodes = [(n[0], {"type": n[1]}) for n in all_nodes]
    all_edges = set([tuple(e) for g in new_graphs.values() for e in g["edges"]])
    # all_edges = set([tuple(e) for g in d.values() for e in g["graph"]["edges"]])
    all_edges = [(e[0], e[1]) for e in all_edges if len(set(e) - node_set) == 0]
    # all_edges = list(all_edges)
    print(all_nodes)
    print(all_edges)

    G = nx.DiGraph()
    G.add_nodes_from(all_nodes)
    G.add_edges_from(all_edges)

    # save graph to file
    with open(args.base_path + f"{OUTPUT_TYPE}/nodes{'_e' if args.evaluate else ''}{args.model}.json", 'w') as file:
        json.dump(all_nodes, file)
    nx.write_adjlist(G, args.base_path + f"{OUTPUT_TYPE}/graph_{'_e' if args.evaluate else ''}{args.model}.adjlist", delimiter='*')
    G = nx.read_adjlist(args.base_path + f"{OUTPUT_TYPE}/graph_{'_e' if args.evaluate else ''}{args.model}.adjlist", delimiter='*')
    # add node_types to G
    nx.set_node_attributes(G, {n[0]: n[1]["type"] for n in all_nodes}, "type")

    return G

def plot_paths(d, G, testimony_ids, conversion_d):
    """

    :param testimony_ids:
    :param d:
    :param conversion_d:
    :param G:
    :return:
    """
    G_paths = {}
    ids = []
    for id in testimony_ids[:20]:
        # if id == "32783":
        #     continue
        if id in d:
            G_path = nx.DiGraph()
            path_d = d[id]["path"]
            # G_path.add_nodes_from([n for n in path_d["nodes"]])
            # G_path.add_edges_from(path_d["edges"])
            edge_list = [[conversion_d.get(e[0], e[0]), conversion_d.get(e[1], e[1])] for e in path_d["edges"]]
            node_set = set([n for n in G.nodes])
            # test edge_list
            connections = [0 if (e[1] == edge_list[i+1][0] and e[0] in node_set) else 1 for i, e in enumerate(edge_list[:-1])]
            if sum(connections) > 0:
                print("path is not continuous or some locations are not in the list")
                continue

            G_path.add_edges_from(edge_list)
            G_paths[id] = G_path
            ids.append(id)

    for id in ids:
        plot_path(G, G_paths=G_paths, i=id)


# **********************************

def find_triples(d):
    """

    :param d:
    :return:
    """
    triples = {}
    for i, _d in d.items():
        path_d = _d["path"]
        path = path_d["nodes"]
        for j, p in enumerate(path[:-2]):
            if (p[0], path[j+1][0], path[j+2][0]) in triples:
                triples[(p[0], path[j+1][0], path[j+2][0])].append(i)
            else:
                triples[(p[0], path[j+1][0], path[j+2][0])] = [i]
    return triples

def find_pairs(d):
    """

    :param d:
    :return:
    """
    pairs = {}
    for i, _d in d.items():
        path_d = _d["path"]
        path = path_d["nodes"]
        for j, p in enumerate(path[:-1]):
            if (p[0], path[j+1][0]) in pairs:
                pairs[(p[0], path[j+1][0])].append([i, (0 if j == 0 else path[j-1][1]["sentence"], p[1]["sentence"], path[j+1][1]["sentence"], -1 if j == len(path)-2 else path[j+2][1]["sentence"])])
            else:
                pairs[(p[0], path[j+1][0])] = [[i, (0 if j == 0 else path[j-1][1]["sentence"], p[1]["sentence"], path[j+1][1]["sentence"], -1 if j == len(path)-2 else path[j+2][1]["sentence"])]]
    return pairs


# ***********************************

def main():
    # combine_singles(model=args.model)
    print(sys.argv)

    model = args.model

    # evaluate_models()

    data_path = args.base_path + "data/"
    name = "sf_raw_text.json" if not args.lake_district else "lake_district.json"
    with open(data_path + name, 'r') as infile:
        testimony_ids = list(json.load(infile).keys())

    if args.evaluate or args.n >= 0:
        d = get_graphs(testimony_ids, model=model, load=False)
        # d = get_graphs(testimony_ids, model=model, load=True)
        # conversion_d = get_conversion_d(d, load=False, model=model, save=True)
        # G = get_joint_graph(d, conversion_d)

        if args.evaluate:
            gold_d = get_gold_xlsx()
            for i, _d in d.items():
                path_d = _d["path"]
                gold_path = gold_d[i][1]
                # remove repeating locations in gold path. remove only if they are consecutive
                gold_path = [gold_path[i] for i in range(1, len(gold_path)-1) if gold_path[i] != gold_path[i-1]]  # and remove start and end

                evaluate(path_d, gold_path, d, i)
        return

    d = get_graphs(testimony_ids, model=model, load=True)
    t = find_triples(d)
    p = find_pairs(d)

    # take pairs and triples that appear more than once
    m_pairs = {k: v for k, v in p.items() if len(v) > 8}
    m_triples = {k: v for k, v in t.items() if len(v) > 1}

    # conversion_d = get_conversion_d(d, load=False, model=model, save=True)
    conversion_d = get_conversion_d(d, load=True, model=model)
    print("\nConversion dict")
    print(conversion_d)

    G = get_joint_graph(d, conversion_d)
    # path_lens = []
    # for i, _d in d.items():
    #     path_d = _d["path"]['nodes']
    #     path_lens.append(len(path_d))
    # print("Path lens: ", path_lens)
    plot_paths(d, G, testimony_ids, conversion_d)

    # the median path length in d
    path_lens = [len(_d["path"]["nodes"]) for _d in d.values()]
    print("Path lens:")
    print(path_lens)
    print("Median path length:")
    print(np.median(path_lens))

if __name__ == "__main__":
    main()
