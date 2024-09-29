
import pandas as pd
from scipy.spatial.distance import cityblock

from testimony_locations import *
import networkx as nx


# ***********************************
# for evaluation

def get_gold_xlsx():
    """
    :param data_path:
    :return:
    """
    import pandas as pd
    sheets = pd.read_excel(args.base_path + "data/gold_loc_xlsx/test_set-derived_from_testimonies1_full - 13.4.xlsx",
                           sheet_name=None, usecols="B:D")
    d = {}

    for t, df in sheets.items():
        df.dropna(inplace=True)
        locs = df['location'].str.rstrip().to_list()
        d[t] = [df['text'].to_list(), locs]
    return d


def modified_edit_distance(s1, s2, modified=True):
    m = len(s1)
    n = len(s2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    # Initialize the DP table
    for i in range(m + 1):
        dp[i][0] = i if not modified else 0
    for j in range(n + 1):
        dp[0][j] = j

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                if modified:
                    # No penalty for deletion from s1
                    dp[i][j] = min(dp[i - 1][j], 1 + dp[i][j - 1], 1 + dp[i - 1][j - 1])
                else:
                    dp[i][j] = min(1 + dp[i - 1][j], 1 + dp[i][j - 1], 1 + dp[i - 1][j - 1])

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


def evaluate_gold(path, gold_path, d=None, use_type=True):
    """
    Compare the path to the gold path
    :param path:
    :param gold_path:
    :return:
    """
    if use_type:
        # get types of locations
        loc_to_type = {n[0]: n[1]["type"] for _d in d.values() for n in _d["graph"]["nodes"]}

        # add type to each location in the path
        _path = [p + ", " + loc_to_type.get(p, "") for p in path]
    else:
        _path = path
    _path = align_locations(_path, gold_path)

    # use the modified edit distance to compare the paths
    if not use_type:
        return modified_edit_distance([i for i, p in enumerate(gold_path)], _path), modified_edit_distance(_path, [i for i, p in enumerate(gold_path)]), modified_edit_distance(
            [i for i, p in enumerate(gold_path)], _path, modified=False)
    return modified_edit_distance([i for i, p in enumerate(gold_path)], _path), modified_edit_distance(
        [i for i, p in enumerate(gold_path)], _path, modified=False)


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
            es[model]["total"] = [es[model]["total"][0] + es[model][i][0] / len(ids),
                                  es[model]["total"][1] + es[model][i][1] / len(ids)]

        # convert gold path to indices
        locs = list(set(gold_path))
        _gold_path = [locs.index(l) for l in gold_path]
        # find most common index in gold_path
        _max_path = [max(set(_gold_path), key=_gold_path.count) for _ in _gold_path]
        _rand_path = np.random.randint(0, 2533, len(_gold_path))

        es["max"][i] = [modified_edit_distance(_max_path, _gold_path) / len(gold_path),
                        modified_edit_distance(_max_path, _gold_path, modified=False) / len(gold_path)]
        es["max"]["total"] = [es["max"]["total"][0] + es["max"][i][0] / len(ids),
                              es["max"]["total"][1] + es["max"][i][1] / len(ids)]

        es["random"][i] = [modified_edit_distance(_rand_path, _gold_path) / len(gold_path),
                           modified_edit_distance(_rand_path, _gold_path, modified=False) / len(gold_path)]
        es["random"]["total"] = [es["random"]["total"][0] + es["random"][i][0] / len(ids),
                                 es["random"]["total"][1] + es["random"][i][1] / len(ids)]

    print(es)


def get_gold_xlsx2(data_path="", name=""):
    """
    Make spacy docs from annotated documents in xlsx format (with multiple sheets)
    :param data_path:
    :return:
    """
    sheets = pd.read_excel(data_path + name + ".xlsx", sheet_name=None, usecols="B:C")
    d = {}
    for t, df in list(sheets.items())[:3]:
        df.dropna(inplace=True)
        locs = df['location']
        if len(locs) > 0:
            locs = df['location'].str.rstrip().to_list()
            d[t] = locs
        # d[t] = [df['text'].to_list(), locs]
    return d



def make_tree(G, random_tree=False):
    # Create a new directed graph
    DG = nx.DiGraph()

    # Add all nodes from G to DG
    for node, data in G.nodes(data=True):
        DG.add_node(node, **data)

    countries = []
    cities = []
    naturals = []
    facilities = []

    for node, data in G.nodes(data=True):
        if data["type"] == "Country":
            countries.append([node, data])
        elif data["type"] == "City":
            cities.append([node, data])
        elif data["type"] == "Natural":
            naturals.append([node, data])
        elif data["type"] == "Facility":
            facilities.append([node, data])


    if not random_tree:
        from geopy.distance import geodesic
        # make edges from each city to each country. Each edge will have weight 1
        city2country = [(city[0], country[0], 1.) for city in cities for country in countries]
        # natural2city = [(natural, city, 1.) for natural in naturals for city in cities]
        natural2city = [(natural[0], city[0], geodesic(natural[1]["coords"], city[1]["coords"]).kilometers) for natural in naturals for city in cities]
        facility2city = [(facility[0], city[0], geodesic(facility[1]["coords"], city[1]["coords"]).kilometers) for facility in facilities for city in cities]
        DG.add_weighted_edges_from(city2country)
        DG.add_weighted_edges_from(natural2city)
        DG.add_weighted_edges_from(facility2city)

        # find the minimal spanning tree
        T = nx.minimum_spanning_tree(DG.to_undirected()).to_directed()
        # remove all edges from city to natural, from city to facility and from country to city


        return T
    else:
        # choose one country for each city
        city2country = [(city[0], random.choice(countries)[0]) for city in cities]
        # choose one city for each natural
        natural2city = [(natural[0], random.choice(cities)[0]) for natural in naturals]

        DG.add_edges_from(city2country)
        DG.add_edges_from(natural2city)
        return DG


def test_gis(G):
    T = make_tree(G)
    RT = make_tree(G, random_tree=True)
    return


# ********************************

if __name__ == '__main__':
    data_path = "/cs/labs/oabend/eitan.wagner/downloads/"
    names = ["Annotation - locations - Nicole", "Annotation - locations - Xena"]
    d = get_gold_xlsx2(data_path, name=names[0])
    d1 = get_gold_xlsx2(data_path, name=names[1])
    args.base_path = "/cs/snapless/oabend/eitan.wagner/segmentation/"
    gold_d = get_gold_xlsx()
    paths = {}
    # for i in ["24529", "37179", "38081"]:
    for i in ["37179", "38081"]:
    # for i, _d in d.items():
        gold_path = gold_d[i][1]
        gold_path = [gold_path[i] for i in range(1, len(gold_path) - 1) if gold_path[i] != gold_path[i - 1]]
        paths[i] = gold_path
        print(i)
        print(evaluate_gold(d[i], d1[i], use_type=False))
        print(evaluate_gold(d[i], paths[i], use_type=False))
        print(evaluate_gold(d1[i], paths[i], use_type=False))
        print(len(d1[i]), len(d[i]), len(gold_path))
    # print("\n37179")
    # print(evaluate_gold(d["37179"], d1["37179"], use_type=False))
    # print(len(d1["37179"]), len(d["37179"]))
    # print("\n38081")
    # print(evaluate_gold(d["38081"], d1["38081"], use_type=False))
    # print(len(d1["38081"]), len(d["38081"]))
    print("done")