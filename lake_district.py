
import json
import os
import xml.etree.ElementTree as ET
import pandas as pd

def load_ld(path):
    # Parse the XML file and get the root element
    tree = ET.parse(path + "LD80_transcribed/")
    root = tree.getroot()

    # Extract all text within the root element
    text = ''.join(root.itertext())

    return text

def get_file_list(path):
    # path is a string
    # returns a list of strings
    return os.listdir(path)

def extract_enamex_elements(path):
    try:
        # Parse the XML file and get the root element
        tree = ET.parse(path)
        root = tree.getroot()

        # Find all elements of type <enamex>
        enamex_elements = root.findall('.//enamex')

        gis_dict = {elem.text: (elem.attrib["lat"], elem.attrib["long"]) for elem in enamex_elements if "lat" in elem.attrib and "long" in elem.attrib}
        # Extract the text or attributes from these elements
        # enamex_list = [elem.text for elem in enamex_elements]
    except:
        print("bad file:", path)
        gis_dict = {}
    return gis_dict


def get_gis(path="/cs/labs/oabend/eitan.wagner/LakeDistrictCorpus/"):
    # path is a string
    # returns a dictionary
    files = get_file_list(path + "LD80_geoparsed/")
    gis = {file[:-4]: extract_enamex_elements(os.path.join(path + "LD80_geoparsed/", file)) for file in files}
    return gis


def name_conversion(path="/cs/labs/oabend/eitan.wagner/LakeDistrictCorpus/"):
    # open csv file as dataframe
    df = pd.read_csv(path + "LD80_metadata/LD80_metadata.csv")
    # create a dictionary with the old and new names
    return dict(zip(df["Transcribed Filename"], df["Geoparsed Filename"]))

# *************************************

if __name__ == "__main__":

    path = "/cs/labs/oabend/eitan.wagner/LakeDistrictCorpus/"
    gis = get_gis(path)

    # path = "/cs/labs/oabend/eitan.wagner/LakeDistrictCorpus/LD80_transcribed/"
    # files = get_file_list(path)
    # d = {file[:-4]: load_ld(os.path.join(path, file)).strip().replace("\n\n\n\n", "\n\n") for file in files}
    # # print(d)
    # with open("/cs/labs/oabend/eitan.wagner/LakeDistrictCorpus/lake_district.json", "w") as f:
    #     json.dump(d, f)

