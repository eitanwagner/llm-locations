
import json
import os

import xml.etree.ElementTree as ET

def load_ld(path):
    # Parse the XML file and get the root element
    tree = ET.parse(path)
    root = tree.getroot()

    # Extract all text within the root element
    text = ''.join(root.itertext())

    return text

def get_file_list(path):
    # path is a string
    # returns a list of strings
    return os.listdir(path)


if __name__ == "__main__":
    path = "/cs/labs/oabend/eitan.wagner/LakeDistrictCorpus/LD80_transcribed/"
    files = get_file_list(path)

    d = {file[:-4]: load_ld(os.path.join(path, file)).strip().replace("\n\n\n\n", "\n\n") for file in files}
    # print(d)
    with open("/cs/labs/oabend/eitan.wagner/LakeDistrictCorpus/lake_district.json", "w") as f:
        json.dump(d, f)

