
import pandas as pd

def get_gold_xlsx2(data_path="", name=""):
    """
    Make spacy docs from annotated documents in xlsx format (with multiple sheets)
    :param data_path:
    :return:
    """
    sheets = pd.read_excel(data_path + name + ".xlsx", sheet_name=None, usecols="B:C")
    d = {}
    for t, df in sheets.items():
        df.dropna(inplace=True)
        locs = df['location']
        if len(locs) > 0:
            locs = df['location'].str.rstrip().to_list()
        d[t] = locs
        # d[t] = [df['text'].to_list(), locs]
    return d


if __name__ == '__main__':
    data_path = "/cs/labs/oabend/eitan.wagner/downloads/"
    names = ["Annotation - locations - Nicole", "Annotation - locations - Xena"]
    d = get_gold_xlsx2(data_path, name=names[0])
    d1 = get_gold_xlsx2(data_path, name=names[1])
    print("done")