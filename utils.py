
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-bp", '--base_path', type=str, help="path to project (name included)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("-ld", "--lake_district", action="store_true")
    parser.add_argument("--set", type=str, default="test")
    parser.add_argument("--n", type=int, default=-1)
    args = parser.parse_args()
    return args
