from src.preprocessing_util import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ont_files", type=str, default=False, help="create ontology files")
args = parser.parse_args()

preprocess_we(coref=False)
preprocess_we(coref=True)

prep_we_eval()

if args.ont_files:
    get_roles_file(shared=True)
    get_roles_file(shared=False)
    get_feasible_roles_file()
    get_mention_types_file()