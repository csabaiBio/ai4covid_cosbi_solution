import argparse
from pathlib import Path
import os
from openpyxl import load_workbook
import pandas as pd
import torch
import numpy as np
import random


def get_args():
    parser = argparse.ArgumentParser(description="Configuration File")
    parser.add_argument("-f", "--cfg_file", help="Path of Configuration File", type=str)
    parser.add_argument("--mode", default='client')
    parser.add_argument("--port", default=53667)
    return parser.parse_args()


def seed_all(seed):
    if not seed:
        seed = 0
    print("Using Seed : ", seed)

    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Empty and create direcotory
def create_dir(dir):
    if not os.path.exists(dir):
        Path(dir).mkdir(parents=True, exist_ok=True)


def delete_file(file_path):
    try:
        os.remove(file_path)
    except FileNotFoundError:
        pass


def save_results(sheet_name, file, table, index=False, header=True):
    try:
        book = load_workbook(file)
        writer = pd.ExcelWriter(file)
        writer.book = book
    except FileNotFoundError:
        writer = pd.ExcelWriter(file)
    table.to_excel(writer, sheet_name=sheet_name, index=index, header=header)
    writer.save()
    writer.close()

