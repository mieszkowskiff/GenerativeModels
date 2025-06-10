import sys
from cleanfid import fid

def calc_fid(folder1, folder2):
    score = fid.compute_fid(folder1, folder2)
    print(f"FID:'{folder1}' and '{folder2}': {score:.4f}")

if __name__ == "__main__":


    folder1 = "./cats/cats/"
    folder2 = "./generated/AutoEncoder/"
    calc_fid(folder1, folder2)
