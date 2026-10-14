"""Download and unzip PolypGen v3 (Synapse syn45200214) into the data folder.

Before the first run, store your Synapse token on the HPC (never in this repo):
    /l/users/mohamed.abdalla/medai/venv/bin/synapse config
The token needs only the "view" and "download" scopes.
Run:
    /l/users/mohamed.abdalla/medai/venv/bin/python scripts/download_polypgen.py
"""
import subprocess

import synapseclient

POLYPGEN_V3 = "syn45200214"  # PolypGen2021_MultiCenterData_v3.zip
DATA_DIR = "/l/users/mohamed.abdalla/medai/data/polypgen"


def main():
    syn = synapseclient.Synapse()
    syn.login()  # reads the token from ~/.synapseConfig
    entity = syn.get(POLYPGEN_V3, downloadLocation=DATA_DIR)
    print(f"Downloaded {entity.path}")
    # System unzip: Python zipfile failed on Lustre with "Bad address" (2026-10-14).
    unzip(entity.path, DATA_DIR)
    inner = f"{DATA_DIR}/PolypGen2021_MultiCenterData_v3/PolypGen2021_MultiCenterData_Concatenated.zip"
    unzip(inner, f"{DATA_DIR}/PolypGen2021_MultiCenterData_v3")  # holds the sequence data
    print(f"Unzipped into {DATA_DIR}")


def unzip(path, target):
    subprocess.run(["unzip", "-q", "-o", path, "-x", "__MACOSX/*", "-d", target], check=True)


if __name__ == "__main__":
    main()
