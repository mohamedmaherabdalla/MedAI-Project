"""Download PolypGen (and anything else in your Synapse download list) to the data folder.

Before the first run, store your Synapse token on the HPC (never in this repo):
    /l/users/mohamed.abdalla/medai/venv/bin/synapse config
Then run:
    /l/users/mohamed.abdalla/medai/venv/bin/python scripts/download_polypgen.py
"""
import synapseclient

DATA_DIR = "/l/users/mohamed.abdalla/medai/data/polypgen"


def main():
    syn = synapseclient.Synapse()
    syn.login()  # reads the token from ~/.synapseConfig
    manifest = syn.get_download_list(downloadLocation=DATA_DIR)
    print(f"Done. The list of downloaded files is in {manifest}")


if __name__ == "__main__":
    main()
