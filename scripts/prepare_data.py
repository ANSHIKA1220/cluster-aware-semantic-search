import pickle
from utils.preprocessing import load_dataset


def main():

    documents, targets, target_names = load_dataset()

    data = {
        "documents": documents,
        "targets": targets,
        "target_names": target_names
    }

    with open("data/newsgroups_raw.pkl", "wb") as f:
        pickle.dump(data, f)

    print("Dataset saved")


if __name__ == "__main__":
    main()