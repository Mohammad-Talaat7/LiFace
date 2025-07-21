import json
import os

from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm


def download_huggingface_dataset(dataset_name, output_dir, split="train"):
    """
    Download and process datasets from Hugging Face
    Saves images and all attributes for each image in a JSON file.
    """
    print(f"Downloading {dataset_name} dataset...")
    if split == ["train", "test", "valid"]:
        dataset = load_dataset(dataset_name, split=split)
    else:
        dataset = concatenate_datasets(
            [  # type: ignore
                load_dataset(dataset_name, split="train"),
                load_dataset(dataset_name, split="valid"),
                load_dataset(dataset_name, split="test"),
            ]
        )

    # Create output directory
    dataset_dir = os.path.join(output_dir, dataset_name.split("/")[-1])
    images_dir = os.path.join(os.path.join(dataset_dir, split), "images")
    os.makedirs(images_dir, exist_ok=True)

    metadata = []
    identity_counts = {}

    # Process CelebA dataset
    for idx, item in enumerate(
        tqdm(dataset, desc=f"Processing {dataset_name}")
    ):
        identity = str(item["celeb_id"])
        identity_counts.setdefault(identity, 0)

        # Save image
        filename = f"{idx}_{identity}_{identity_counts[identity]}.jpg"
        item["image"].save(os.path.join(images_dir, filename))

        # Record all attributes in metadata
        item_metadata = {
            "filename": filename,
            **{
                k: v for k, v in item.items() if k != "image"
            },  # Exclude PIL image object
        }
        metadata.append(item_metadata)

        identity_counts[identity] += 1

    # Save metadata to JSON file
    metadata_path = os.path.join(
        os.path.join(dataset_dir, split), "metadata.json"
    )
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Downloaded {len(identity_counts)} identities to {dataset_dir}")
    return dataset_dir


if __name__ == "__main__":
    # Create raw data directory
    RAW_DIR = "../data/raw"
    os.makedirs(RAW_DIR, exist_ok=True)

    # Download datasets from Hugging Face
    celeba_dir = download_huggingface_dataset(
        "flwrlabs/celeba", RAW_DIR, split="All"
    )

    print("Data collection complete!")
    print(f"Raw Data: {RAW_DIR}")
