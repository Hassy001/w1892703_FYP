import os
import shutil
import pandas as pd

CSV_PATH = "HAM10000_metadata.csv"
IMG_DIRS = ["HAM10000_images_part_1", "HAM10000_images_part_2"]
OUT_BASE = "data_mc"

VALID_LABELS = {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}


def find_image_path(image_id: str) -> str:
    for d in IMG_DIRS:
        p = os.path.join(d, f"{image_id}.jpg")
        if os.path.exists(p):
            return p
    return ""


def main():
    df = pd.read_csv(CSV_PATH)
    df = df[["image_id", "dx"]].copy()
    df = df[df["dx"].isin(VALID_LABELS)]

    for lab in VALID_LABELS:
        os.makedirs(os.path.join(OUT_BASE, lab), exist_ok=True)

    copied = 0
    missing = 0

    for _, row in df.iterrows():
        image_id = row["image_id"]
        lab = row["dx"]

        src = find_image_path(image_id)
        if not src:
            missing += 1
            continue

        dst = os.path.join(OUT_BASE, lab, os.path.basename(src))
        shutil.copy2(src, dst)
        copied += 1

    print("Done")
    print("Copied:", copied)
    print("Missing:", missing)
    print("Output:", OUT_BASE)


if __name__ == "__main__":
    main()
