import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os

import cv2
import albumentations as A

from sklearn.model_selection import train_test_split


# Define the augmentation pipeline
aug = A.Compose(
    [
        A.HorizontalFlip(p=1.0),  # Always apply horizontal flip
    ],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_id"]),
)


def make_dataframe(df):
    # Extract the cloth type from the image_name
    df["Cloth_Type"] = df["image_name"].apply(lambda x: x.split("/")[0].split("-")[0])

    # Get the count of each color for each cloth type
    color_count = df.groupby(["Cloth_Type", "Color"]).size().unstack().fillna(0)
    total = color_count.sum(axis=1)

    color_count["total"] = total
    color_count = color_count.sort_values(by="total", ascending=False)

    return color_count


# Data Augmentation
def augment_and_save(aug, df, path, type="train+val"):
    length = len(df)
    for idx in range(len(df)):
        if df["Color"].iloc[idx] in [1, 2, 8, 9]:
            image_path = os.path.join(path, type, df["image_name"].iloc[idx])
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            BBox_xmin = df["BBox_xmin"].iloc[idx]
            BBox_ymin = df["BBox_ymin"].iloc[idx]
            BBox_xmax = df["BBox_xmax"].iloc[idx]
            BBox_ymax = df["BBox_ymax"].iloc[idx]

            # Apply the augmentation
            augmented = aug(
                image=image,
                bboxes=[[BBox_xmin, BBox_ymin, BBox_xmax, BBox_ymax]],
                category_id=[df["Color"].iloc[idx]],
            )
            image = augmented["image"]
            bboxes = augmented["bboxes"]

            # Draw the bounding box on the image
            # cv2.rectangle(image, (int(bboxes[0][0]), int(bboxes[0][1])), (int(bboxes[0][2]), int(bboxes[0][3])), (255, 0, 0), 2)

            # Save the augmented image
            # Generate a new image path
            base_name = os.path.splitext(image_path)[
                0
            ]  # Get the base name without the extension
            ext = os.path.splitext(image_path)[1]  # Get the file extension
            new_image_path = (
                base_name + "_aug" + ext
            )  # Create new path with '_aug' appended before the extension
            image_name = new_image_path.split("/")[-2:]

            cv2.imwrite(new_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            df.loc[length + idx] = [
                "/".join(image_name),
                int(bboxes[0][0]),
                int(bboxes[0][1]),
                int(bboxes[0][2]),
                int(bboxes[0][3]),
                df["Color"].iloc[idx],
                image_name[0],
            ]


if __name__ == "__main__":
    # Load the data
    train_df = pd.read_csv("./Dataset/Fashion-How24_sub2_train.csv")
    val_df = pd.read_csv("./Dataset/Fashion-How24_sub2_val.csv")

    # Create a new directory to store the train and validation data
    if not os.path.exists("./Dataset/train+val"):
        os.mkdir("./Dataset/train+val")

    # Copy the train and validation data to the new directory
    os.system("cp -r ./Dataset/train/* ./Dataset/train+val/")
    os.system("cp -r ./Dataset/val/* ./Dataset/train+val/")

    # Concatenate the train and validation dataframes
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    train_val_df.to_csv("./Dataset/Fashion-How24_sub2_train+val.csv", index=False)

    # Create a new dataframe with the count of each color for each cloth type
    new_df = make_dataframe(train_val_df)

    augment_and_save(aug, train_val_df, "./Dataset/", "train+val")

    # Drop the Cloth_Type in the columns
    train_val_df = train_val_df.drop(labels="Cloth_Type", axis=1)
    train_val_df.to_csv("./Dataset/Fashion-How24_sub2_train+val_aug.csv", index=False)

    # Split the augmented data into train and validation data
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.2, stratify=train_val_df["Color"], random_state=42
    )

    train_df.to_csv("./Dataset/Fashion-How24_sub2_train_aug.csv", index=False)
    val_df.to_csv("./Dataset/Fashion-How24_sub2_val_aug.csv", index=False)
