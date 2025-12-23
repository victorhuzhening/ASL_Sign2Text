from data import *
from mp_model_config import *
import os
import torch
from torch.utils.data import DataLoader
import argparse


def main():
    """
    Precomputes training dataset from raw ASL videos.
    Batch_num needs to be 1 to for single-sampling trick.
    For local: use num_workers = 0
    For multiprocessing: tweak num_workers to environment
    """
    parser = argparse.ArgumentParser(
        description="Precomputes training dataset from raw ASL videos."
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Directory with raw ASL videos.",
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        required=True,
        help="Path to labels file.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Path to save precomputed dataset.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--hand_config_path",
        type=str,
        default="../data/pretrained_model/hand_landmarker.task",
    )
    parser.add_argument(
        "--pose_config_path",
        type=str,
        default="../data/pretrained_model/pose_landmarker_full.task",
    )
    args = parser.parse_args()

    hand_cfg = HandCfg(args.hand_config_path)
    options = hand_cfg.create_options()
    hand_landmarker = hand_cfg.HandLandmarker.create_from_options(options)

    pose_cfg = PoseCfg(args.pose_config_path)
    options = pose_cfg.create_options()
    pose_landmarker = pose_cfg.PoseLandmarker.create_from_options(options)

    save_dir = "precomputed_train"
    os.makedirs(save_dir, exist_ok=True)

    precompute_dataset = ASLData(
        video_dir=args.video_dir,
        hand_landmarker=hand_landmarker,
        pose_landmarker=pose_landmarker,
        labels_path=args.labels_path,
        max_frames=300
    )

    precompute_dataloader = DataLoader(
        precompute_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: b[0],
    )

    torch.save(
        {
            "vocab": precompute_dataset.vocab,
            "pad_id": precompute_dataset.pad_id,
        },
        os.path.join(save_dir, "vocab_meta.pt"),
    )

    for idx, sample in enumerate(precompute_dataloader):
        sample_to_save = {
            "features": sample["features"],
            "feature_len": int(sample["feature_len"]),
            "label_ids": sample["label_ids"],
            "label_len": int(sample["label_len"]),
            "filename": sample["filename"],
            "raw_label": sample["raw_label"],
        }

        out_path = os.path.join(save_dir, f"sample_{idx:05d}.pt")
        torch.save(sample_to_save, out_path)

        if (idx + 1) % 50 == 0:
            print(f"PROGRESS: Saved {idx+1} samples...")

if __name__ == "__main__":
    main()