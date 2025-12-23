import os
from glob import glob
import pyarrow as pa
import pyarrow.csv as csv
from torch.utils.data import Dataset
from utils import *
from tokenizer import *
from transforms import *
from pathlib import Path


class CameraCfg:
    """
    Camera configuration to set up a livestream input using webcam.
    Function used for demo and livestream inference - currently unused :(
    Output is defined using VideoWriter.
    """

    def __init__(self, cameraIdx: int, fps: float, is_array: bool):
        self.CameraIdx = cameraIdx  # default camera (usually webcam)
        self.FPS = fps
        self.OutputPath = "output.mp4" if not is_array else "coordinates.csv"

    def create_camera(self):
        cam = cv2.VideoCapture(self.CameraIdx, cv2.CAP_DSHOW)
        if not cam.isOpened():
            raise RuntimeError("ERROR: could not open camera")
        frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # define codec
        out = cv2.VideoWriter(self.OutputPath, fourcc, self.FPS, (frame_width, frame_height))
        return cam, out, frame_width, frame_height


def load_sentence_labels(csv_path):
    """
    Helper to create labels lookup dict.
    Reads from the csv:
        ["sentence_id", "sentence"]

    Returns:
        SENTENCE_LABELS: {
            "video_id": sentence label
        }
    """
    parse_options = csv.ParseOptions(delimiter="\t")

    convert_options = csv.ConvertOptions(
        column_types={
            "SENTENCE_NAME": pa.string(),
            "SENTENCE": pa.string(),
        },
        include_columns=["SENTENCE_NAME", "SENTENCE"],
    )

    table = csv.read_csv(csv_path, parse_options=parse_options, convert_options=convert_options)

    sentence_id = table["SENTENCE_NAME"].to_pylist()
    sentence = table["SENTENCE"].to_pylist()
    return dict(zip(sentence_id, sentence))


def build_feature_tensor_from_sequences(
        left_seq,
        right_seq,
        pose_seq,
        max_frames: int | None = None,
        do_transform=False,
):
    """
    Build a [T, D] FloatTensor feature tensor from hand and pose coordinate sequences, where T is frame dimension,
    and D is number of features.
    Returns ML-friendly PyTorch FloatTensor arrays.
    This function is also used for inference.
    """
    frame_dim = min(len(left_seq), len(right_seq), len(pose_seq))
    if frame_dim == 0:
        raise RuntimeError("Video produced 0 frames while building feature tensor.")

    pos_indices = list(range(frame_dim))

    # Clip frames if max_frames (optional)
    if max_frames:
        pos_indices = pos_indices[: max_frames]

    if do_transform:
        pos_indices = temporal_jitter_and_shuffle(
            pos_indices,
            num_frames=len(pos_indices),
            max_jitter=1,
            jitter_prob=0.2,
            shuffle_prob=0.15,
        )

    left_hand = torch.stack([left_seq[i] for i in pos_indices], dim=0)
    right_hand = torch.stack([right_seq[i] for i in pos_indices], dim=0)
    pose = torch.stack([pose_seq[i] for i in pos_indices], dim=0)

    if do_transform:
        left_hand = random_affine_transforms(left_hand)
        right_hand = random_affine_transforms(right_hand)
        pose = random_affine_transforms(pose)

    return torch.cat([
        left_hand.reshape(left_hand.size(0), -1),
        right_hand.reshape(right_hand.size(0), -1),
        pose.reshape(pose.size(0), -1),
    ], dim=1).float()


class ASLData(Dataset):
    def __init__(self,
                 video_dir,
                 hand_landmarker,
                 pose_landmarker,
                 labels_path,
                 max_frames,
                 do_transform=False,
                 frame_subsample=1,
                 min_frequency=1,
                 vocab=None,
                 output_dir=None):
        super().__init__()
        self.video_dir = video_dir
        self.hand_landmarker = hand_landmarker
        self.pose_landmarker = pose_landmarker
        self.labels_path = labels_path
        self.max_frames = max_frames
        self.do_transform = do_transform
        self.vocab = vocab
        self.frame_subsample = frame_subsample
        self.output_dir = str(output_dir) if output_dir is not None else None
        self.sentence_labels = load_sentence_labels(self.labels_path)

        # vocab + tokenizer block
        if vocab is None:
            sentences = self.sentence_labels.values()
            self.vocab = build_vocab_from_sentences(sentences, min_freq=min_frequency)
        else:
            self.vocab = vocab

        self.pad_id = self.vocab["<pad>"]
        self.unk_id = self.vocab["<unk>"]

        def _tokenizer_fn(text):
            tokens = ["<bos>"] + basic_tokenize(text) + ["<eos>"]
            return [self.vocab.get(tok, self.unk_id) for tok in tokens]

        self.tokenizer_fn = _tokenizer_fn

        # Read only valid MP4 videos
        all_paths = [Path(video_dir) / p for p in os.listdir(self.video_dir)]
        all_paths = [p for p in all_paths if p.is_file()]

        label_keys = set(self.sentence_labels.keys())
        valid_videos = []
        for p in all_paths:
            if p.suffix.lower() != ".mp4":
                continue
            if p.stem not in label_keys:
                continue
            valid_videos.append(p)

        self.video_paths = sorted(valid_videos)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        """
        Returns a dict ready for model training/prediction:
        """
        video_path = self.video_paths[idx]
        file_name_ext = os.path.basename(video_path)
        file_name, _ = os.path.splitext(file_name_ext)
        label_text = self.sentence_labels[file_name]

        idx_seq, left_seq, right_seq, pose_seq = extract_coordinate_sequences_from_video(video_path,
                                                                                         self.hand_landmarker,
                                                                                         self.pose_landmarker,
                                                                                         self.frame_subsample,
                                                                                         )
        feature_tensor = build_feature_tensor_from_sequences(
            left_seq=left_seq,
            right_seq=right_seq,
            pose_seq=pose_seq,
            max_frames=self.max_frames,
            do_transform=self.do_transform,
        )

        tensor_len = len(feature_tensor)
        label_ids_list = self.tokenizer_fn(label_text)
        label_ids = torch.tensor(label_ids_list, dtype=torch.long)
        label_len = len(label_ids_list)

        return {
            "features": feature_tensor,  # [T', D]
            "feature_len": tensor_len,  # int
            "label_ids": label_ids,  # [L]
            "label_len": label_len,  # int
            "filename": str(video_path),  # str
            "raw_label": label_text,  # str
        }


def asl_collate_func(batch, pad_id):
    """
    Collate function for ASLData dataset.
    Pads sequences in time dimension, and label sequences in length, using pad_id for text.
    """
    batch_size = len(batch)

    feature_len = [b["feature_len"] for b in batch]
    label_len = [b["label_len"] for b in batch]

    max_feature_len = max(feature_len)
    max_label_len = max(label_len)

    feature_dim = batch[0]["features"].shape[1]

    feature_batch = torch.zeros(batch_size, max_feature_len, feature_dim, dtype=torch.float32)
    label_batch = torch.full(
        (batch_size, max_label_len), fill_value=pad_id, dtype=torch.long  # use long for int
    )

    feature_len_tensor = torch.tensor(feature_len, dtype=torch.long)
    label_len_tensor = torch.tensor(label_len, dtype=torch.long)

    filenames = []
    raw_labels = []

    for i, sample in enumerate(batch):
        sample_feature_len = sample["feature_len"]
        sample_label_len = sample["label_len"]

        feature_batch[i, :sample_feature_len] = sample["features"]
        label_batch[i, :sample_label_len] = sample["label_ids"]

        filenames.append(sample["filename"])
        raw_labels.append(sample["raw_label"])

    return {
        "features": feature_batch,  # [B, max_T, D]
        "feature_len": feature_len_tensor,  # [B]
        "labels": label_batch,  # [B, max_L]
        "label_len": label_len_tensor,  # [B]
        "filenames": filenames,  # str
        "raw_labels": raw_labels,  # str
    }


class PrecomputedASLData(Dataset):
    """
    Loads precomputed ASL samples from feature directory.
    A separate vocab meta file must exist in the same directory.
    """

    def __init__(self, data_dir: str):
        super().__init__()
        self.sample_paths = sorted(glob(os.path.join(data_dir, "sample_*.pt")))
        if not self.sample_paths:
            raise RuntimeError(f"No sample_*.pt files found in {data_dir}")

        vocab_meta = torch.load(os.path.join(data_dir, "vocab_meta.pt"))
        self.vocab = vocab_meta["vocab"]
        self.pad_id = vocab_meta["pad_id"]

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        sample = torch.load(self.sample_paths[idx], map_location="cpu")
        # Safety check to satisfy collate function expectations
        return {
            "features": sample["features"],
            "feature_len": sample["feature_len"],
            "label_ids": sample["label_ids"],
            "label_len": sample["label_len"],
            "filename": sample["filename"],
            "raw_label": sample["raw_label"],
        }
