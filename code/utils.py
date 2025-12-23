import cv2
import torch
import mediapipe as mp
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
import math
from collections import Counter


def iter_video_as_frames(path, frame_subsample=1):
    """
    Iterates through entire video file by frame, optionally sampling every nth frame using
    frame_subsample parameter.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("ERROR: Cannot open video file")

    try:
        frame_idx = 0
        while True:
            ok = cap.grab()
            if not ok:
                break

            # We read every nth frame
            if frame_idx % frame_subsample == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # MediaPipe requires RGB but cv2 returns BGR
                yield frame_idx, frame_rgb

            frame_idx += 1
    finally:
        cap.release()


def landmarks_to_tensor(landmarks, n):
    """
    Memory efficient helper that pre-allocates a Torch tensor with zeros
    and fills in landmark coordinates.
    """
    landmarks_tensor = torch.tensor(
        [(landmark.x, landmark.y, landmark.z) for landmark in landmarks],
        dtype=torch.float32
    )
    if landmarks_tensor.shape[0] < n:
        landmarks_tensor = torch.cat([landmarks_tensor,
                                      torch.zeros((n - landmarks_tensor.shape[0], 3),
                                                  dtype=torch.float32)],
                                     dim=0)
    return landmarks_tensor


def get_pose_coordinates(result: PoseLandmarkerResult):
    """
    Extract the 3D world landmarks from the pose landmark result.
    Empty or weird coordinates are automatically treated as zeros.

    Returns:
        pose_world_coordinates: torch.Tensor  (33, 3)
    """
    landmarks = getattr(result, "pose_world_landmarks", [])
    if not landmarks:
        return torch.zeros((33,3), dtype=torch.float32)
    coordinates = landmarks_to_tensor(landmarks[0], 33)
    return coordinates


def get_hand_coordinates(result: HandLandmarkerResult):
    """
    From a single HandLandmarkerResult, return two (21, 3) arrays:
    left_hand, right_hand, each with [x, y, z] per joint.
    If a hand is missing, it's all zeros.

    Returns:
        left_hand:  torch.Tensor  (21, 3)
        right_hand: torch.Tensor  (21, 3)
    """
    left = torch.zeros((21, 3), dtype=torch.float32)
    right = torch.zeros((21, 3), dtype=torch.float32)

    hand_landmarks = getattr(result, "hand_landmarks", [])
    handedness = getattr(result, "handedness", [])

    for idx, landmark_list in enumerate(hand_landmarks):
        landmark_array = landmarks_to_tensor(landmark_list, 21)

        hand_label = "unknown"  # initialize left or right handedness
        if idx < len(handedness) and len(handedness[idx]) > 0:
            hand_label = handedness[idx][0].category_name.lower()

        # Fill in left and right hand, if weird we fill in left instead
        if "left" in hand_label:
            left = landmark_array
        elif "right" in hand_label:
            right = landmark_array
        else:
            left = landmark_array
    return left, right


def extract_coordinate_sequences_from_video(
        video_path: str,
        hand_landmarker,
        pose_landmarker,
        frame_subsample=1,
):
    """
    Runs MediaPipe and MMPose models over all frames and returns
    hand/pose coordinate sequences. Logic taken from extract_coordinate_sequences because
    this function needs to be used for inference.
    return:
        hand_seq: dict
        pose_seq: dict
    """
    POSE_SEQ = []
    LEFT_SEQ = []
    RIGHT_SEQ = []
    IDX_SEQ = []

    for frame_idx, frame_rgb in iter_video_as_frames(video_path, frame_subsample=frame_subsample):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=frame_rgb)
        hand_landmarks = hand_landmarker.detect(mp_image)
        pose_landmarks = pose_landmarker.detect(mp_image)  # Raw landmarks from MP model: PoseLandmarkerResult object

        left_hand_coordinates, right_hand_coordinates = get_hand_coordinates(hand_landmarks)
        pose_coordinates = get_pose_coordinates(pose_landmarks)

        POSE_SEQ.append(pose_coordinates)
        LEFT_SEQ.append(left_hand_coordinates)
        RIGHT_SEQ.append(right_hand_coordinates)
        IDX_SEQ.append(frame_idx)

    return IDX_SEQ, LEFT_SEQ, RIGHT_SEQ, POSE_SEQ


def build_id_to_token(vocab: dict) -> dict:
    """
    Builds id to token mapping helper.
    Converts {token: id} → {id: token}.
    """
    return {idx: tok for tok, idx in vocab.items()}


def tokens_to_text(
        ids,
        id_to_token,
        pad_id: int,
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
):
    """
    Convert a sequence of token IDs into a space-separated string.
    Skips <pad> and <bos>, stops at first <eos>.
    """
    tokens = []
    for i in ids:
        i = int(i)

        if i == pad_id:
            continue  # ignore padding

        tok = id_to_token.get(i, "<unk>")

        if tok == bos_token:
            continue  # skip <bos>

        if tok == eos_token:
            break  # stops at first <eos>

        tokens.append(tok)

    return " ".join(tokens)


def bleu1(pred_tokens, label_tokens):
    """
    Custom BLEU score calculation between label and prediction tokens.
    Avoids extra dependencies.
    """
    if len(pred_tokens) == 0:
        return 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(label_tokens)

    overlap = sum(min(pred_counts[word], ref_counts[word]) for word in pred_counts)

    precision = overlap / len(pred_tokens)

    # brevity penalty
    label_len = len(label_tokens)
    pred_len = len(pred_tokens)

    if pred_len == 0:
        return 0.0
    if pred_len > label_len:
        bleu_score = 1.0
    else:
        bleu_score = math.exp(1.0 - label_len / pred_len)

    return bleu_score * precision


def rouge1_f1(pred_tokens, label_tokens):
    """
    Custom ROUGE score calculation between label and prediction tokens.
    Avoids extra dependencies.
    """
    if not pred_tokens or not label_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    label_counts = Counter(label_tokens)

    overlap = sum(min(pred_counts[word], label_counts[word]) for word in pred_counts)

    precision = overlap / len(pred_tokens)
    recall = overlap / len(label_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)
