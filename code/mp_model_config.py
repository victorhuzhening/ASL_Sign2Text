import os
import mediapipe as mp



class HandCfg:
    """
    MediaPipe configuration options for Hand and Pose Landmarker models

    To get model task, run:
    MediaPipeCFG = MediaPipeCfg(model_path)
    options = MediaPipeCFG.options
    HandLandmarker = MediaPipeCFG.HandLandmarker.create_from_options(options)
    """
    def __init__(self, model_path: str = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"File {model_path} does not exist")

        self.ModelPath = model_path
        self.BaseOptions = mp.tasks.BaseOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        self.HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        self.NumHands = 2

    def create_options(self):
        hand_landmarker_options = self.HandLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=self.ModelPath),
            running_mode=self.VisionRunningMode.IMAGE,
            num_hands=self.NumHands,
        )
        return hand_landmarker_options



class PoseCfg:
    """
    MediaPipe configuration options for Hand and Pose Landmarker models

    To get model task, run:
    MediaPipeCFG = MediaPipeCfg(model_path)
    options = MediaPipeCFG.options
    HandLandmarker = MediaPipeCFG.HandLandmarker.create_from_options(options)
    """
    def __init__(self, model_path: str = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"File {model_path} does not exist")

        self.ModelPath = model_path
        self.BaseOptions = mp.tasks.BaseOptions
        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker
        self.PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        self.PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
        self.VisionRunningMode = mp.tasks.vision.RunningMode

    def create_options(self):
        pose_landmarker_options = self.PoseLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=self.ModelPath),
            running_mode=self.VisionRunningMode.IMAGE,
        )
        return pose_landmarker_options