import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import Counter

# Define landmark groups
LANDMARK_GROUPS = {
    'arms': [11, 12, 13, 14, 15, 16],  # shoulder to wrist landmarks
    'legs': [23, 24, 25, 26, 27, 28],  # hip to ankle landmarks
    'core': [23, 24, 11, 12],  # hips and shoulders
    'upper_body': [11, 12, 13, 14, 15, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # above hips
}

class WorkoutPredictor:
    def __init__(self, model_paths, class_mapping_path):
        # Load all models
        self.models = [tf.keras.models.load_model(model_path) for model_path in model_paths]
        self.class_mapping = np.load(class_mapping_path, allow_pickle=True).item()
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Model 2 instead of 1
            min_detection_confidence=0.5
        )
        self.max_frames = 45
        self.expected_features = 297  # Model expects 297 features per frame

    def normalize_poses(self, poses):
        normalized_poses = []
        for pose in poses:
            landmarks = pose.reshape(-1, 4)
            left_hip = landmarks[23][:3]
            right_hip = landmarks[24][:3]
            hip_center = (left_hip + right_hip) / 2
            normalized_landmarks = landmarks.copy()
            normalized_landmarks[:, :3] -= hip_center
            normalized_poses.append(normalized_landmarks.flatten())
        return np.array(normalized_poses)

    def calculate_velocity(self, poses):
        velocity = np.zeros_like(poses)
        velocity[1:] = poses[1:] - poses[:-1]
        return velocity

    def extract_poses_from_video(self, video_path):
        poses = []
        confidence_scores = []

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(0, total_frames-1, self.max_frames, dtype=int)

            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (640, 480))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(frame_rgb)

                if results.pose_landmarks:
                    confidence = np.mean([lm.visibility for lm in results.pose_landmarks.landmark])

                    if confidence > 0.4:
                        landmarks = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark])
                        poses.append(landmarks.flatten())
                        confidence_scores.append(confidence)
                    else:
                        poses.append(np.zeros(33 * 4))
                        confidence_scores.append(0.0)
                else:
                    poses.append(np.zeros(33 * 4))
                    confidence_scores.append(0.0)

            cap.release()

            if len(poses) < self.max_frames:
                last_pose = poses[-1] if poses else np.zeros(33 * 4)
                poses.extend([last_pose] * (self.max_frames - len(poses)))
                confidence_scores.extend([0.0] * (self.max_frames - len(confidence_scores)))

            poses = np.array(poses)
            normalized_poses = self.normalize_poses(poses)
            velocity_features = self.calculate_velocity(normalized_poses)

            # Concatenate normalized poses and velocity features
            final_poses = np.concatenate([normalized_poses, velocity_features], axis=1)

            # Add dummy features if needed to match the expected input size of 297 features
            if final_poses.shape[1] < self.expected_features:
                dummy_features = np.zeros((final_poses.shape[0], self.expected_features - final_poses.shape[1]))
                final_poses = np.concatenate([final_poses, dummy_features], axis=1)

            return final_poses, np.array(confidence_scores)

        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            return np.zeros((self.max_frames, 33 * 4 * 2)), np.zeros(self.max_frames)

    def preprocess_video(self, video_path):
        poses, _ = self.extract_poses_from_video(video_path)
        if poses.shape[0] < self.max_frames:
            padding = np.zeros((self.max_frames - poses.shape[0], poses.shape[1]))
            poses = np.vstack((poses, padding))
        return poses

    def predict(self, video_path):
        preprocessed_data = self.preprocess_video(video_path)
        if preprocessed_data is None:
            print("Failed to preprocess video.")
            return None

        preprocessed_data = np.expand_dims(preprocessed_data, axis=0)

        # Get predictions from all models
        model_predictions = []
        for model in self.models:
            predictions = model.predict(preprocessed_data)
            predicted_class_idx = np.argmax(predictions, axis=1)[0]
            model_predictions.append(predicted_class_idx)

        # Majority voting for final class prediction
        most_common_class_idx = Counter(model_predictions).most_common(1)[0][0]

        for class_name, class_idx in self.class_mapping.items():
            if class_idx == most_common_class_idx:
                return class_name

        return "Unknown"

if __name__ == "__main__":
    model_paths = [
        "k_fold_CNN_LSTM_landmark/model_fold_1.keras",
        "k_fold_CNN_LSTM_landmark/model_fold_2.keras",
        "k_fold_CNN_LSTM_landmark/model_fold_3.keras",
        "k_fold_CNN_LSTM_landmark/model_fold_4.keras",
        "k_fold_CNN_LSTM_landmark/model_fold_5.keras",
        "k_fold_CNN_LSTM_landmark/model_fold_6.keras",
        "k_fold_CNN_LSTM_landmark/model_fold_7.keras",
        "k_fold_CNN_LSTM_landmark/model_fold_8.keras",
        "k_fold_CNN_LSTM_landmark/model_fold_9.keras",
        "k_fold_CNN_LSTM_landmark/model_fold_10.keras",
    ]
    class_mapping_path = "workout_processed_data_landmark/class_mapping.npy"
    video_path = "Dataset\\push-up\\push-up_1.mp4" # replace with testing video

    predictor = WorkoutPredictor(model_paths, class_mapping_path)
    predicted_class = predictor.predict(video_path)
    print(f"Predicted Exercise Type: {predicted_class}")
