from DataPreProcessing import WorkoutDataPreparator
from train import WorkoutModelTrainer
if __name__ == "__main__":
    base_folder = r"Dataset"
    target_classes = ['bench press', 'barbell biceps curl', 'chest fly machine', 'deadlift',
                      'decline bench press', 'hammer curl', 'hip thrust', 'incline bench press',
                      'lat pulldown', 'lateral raise', 'leg extension', 'leg raises',#'plank',
                      'pull Up', 'push-up', 'romanian deadlift', 'russian twist', 'shoulder press',
                      'squat', 't bar row', 'tricep dips', 'tricep Pushdown']
    save_dir = r"workout_processed_data_landmark"

    preparator = WorkoutDataPreparator(
        base_folder=base_folder,
        target_classes=target_classes,
        max_frames=45
    )
    X, y, class_mapping, confidences = preparator.prepare_and_save_dataset(save_dir)

    # Updated paths to match new data location
    data_dir = "workout_processed_data_landmark"
    save_path = "k_fold_CNN_LSTM_landmark"

    trainer = WorkoutModelTrainer(data_dir)
    fold_results, fold_histories = trainer.train_with_kfold(
        epochs=100,
        batch_size=16,
        n_splits=10,
        save_path=save_path
    )