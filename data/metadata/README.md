file_name → direct path into your dataset

video_type → which deepfake technique (for confusion matrix)

label → ground truth (real/fake)

frame_counts → tells us how many frame samples we could extract

width, height → useful for error handling & compression checks

codec → useful for describing dataset in the ICA

file_size → useful for comparing model latency vs input size

id → simple unique identifier for logs, results, dataframe merge

Created video_type, renamed video files and modified original metadata records to map to corresponding files for all chosen/randomly selected videos (10 from each deepfake technique, 60 originals),