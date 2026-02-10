# Thesis Dataset Generation - Ongoing Progress

This document tracks the batched extraction of fMRI data for the motor task dataset. Due to storage limitations, data is processed in batches and must be moved to external storage immediately after each run.

## 📊 Current Status
- **Current Batch:** Batch 7 (Final batch for 4-class extraction)
- **Subjects:** sub-61 to sub-68
- **Configuration:** `EXTRACT_4_CLASSES_ONLY = True`
- **Target Classes:** Left leg, Right leg, Forearm, Upper arm

## 🛠 Batch Plan (4 Classes)
- [x] **Batch 1:** sub-01 to sub-10
- [x] **Batch 2:** sub-11 to sub-20
- [x] **Batch 3:** sub-21 to sub-30
- [x] **Batch 4:** sub-31 to sub-40
- [x] **Batch 5:** sub-41 to sub-50
- [x] **Batch 6:** sub-51 to sub-60
- [x] **Batch 7:** sub-61 to sub-68 (COMPLETED)

## 📥 Post-Extraction Instructions
1.  **Download:** Retrieve the output from the Kaggle working directory: `/kaggle/working/batch_07`.
2.  **Backup:** Upload the downloaded files to the OneDrive folder: `/thesis_data/batch_07/`.
3.  **Cleanup:** Ensure the Kaggle working directory is cleared before starting any new extractions to avoid "Disk full" errors.

## 🚀 Next Steps
- **Scenario A (Data Merging):** If 4 classes are sufficient, proceed to merging all 7 batches for model training.
- **Scenario B (Full Dataset):** To extract all 13 classes, reset `BATCH_NUMBER = 1`, set `EXTRACT_4_CLASSES_ONLY = False`, and follow the ~16-batch plan.
