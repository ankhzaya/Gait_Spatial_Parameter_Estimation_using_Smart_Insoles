# Gait_Spatial_Parameter_Estimation_using_Smart_Insoles

## Getting Started

1. **Download the checkpoint and sample dataset**  
   [checkpoints and dataset](https://drive.google.com/file/d/1Oon0OVKRZRGloLiZz0D-TCSHbB7Z0ubo/view?usp=sharing)

2. **Edit walking speed in the config file**  
   Change `configs.test.speeds` as needed.

3. **Run the full test and analysis pipeline:**  
   - Create stride-wise estimations:
     ```bash
     python3 stride_test.py
     ```
   - Average by subject and trial (creates subject-wise results):
     ```bash
     python3 avg_by_sbj_trial.py
     ```
   - Evaluate the results and create scatter/Bland-Altman plots:
     ```bash
     python3 combined_bald_altman.py
     python3 combined_scatter_plots.py
     ```
   > You can change the evaluation mode between stride-wise and subject-wise as needed in the scripts.

