This is the code repository for our paper https://aclanthology.org/2021.emnlp-main.119.pdf.
For plots in the paper:
type can take following values:
sra: This corresponds to attractors being semantically related to the target/background.
sua: This corresponds to attractors being semantically unrelated to the target/background.
inter: This corresponds to attractors being semantically related to the target/background and being present between key entity and critical fact.
T: This corresponds to attractor being semantically related to target word.
B: This corresponds to attractor being semantically related to background word.
Accuracy plots:
python plotDistanceVsAccuracy.py type 
Relative Prob plots:
python plotRelativeProbVsAccuracy.py type
Accuracy plot for Varying Key entity position:
python plotDistanceVsAccuracyKeyEntityLater.py


