To build the baseline state-action mapping matrix, run the training pipeline. This will update state matrices and output a diagnostic performance visualization graph upon completion.
run `python train.py`

Once q_table.npy is generated, spin up the verification loop to animate a greedy evaluation descent sequence using Rocky's terminal configuration:
run `python run.py`
