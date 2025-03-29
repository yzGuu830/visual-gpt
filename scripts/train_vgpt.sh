echo "<stage 1> Training VQGAN tokenizer for the STFDogs dataset..."
python train_tokenizer.py  --exp_name vqgan-stfdogs  --output_path ../outputs  --conf conf/stfdogs.yaml  --wandb visual-gpt

echo "<stage 2> Training Visual GPT model on the STFDogs dataset..."
python train_gpt.py --exp_name vqgan-stfdogs  --output_path ../outputs  --tokenizer_path ../outputs/vqgan-stfdogs  --conf conf/stfdogs.yaml  --wandb visual-gpt