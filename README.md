
# Thai Supreme Court GPT (from scratch, PyTorch)

Implements a GPT model **from scratch** (attention, transformer blocks, masking, etc.) to generate Thai Supreme Court–style texts. Uses **WangchanBERTa SentencePiece tokenizer** from HuggingFace per project spec.

## Environment
```bash
pip install -r requirements.txt
```

## Data
Put all `.txt` court-decision files into `data/`. 

## Train (Large config by default)
```bash
python train.py --data_dir data --epochs 3 --batch_size 8 --block_size 512 
python train.py --device cuda --data_dir data --epochs 1 --max_steps 50000 --batch_size 24 --block_size 512 --n_layer 12 --n_head 12 --n_embd 768 --num_workers 8 --save_every 20000 --eval_every 0 --tokenizer_name "airesearch/wangchanberta-base-att-spm-uncased" --save_full_state --throttle_ms 0
```



## Generate
```bash
python generate.py --ckpt checkpoints\gpt_step50000.pt --tokenizer_name "airesearch/wangchanberta-base-att-spm-uncased" --prompt "ซื้อขายที่ดินต้องเป็นหนังสือ มิฉะนั้นสัญญาเป็นโมฆะทันที" --max_new_tokens 200 --temperature 0.9 --top_p 0.95

```
## Please download the data and gpt_step50000.pt I have provided in canvas for generating text
