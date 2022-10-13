### Multi-Granularity Cross-modal Alignment for Generalized Medical Visual Representation Learning 

### Usage

```
pip install -e .
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirments.txt
```


### Training

```
cd mgca/models/mgcm
CUDA_VISIBLE_DEVICES=0,1 python mgcm_module.py --gpus 2 --strategy ddp
```

### Finetune

```
cd mgca/models/mgcm
CUDA_VISIBLE_DEVICES=1 python mgcm_finetuner.py --gpus 1 --dataset chexpert --data_pct 0.01
```

More downstream tasks and pre-trained model will come soon!