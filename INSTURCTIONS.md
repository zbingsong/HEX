# How to

First go to MUSK's HuggingFace repo and request access to the model. Once you are granted access, download the model's weights:
```bash
# assume you are inside HEX/
wget --header="Authorization: Bearer <your huggingface token>" https://huggingface.co/xiangjx/musk/resolve/main/model.safetensors
mv model.safetensors hex
```

Run the following:

```bash
conda env create -f environment.yml -n hex
conda activate hex
pip install fairscale
pip install einops
git clone https://github.com/lilab-stanford/MUSK /tmp/MUSK
pip install -r /tmp/MUSK/requirements.txt
pip install -e /tmp/MUSK

python hex/infer_wsi_hex.py \
    --wsi-path <path to WSI .svs file> \
    --checkpoint-path hex/checkpoint.pth \
    --output-dir <output directory> \
    --level <WSI level to run inference on> \
    --patch-size 224 \
    --stride <downsampling factor, default=4> \
    --batch-size <inference batch size>
```

# Important Things

- `HEX` predicts a single value per 224x224 tile per biomarker.
- In the implemented WSI inference pipeline, the strategy is to run inference at a certain stride, which produces an output that is a downsampling of the input. The downsampling factor is equal to the stride value. Tile coordinates `(x, y)` in input corresponds to output coordinates `(x/stride, y/stride)`. If stride is less than tile size (224), each pixel in input is used multiple times in inference, and the inference output is a "blurred" version of the biomarker image.
