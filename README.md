# UWMoE
UWMoE: Uncertainty-Guided Wavelet Mixture-of-Experts for Unified Hyperspectral Image Restoration
## Structure

```text
HSI-AiOR/
├── data/                   # Raw, preprocessed, and LMDB data
├── net/
│   ├── S2UM.py             # Main restoration network
│   └── MoE.py              # Mixture-of-Experts modules
├── utils/
│   ├── mat_data.py         # Dataset preprocessing and splitting
│   ├── lmdb_patch.py       # Patch extraction and LMDB generation
│   ├── dataset_utils.py    # Training and testing datasets
│   └── ...                 # Degradations, metrics, and utility functions
├── main.py                 # Training and testing entry point
└── test_function.py        # Evaluation functions
```

## Data Preparation

### 1. Organize the Raw Data

Download the required hyperspectral datasets and place them under `data/`.



### 2. Preprocess the Data

Open `utils/mat_data.py` and:

1. Replace the example paths such as `/home/xxx/...` and `/data/...`.
2. Select the preprocessing function corresponding to your dataset.
3. Uncomment only the required function in the final `if __name__ == '__main__':` block.
4. Save all generated files under the project `data/` directory.

For example:

```python
if __name__ == '__main__':
    create_PaviaC_dataset()
```

Run the preprocessing script from the project root:

```bash
python utils/mat_data.py
```


### 3. Generate the LMDB Training Set

Open `utils/lmdb_patch.py` and update:

- `source_folders`: directories containing the preprocessed training `.mat` files;
- `destination_folders`: directories used to collect the `.mat` files;
- `destination_folders_patch`: output paths for the LMDB datasets.



Generate the training patches:

```bash
python utils/lmdb_patch.py
```

The generated database should have the following structure:

```text
data/Train/
└── Remote_sensing_minmax_patch_64.db/
    ├── data.mdb
    ├── lock.mdb
    └── meta_info.txt
```

## Training

Set the training parameters in `train()` inside `main.py`, including:

- `data_type`: `natural_scene` or `remote_sensing`;
- `db_path`: path to the generated LMDB directory;
- `epochs`, `batch_size`, and `lr`;
- `num_workers`;
- `num_gpus`: use `[0]` for the first GPU;
- `ckpt_path`: optional pretrained checkpoint.

Keep `train()` enabled at the end of `main.py`:

```python
if __name__ == '__main__':
    train()
    # test()
```

Start training:

```bash
python main.py
```

Checkpoints are saved under `ckpt/`, and TensorBoard logs are saved under `logs/`.

## Testing

The available modes are:

| Mode | Task |
| --- | --- |
| `0` | Gaussian denoising |
| `1` | Non-i.i.d. Gaussian denoising |
| `2` | Destriping |
| `3` | Deadline denoising |
| `4` | Impulse denoising |
| `5` | Gaussian deblurring |
| `6` | Motion deblurring |
| `7` | Super-Resolution |
| `8` | Inpainting |
| `9` | Dehazing |
| `10` | Band Completion |
| `11` | Poisson denoising |

To evaluate one task, replace `for mode in range(12)` with a list containing the required mode, for example `for mode in [0]`.

Enable `test()` at the end of `main.py`:

```python
if __name__ == '__main__':
    # train()
    test()
```

Run testing:

```bash
python main.py
```

Restored images and quantitative results are saved under `output/`.