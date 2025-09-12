```
プログラムがinput_idsが原因でエラーになる環境(dockerコンテナ内)
================================================================================
 [ CHECK ENVIRONMENT INFO ]
================================================================================
python      : 3.10.14 (main, May  6 2024, 19:42:50) [GCC 11.2.0]
================================================================================
torch             : 2.3.1      @ /opt/conda/lib/python3.10/site-packages/torch/__init__.py
/opt/conda/lib/python3.10/site-packages/transformers/utils/hub.py:127: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
transformers      : 4.44.2     @ /opt/conda/lib/python3.10/site-packages/transformers/__init__.py
accelerate        : 0.33.0     @ /opt/conda/lib/python3.10/site-packages/accelerate/__init__.py
datasets          : 2.20.0     @ /opt/conda/lib/python3.10/site-packages/datasets/__init__.py
peft              : 0.11.0     @ /opt/conda/lib/python3.10/site-packages/peft/__init__.py
evaluate          : 0.4.2      @ /opt/conda/lib/python3.10/site-packages/evaluate/__init__.py
jiwer             : unknown    @ /opt/conda/lib/python3.10/site-packages/jiwer/__init__.py
huggingface_hub   : 0.34.4     @ /opt/conda/lib/python3.10/site-packages/huggingface_hub/__init__.py
soundfile         : 0.12.1     @ /opt/conda/lib/python3.10/site-packages/soundfile.py
pydub             : unknown    @ /opt/conda/lib/python3.10/site-packages/pydub/__init__.py
pandas            : 2.2.2      @ /opt/conda/lib/python3.10/site-packages/pandas/__init__.py
fugashi           : unknown    @ /opt/conda/lib/python3.10/site-packages/fugashi/__init__.py
unidic_lite       : unknown    @ /opt/conda/lib/python3.10/site-packages/unidic_lite/__init__.py
debugpy           : 1.8.1      @ /opt/conda/lib/python3.10/site-packages/debugpy/__init__.py
torchcodec        : Not installed (ImportError: cannot import name 'register_fake' from 'torch.library' (/opt/conda/lib/python3.10/site-packages/torch/library.py))
ruamel.yaml       : 0.17.40    @ /opt/conda/lib/python3.10/site-packages/ruamel/yaml/__init__.py
ruamel.yaml.clib  : Not installed (ModuleNotFoundError: No module named 'ruamel.yaml.clib')
numpy             : 1.26.4     @ /opt/conda/lib/python3.10/site-packages/numpy/__init__.py
librosa           : Not installed (ModuleNotFoundError: No module named 'librosa')
================================================================================
torch.cuda.is_available : True
torch.version.cuda      : 12.1
cuDNN version           : 8902
num GPUs                : 1
  - GPU[0] name=NVIDIA RTX 4500 Ada Generation, total_mem=24569MB, cc=8.9
================================================================================
ffmpeg                  : /opt/conda/bin/ffmpeg
ffmpeg version line     : ffmpeg version 4.3 Copyright (c) 2000-2020 the FFmpeg developers
================================================================================
env (HF/Torch caches)   : {
  "HF_HOME": "/root/.cache/huggingface",
  "TRANSFORMERS_CACHE": "/root/.cache/torch/transformers",
  "HF_DATASETS_CACHE": null,
  "TORCH_HOME": null
}
================================================================================
 [ END ]
================================================================================
```

```
プログラムが動いた環境(ローカル)
================================================================================
 [ CHECK ENVIRONMENT INFO ]
================================================================================
python      : 3.9.23 (main, Jun  5 2025, 13:40:20)
[GCC 11.2.0]
================================================================================
torch             : 2.8.0+cu128 @ /home/mtmt/anaconda3/envs/audio_analysis/lib/python3.9/site-packages/torch/__init__.py
transformers      : 4.56.1     @ /home/mtmt/anaconda3/envs/audio_analysis/lib/python3.9/site-packages/transformers/__init__.py
accelerate        : 1.10.1     @ /home/mtmt/anaconda3/envs/audio_analysis/lib/python3.9/site-packages/accelerate/__init__.py
datasets          : 4.0.0      @ /home/mtmt/anaconda3/envs/audio_analysis/lib/python3.9/site-packages/datasets/__init__.py
peft              : 0.17.1     @ /home/mtmt/anaconda3/envs/audio_analysis/lib/python3.9/site-packages/peft/__init__.py
evaluate          : 0.4.5      @ /home/mtmt/anaconda3/envs/audio_analysis/lib/python3.9/site-packages/evaluate/__init__.py
jiwer             : unknown    @ /home/mtmt/anaconda3/envs/audio_analysis/lib/python3.9/site-packages/jiwer/__init__.py
huggingface_hub   : 0.34.4     @ /home/mtmt/anaconda3/envs/audio_analysis/lib/python3.9/site-packages/huggingface_hub/__init__.py
soundfile         : 0.13.1     @ /home/mtmt/anaconda3/envs/audio_analysis/lib/python3.9/site-packages/soundfile.py
pydub             : unknown    @ /home/mtmt/anaconda3/envs/audio_analysis/lib/python3.9/site-packages/pydub/__init__.py
pandas            : 2.3.2      @ /home/mtmt/anaconda3/envs/audio_analysis/lib/python3.9/site-packages/pandas/__init__.py
fugashi           : unknown    @ /home/mtmt/anaconda3/envs/audio_analysis/lib/python3.9/site-packages/fugashi/__init__.py
unidic_lite       : unknown    @ /home/mtmt/anaconda3/envs/audio_analysis/lib/python3.9/site-packages/unidic_lite/__init__.py
debugpy           : 1.8.16     @ /home/mtmt/anaconda3/envs/audio_analysis/lib/python3.9/site-packages/debugpy/__init__.py
torchcodec        : Not installed (RuntimeError: Could not load libtorchcodec. Likely causes:
          1. FFmpeg is not properly installed in your environment. We support
             versions 4, 5, 6 and 7.
          2. The PyTorch version (2.8.0+cu128) is not compatible with
             this version of TorchCodec. Refer to the version compatibility
             table:
             https://github.com/pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec.
          3. Another runtime dependency; see exceptions below.
        The following exceptions were raised as we tried to load libtorchcodec:

[start of libtorchcodec loading traceback]
FFmpeg version 7: libavutil.so.59: cannot open shared object file: No such file or directory
FFmpeg version 6: libavutil.so.58: cannot open shared object file: No such file or directory
FFmpeg version 5: libavutil.so.57: cannot open shared object file: No such file or directory
FFmpeg version 4: /lib/x86_64-linux-gnu/libgobject-2.0.so.0: undefined symbol: ffi_type_uint32, version LIBFFI_BASE_7.0
[end of libtorchcodec loading traceback].)
ruamel.yaml       : 0.18.15    @ /home/mtmt/anaconda3/envs/audio_analysis/lib/python3.9/site-packages/ruamel/yaml/__init__.py
ruamel.yaml.clib  : Not installed (ModuleNotFoundError: No module named 'ruamel.yaml.clib')
numpy             : 2.0.2      @ /home/mtmt/anaconda3/envs/audio_analysis/lib/python3.9/site-packages/numpy/__init__.py
librosa           : Not installed (ModuleNotFoundError: No module named 'librosa')
================================================================================
torch.cuda.is_available : True
torch.version.cuda      : 12.8
cuDNN version           : 91002
num GPUs                : 1
  - GPU[0] name=NVIDIA TITAN RTX, total_mem=24204MB, cc=7.5
================================================================================
ffmpeg                  : /usr/bin/ffmpeg
ffmpeg version line     : ffmpeg version 4.2.7-0ubuntu0.1 Copyright (c) 2000-2022 the FFmpeg developers
================================================================================
env (HF/Torch caches)   : {
  "HF_HOME": null,
  "TRANSFORMERS_CACHE": null,
  "HF_DATASETS_CACHE": null,
  "TORCH_HOME": null
}
================================================================================
 [ END ]
================================================================================
```
