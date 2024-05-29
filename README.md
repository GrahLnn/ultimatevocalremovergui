# Ultimate Vocal Remover GUI v5.6

去掉了GUI使得能够变成库来调用

当前只提取了`MDX23C-8KFFT-InstVoc_HQ.ckpt`的调用

准备`MDX23C-8KFFT-InstVoc_HQ.ckpt`和预先保存的参数`pkl`文件（结构复杂，字段极多懒得一个一个解了，全部打包传进去）

```python
from uvr.uvr_cli import infer
with open("asset/model/my_object.pkl", "rb") as f:
    process_data = pickle.load(f)
infer(
    part,
    out_path,
    "asset/model/MDX23C-8KFFT-InstVoc_HQ.ckpt",
    "asset/model/UVR-DeNoise-Lite.pth",
    "asset/model/UVR-DeEcho-DeReverb.pth",
    process_data,
)
```
