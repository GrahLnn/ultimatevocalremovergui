from setuptools import setup, find_packages

setup(
    name="uvr",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # 依赖列表
        "altgraph",
        "audioread",
        "certifi",
        "cffi",
        "cryptography",
        "einops",
        "future",
        "julius",
        "kthread",
        "librosa",
        "llvmlite",
        "matchering",
        "ml_collections",
        "natsort",
        "omegaconf",
        "opencv-python",
        "Pillow",
        "psutil",
        "pydub",
        "pyglet",
        "pyperclip",
        "pyrubberband",
        "pytorch_lightning",
        "PyYAML",
        "resampy",
        "scipy",
        "soundstretch",
        "torch",
        "urllib3",
        "wget",
        "samplerate",
        "screeninfo",
        "diffq",
        "playsound",
        "onnx",
        "onnxruntime",
        "onnxruntime-gpu",
        "onnx2pytorch",
        "SoundFile",
        # "PySoundFile.post1; sys_platform == 'darwin'",
        "numpy",
    ],
    # 其他元数据
)
