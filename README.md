# PDF to Markdown Multi‑GPU Converter

A high‑throughput tool to batch‑convert PDF files to Markdown using [`marker`](https://github.com/VikParuchuri/marker) across multiple NVIDIA GPUs.  
It shards pending PDFs in a round‑robin fashion, runs one `marker` process per GPU, and automatically skips already converted files.

## Features

- **Multi‑GPU sharding** – distribute PDFs across any number of GPUs
- **Resume capability** – skips PDFs whose output Markdown already exists and meets minimum size
- **Per‑GPU logging** – each GPU writes its own log file (stdout/stderr) for easy debugging
- **Robust termination** – on `Ctrl+C` or `SIGTERM`, the whole process tree is cleaned up
- **OOM‑safe mode** – conservative marker settings to reduce GPU memory pressure
- **Proxy support** – HTTP/HTTPS/ALL_PROXY for model downloading or API access
- **Success/failure tracking** – generates `_success.txt` and `_failed.txt` in the output directory

## Prerequisites

- **Python 3.10 – 3.12** (3.13 not officially supported)
- **Linux** (the script uses `os.killpg`; may work on macOS, but not tested)
- **NVIDIA GPU(s)** with **nvidia‑smi** available
- **CUDA** compatible with PyTorch (recommended: CUDA 11.8 + PyTorch 2.6.0)

## Installation

1. **Install PyTorch** (CUDA version) **before** installing `marker`:

   ```bash
   pip3 install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


# PDF 转 Markdown 多 GPU 转换工具

一个批量将 PDF 文件转换为 Markdown 的高吞吐工具。它基于 [`marker`](https://github.com/VikParuchuri/marker) 实现，并在多个 NVIDIA GPU 上并发运行。  
脚本采用轮询（round‑robin）方式将待转换的 PDF 分片到各 GPU，每个 GPU 独立运行一个 `marker` 进程，并自动跳过已转换成功的文件。

## 功能特性

- **多 GPU 分片** – 将 PDF 文件平均分配到任意数量的 GPU 上
- **断点续传** – 若输出目录中已存在对应 Markdown 文件且大小满足最小要求，则自动跳过
- **独立日志** – 每个 GPU 拥有独立的日志文件（记录 stdout/stderr），便于调试
- **健壮的进程终止** – 按下 `Ctrl+C` 或收到 `SIGTERM` 信号时，会完整清理所有子进程树
- **OOM 安全模式** – 采用保守的 marker 参数，降低 GPU 显存压力
- **代理支持** – 支持 HTTP/HTTPS/ALL_PROXY，方便模型下载或 API 调用
- **成功/失败记录** – 在输出目录自动生成 `_success.txt` 和 `_failed.txt`

## 环境要求

- **Python 3.10 – 3.12**（3.13 尚未官方支持）
- **Linux**（脚本使用了 `os.killpg`，macOS 可能部分兼容但未测试）
- **NVIDIA GPU** 并已安装 `nvidia-smi` 命令
- **CUDA** 与 PyTorch 兼容（推荐 CUDA 11.8 + PyTorch 2.6.0）

## 安装步骤

1. **先安装 PyTorch（CUDA 版）**，**然后**再安装 `marker`：

   ```bash
   pip3 install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
