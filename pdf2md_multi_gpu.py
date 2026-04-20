# INPUT_DIR = Path("/opt/1panel/apps/copaw/CoPaw/data/workspaces/6jPowX/papers/low_light_uav_detection")
# OUTPUT_DIR = Path("/opt/1panel/apps/copaw/CoPaw/data/workspaces/6jPowX/papers/low_light_uav_detection_md")
# FAILED_LIST = OUTPUT_DIR / "_failed.txt"

# python pdf2md_multi_gpu.py \
#   --input /opt/1panel/apps/copaw/CoPaw/data/workspaces/6jPowX/papers/low_light_uav_detection \
#   --output /opt/1panel/apps/copaw/CoPaw/data/workspaces/6jPowX/papers/low_light_uav_detection_md \
#   --gpus 0,1,2 \
#   --workers-per-gpu 1
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch convert PDF to Markdown with marker using multi-GPU sharding.")
    parser.add_argument("--input", type=Path, required=True, help="Input PDF root directory")
    parser.add_argument("--output", type=Path, required=True, help="Output Markdown root directory")
    parser.add_argument(
        "--gpus",
        type=str,
        default="",
        help='Comma-separated GPU ids, e.g. "0,1,2". Empty means auto-detect all GPUs.',
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="Marker workers per GPU process. Strongly recommend 1 for stability.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing markdown files",
    )
    parser.add_argument(
        "--min-md-bytes",
        type=int,
        default=500,
        help="If markdown file size is smaller than this, treat it as failed",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=10,
        help="Progress polling interval in seconds",
    )
    parser.add_argument(
        "--http-proxy",
        type=str,
        default="",
        help='HTTP proxy, e.g. "http://127.0.0.1:7890"',
    )
    parser.add_argument(
        "--https-proxy",
        type=str,
        default="",
        help='HTTPS proxy, e.g. "http://127.0.0.1:7890"',
    )
    parser.add_argument(
        "--all-proxy",
        type=str,
        default="",
        help='ALL_PROXY, e.g. "socks5://127.0.0.1:7890"',
    )
    parser.add_argument(
        "--disable-hf-transfer",
        action="store_true",
        help="Set HF_HUB_ENABLE_HF_TRANSFER=0",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary sharded input directories",
    )
    parser.add_argument(
        "--oom-safe",
        action="store_true",
        help="Apply conservative marker settings to reduce GPU OOM risk (slower but more stable).",
    )
    parser.add_argument(
        "--marker-disable-multiprocessing",
        action="store_true",
        help="Pass --disable_multiprocessing to marker (reduces peak memory, slower).",
    )
    parser.add_argument(
        "--marker-disable-ocr",
        action="store_true",
        help="Pass --disable_ocr to marker (reduces memory; may reduce quality on scanned PDFs).",
    )
    parser.add_argument(
        "--marker-lowres-image-dpi",
        type=int,
        default=0,
        help="Pass --lowres_image_dpi to marker if > 0.",
    )
    parser.add_argument(
        "--marker-highres-image-dpi",
        type=int,
        default=0,
        help="Pass --highres_image_dpi to marker if > 0.",
    )
    parser.add_argument(
        "--marker-layout-batch-size",
        type=int,
        default=0,
        help="Pass --layout_batch_size to marker if > 0.",
    )
    parser.add_argument(
        "--marker-detection-batch-size",
        type=int,
        default=0,
        help="Pass --detection_batch_size to marker if > 0.",
    )
    parser.add_argument(
        "--marker-recognition-batch-size",
        type=int,
        default=0,
        help="Pass --recognition_batch_size to marker if > 0.",
    )
    parser.add_argument(
        "--marker-max-tasks-per-worker",
        type=int,
        default=0,
        help="Pass --max_tasks_per_worker to marker if > 0 (recycles worker to reduce leaks).",
    )
    return parser.parse_args()


def discover_pdf_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.pdf"))


def output_md_path(pdf_path: Path, input_root: Path, output_root: Path) -> Path:
    # Marker creates a subdirectory for each PDF with the same name (without extension)
    # and places the markdown file inside that subdirectory
    relative_path = pdf_path.relative_to(input_root)
    pdf_stem = pdf_path.stem  # Get filename without extension
    return (output_root / relative_path.parent / pdf_stem / pdf_stem).with_suffix(".md")


def should_skip(pdf_path: Path, input_root: Path, output_root: Path, overwrite: bool) -> bool:
    md_path = output_md_path(pdf_path, input_root, output_root)
    return md_path.exists() and not overwrite


def filter_pending_pdfs(pdf_files: list[Path], input_root: Path, output_root: Path, overwrite: bool) -> list[Path]:
    return [p for p in pdf_files if not should_skip(p, input_root, output_root, overwrite)]


def split_round_robin(items: list[Path], parts: int) -> list[list[Path]]:
    buckets: list[list[Path]] = [[] for _ in range(parts)]
    for idx, item in enumerate(items):
        buckets[idx % parts].append(item)
    return buckets


def detect_gpu_ids() -> list[int]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return []
        gpu_ids = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            gpu_ids.append(int(line))
        return gpu_ids
    except Exception:
        return []


def parse_gpu_ids(gpus_arg: str) -> list[int]:
    if not gpus_arg.strip():
        return detect_gpu_ids()
    ids = []
    for part in gpus_arg.split(","):
        part = part.strip()
        if not part:
            continue
        ids.append(int(part))
    return ids


def build_temp_input_tree(pdf_files: Iterable[Path], src_root: Path, temp_input_root: Path) -> None:
    if temp_input_root.exists():
        shutil.rmtree(temp_input_root)
    temp_input_root.mkdir(parents=True, exist_ok=True)

    for pdf in pdf_files:
        rel = pdf.relative_to(src_root)
        dst = temp_input_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        try:
            os.symlink(pdf.resolve(), dst)
        except Exception:
            shutil.copy2(pdf, dst)


def count_generated_markdowns(output_root: Path, pdf_files: list[Path], input_root: Path, min_md_bytes: int) -> tuple[int, int]:
    ok = 0
    failed = 0
    for pdf in pdf_files:
        md_path = output_md_path(pdf, input_root, output_root)
        if md_path.exists() and md_path.stat().st_size >= min_md_bytes:
            ok += 1
        else:
            failed += 1
    return ok, failed


def save_failed_list(output_root: Path, failed: list[Path]) -> Path:
    failed_path = output_root / "_failed.txt"
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.write_text("\n".join(str(p) for p in failed), encoding="utf-8")
    return failed_path


def save_success_list(output_root: Path, succeeded: list[Path]) -> Path:
    success_path = output_root / "_success.txt"
    success_path.parent.mkdir(parents=True, exist_ok=True)
    success_path.write_text("\n".join(str(p) for p in succeeded), encoding="utf-8")
    return success_path


def build_marker_env(
    gpu_id: int,
    http_proxy: str,
    https_proxy: str,
    all_proxy: str,
    disable_hf_transfer: bool,
) -> dict[str, str]:
    env = os.environ.copy()

    # Pin one process to one GPU.
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["TORCH_DEVICE"] = "cuda"

    # Reduce fragmentation.
    existing_alloc_conf = env.get("PYTORCH_CUDA_ALLOC_CONF", "").strip()
    if existing_alloc_conf:
        if "expandable_segments:True" not in existing_alloc_conf:
            env["PYTORCH_CUDA_ALLOC_CONF"] = existing_alloc_conf + ",expandable_segments:True"
    else:
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if http_proxy:
        env["http_proxy"] = http_proxy
        env["HTTP_PROXY"] = http_proxy
    if https_proxy:
        env["https_proxy"] = https_proxy
        env["HTTPS_PROXY"] = https_proxy
    if all_proxy:
        env["all_proxy"] = all_proxy
        env["ALL_PROXY"] = all_proxy
    if disable_hf_transfer:
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    return env


def launch_marker_process(
    gpu_id: int,
    shard_input_dir: Path,
    output_dir: Path,
    workers_per_gpu: int,
    log_file: Path,
    http_proxy: str,
    https_proxy: str,
    all_proxy: str,
    disable_hf_transfer: bool,
    marker_disable_multiprocessing: bool,
    marker_disable_ocr: bool,
    marker_lowres_image_dpi: int,
    marker_highres_image_dpi: int,
    marker_layout_batch_size: int,
    marker_detection_batch_size: int,
    marker_recognition_batch_size: int,
    marker_max_tasks_per_worker: int,
) -> subprocess.Popen:
    env = build_marker_env(
        gpu_id=gpu_id,
        http_proxy=http_proxy,
        https_proxy=https_proxy,
        all_proxy=all_proxy,
        disable_hf_transfer=disable_hf_transfer,
    )

    cmd = [
        "marker",
        str(shard_input_dir),
        "--output_dir",
        str(output_dir),
        "--workers",
        str(workers_per_gpu),
    ]
    if marker_disable_multiprocessing:
        cmd.append("--disable_multiprocessing")
    if marker_disable_ocr:
        cmd.append("--disable_ocr")
    if marker_lowres_image_dpi > 0:
        cmd.extend(["--lowres_image_dpi", str(marker_lowres_image_dpi)])
    if marker_highres_image_dpi > 0:
        cmd.extend(["--highres_image_dpi", str(marker_highres_image_dpi)])
    if marker_layout_batch_size > 0:
        cmd.extend(["--layout_batch_size", str(marker_layout_batch_size)])
    if marker_detection_batch_size > 0:
        cmd.extend(["--detection_batch_size", str(marker_detection_batch_size)])
    if marker_recognition_batch_size > 0:
        cmd.extend(["--recognition_batch_size", str(marker_recognition_batch_size)])
    if marker_max_tasks_per_worker > 0:
        cmd.extend(["--max_tasks_per_worker", str(marker_max_tasks_per_worker)])

    log_file.parent.mkdir(parents=True, exist_ok=True)
    fp = open(log_file, "a", encoding="utf-8")

    fp.write("\n" + "=" * 100 + "\n")
    fp.write(f"[START] {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    fp.write("[CMD] " + " ".join(cmd) + "\n")
    fp.write(f"[GPU] CUDA_VISIBLE_DEVICES={gpu_id}\n")
    fp.write(f"[PYTORCH_CUDA_ALLOC_CONF] {env.get('PYTORCH_CUDA_ALLOC_CONF', '')}\n")
    fp.flush()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
        universal_newlines=True,
        start_new_session=True,  # 创建新的进程组
    )

    def stream_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(f"[GPU {gpu_id}] {line}")
            sys.stdout.flush()
            fp.write(line)
            fp.flush()
        fp.write(f"[END] {time.strftime('%Y-%m-%d %H:%M:%S')} rc={process.poll()}\n")
        fp.flush()
        fp.close()

    t = threading.Thread(target=stream_output, daemon=False)  # 改为非守护线程
    t.start()
    process._stream_thread = t  # type: ignore[attr-defined]
    process._log_fp = fp  # type: ignore[attr-defined]
    return process


def terminate_processes(processes: list[subprocess.Popen]) -> None:
    """Terminate processes and their entire process groups to avoid orphaned child processes."""
    for p in processes:
        if p.poll() is None:
            try:
                # 首先尝试优雅终止
                p.terminate()
            except Exception:
                pass

    # 给进程一些时间优雅退出
    time.sleep(2)

    for p in processes:
        if p.poll() is None:
            try:
                # 强制终止整个进程组（如果设置了进程组）
                if hasattr(p, 'pid') and p.pid:
                    try:
                        # 尝试终止整个进程组
                        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError, AttributeError):
                        # 如果进程组不存在或没有权限，回退到单个进程终止
                        p.kill()
                else:
                    p.kill()
            except Exception:
                pass
    
    # 最后检查并确保所有进程都已终止
    for p in processes:
        if p.poll() is None:
            try:
                p.wait(timeout=1)
            except Exception:
                pass


def collect_results(
    all_pdfs: list[Path],
    input_root: Path,
    output_root: Path,
    min_md_bytes: int,
) -> tuple[list[Path], list[Path]]:
    succeeded: list[Path] = []
    failed: list[Path] = []

    for pdf in all_pdfs:
        md_path = output_md_path(pdf, input_root, output_root)
        if md_path.exists() and md_path.stat().st_size >= min_md_bytes:
            succeeded.append(pdf)
        else:
            failed.append(pdf)

    return succeeded, failed


def main() -> None:
    processes: list[subprocess.Popen] = []
    launched_jobs: list[tuple[int, list[Path], Path]] = []
    
    try:
        args = parse_args()

        input_root = args.input.resolve()
        output_root = args.output.resolve()
        temp_root = output_root / "_marker_shards"
        logs_root = output_root / "_logs"

        if not input_root.exists():
            print(f"Input directory does not exist: {input_root}")
            sys.exit(1)

        output_root.mkdir(parents=True, exist_ok=True)

        gpu_ids = parse_gpu_ids(args.gpus)
        if not gpu_ids:
            print("No GPU detected or specified. Aborting.")
            sys.exit(1)

        if args.workers_per_gpu < 1:
            print("--workers-per-gpu must be >= 1")
            sys.exit(1)

        if args.workers_per_gpu != 1:
            print(
                "[WARN] workers-per-gpu > 1 may cause OOM for marker on a single GPU. "
                "Use 1 unless you have verified your memory headroom.",
                flush=True,
            )

        all_pdfs = discover_pdf_files(input_root)
        if not all_pdfs:
            print(f"No PDF files found under: {input_root}")
            return

        pending_pdfs = filter_pending_pdfs(all_pdfs, input_root, output_root, args.overwrite)

        print(f"Input root: {input_root}")
        print(f"Output root: {output_root}")
        print(f"Total PDFs found: {len(all_pdfs)}")
        print(f"Pending PDFs: {len(pending_pdfs)}")
        print(f"GPUs: {gpu_ids}")
        print(f"Workers per GPU: {args.workers_per_gpu}")
        print(f"Min markdown bytes: {args.min_md_bytes}")
        print(f"Poll interval: {args.poll_interval}s")
        print(flush=True)

        if not pending_pdfs:
            print("All PDF files already converted.")
            return

        shards = split_round_robin(pending_pdfs, len(gpu_ids))

        def handle_sigint(signum, frame) -> None:  # type: ignore[no-untyped-def]
            print("\n[INFO] Caught interrupt signal, terminating child processes...", flush=True)
            terminate_processes(processes)
            sys.exit(130)

        signal.signal(signal.SIGINT, handle_sigint)
        signal.signal(signal.SIGTERM, handle_sigint)

        marker_disable_multiprocessing = args.marker_disable_multiprocessing
        marker_disable_ocr = args.marker_disable_ocr
        marker_lowres_image_dpi = args.marker_lowres_image_dpi
        marker_highres_image_dpi = args.marker_highres_image_dpi
        marker_layout_batch_size = args.marker_layout_batch_size
        marker_detection_batch_size = args.marker_detection_batch_size
        marker_recognition_batch_size = args.marker_recognition_batch_size
        marker_max_tasks_per_worker = args.marker_max_tasks_per_worker

        if args.oom_safe:
            marker_disable_multiprocessing = True
            marker_layout_batch_size = marker_layout_batch_size or 1
            marker_detection_batch_size = marker_detection_batch_size or 1
            marker_recognition_batch_size = marker_recognition_batch_size or 1
            marker_max_tasks_per_worker = marker_max_tasks_per_worker or 1
            marker_highres_image_dpi = marker_highres_image_dpi or 144

        for gpu_id, shard_pdfs in zip(gpu_ids, shards):
            if not shard_pdfs:
                continue

            shard_input_dir = temp_root / f"gpu_{gpu_id}_input"
            build_temp_input_tree(shard_pdfs, input_root, shard_input_dir)

            log_file = logs_root / f"gpu_{gpu_id}.log"
            process = launch_marker_process(
                gpu_id=gpu_id,
                shard_input_dir=shard_input_dir,
                output_dir=output_root,
                workers_per_gpu=args.workers_per_gpu,
                log_file=log_file,
                http_proxy=args.http_proxy,
                https_proxy=args.https_proxy,
                all_proxy=args.all_proxy,
                disable_hf_transfer=args.disable_hf_transfer,
                marker_disable_multiprocessing=marker_disable_multiprocessing,
                marker_disable_ocr=marker_disable_ocr,
                marker_lowres_image_dpi=marker_lowres_image_dpi,
                marker_highres_image_dpi=marker_highres_image_dpi,
                marker_layout_batch_size=marker_layout_batch_size,
                marker_detection_batch_size=marker_detection_batch_size,
                marker_recognition_batch_size=marker_recognition_batch_size,
                marker_max_tasks_per_worker=marker_max_tasks_per_worker,
            )
            processes.append(process)
            launched_jobs.append((gpu_id, shard_pdfs, shard_input_dir))

            print(
                f"[LAUNCHED] GPU {gpu_id}: {len(shard_pdfs)} PDFs | "
                f"log={log_file}",
                flush=True,
            )

        if not processes:
            print("Nothing to launch.")
            return

        start_time = time.time()

        while True:
            alive = 0
            per_gpu_status: list[str] = []

            for gpu_id, shard_pdfs, _ in launched_jobs:
                ok_count, failed_count = count_generated_markdowns(
                    output_root=output_root,
                    pdf_files=shard_pdfs,
                    input_root=input_root,
                    min_md_bytes=args.min_md_bytes,
                )
                total = len(shard_pdfs)
                per_gpu_status.append(f"GPU {gpu_id}: {ok_count}/{total} ready")
                matching = [p for p in processes if p.poll() is None and p.args[0] == "marker"]  # type: ignore[index]
                # This matching is not used to identify a specific process; we only need total alive.
                _ = failed_count
            for p in processes:
                if p.poll() is None:
                    alive += 1

            elapsed = int(time.time() - start_time)
            print(
                f"[PROGRESS] elapsed={elapsed}s | alive={alive} | " + " | ".join(per_gpu_status),
                flush=True,
            )

            if alive == 0:
                break

            time.sleep(args.poll_interval)

        for p in processes:
            try:
                p.wait(timeout=1)
            except Exception:
                pass

        succeeded, failed = collect_results(
            all_pdfs=pending_pdfs,
            input_root=input_root,
            output_root=output_root,
            min_md_bytes=args.min_md_bytes,
        )

        success_file = save_success_list(output_root, succeeded)
        failed_file = save_failed_list(output_root, failed)

        print("\n=== Final Summary ===")
        print(f"Pending total: {len(pending_pdfs)}")
        print(f"Succeeded:     {len(succeeded)}")
        print(f"Failed:        {len(failed)}")
        print(f"Success list:  {success_file}")
        print(f"Failed list:   {failed_file}")

        if not args.keep_temp and temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)
            print(f"Temporary shard dirs removed: {temp_root}")
        else:
            print(f"Temporary shard dirs kept: {temp_root}")

        if failed:
            print("\nSome PDFs failed. Check these logs:")
            for gpu_id, _, _ in launched_jobs:
                print(f"- {logs_root / f'gpu_{gpu_id}.log'}")
        
    finally:
        # 确保在任何情况下（正常退出、异常、信号）都清理子进程
        if processes:
            print("[CLEANUP] Terminating child processes...", flush=True)
            terminate_processes(processes)
            
            # 等待所有日志线程完成
            for p in processes:
                if hasattr(p, '_stream_thread'):
                    try:
                        p._stream_thread.join(timeout=5.0)
                    except Exception:
                        pass
                if hasattr(p, '_log_fp') and p._log_fp and not p._log_fp.closed:
                    try:
                        p._log_fp.close()
                    except Exception:
                        pass


if __name__ == "__main__":
    main()
