"""
RVC training steps — subprocess + FAISS (same flow as infer-web train tab).
Chạy với cwd = thư mục rvc_standalone (infer/, configs/, i18n/ nằm cùng cấp).
"""
from __future__ import annotations

import logging
import os
import platform
import subprocess
import traceback
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from training_pipeline.params import TrainingParams, resolve_pretrained_paths

logger = logging.getLogger(__name__)


def _run(cmd: str, cwd: Path, on_line: Optional[Callable[[str], None]] = None) -> int:
    logger.info("Execute: %s", cmd)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd) + os.pathsep + env.get("PYTHONPATH", "")
    p = subprocess.Popen(
        cmd,
        shell=True,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    assert p.stdout is not None
    for line in p.stdout:
        line = line.rstrip("\n")
        print(line, flush=True)
        if on_line:
            on_line(line)
    p.wait()
    return p.returncode or 0


def step_preprocess(rvc_root: Path, config, p: TrainingParams) -> None:
    sr = p.sr_dict[p.sample_rate_label]
    exp = p.experiment_name
    os.makedirs(rvc_root / "logs" / exp, exist_ok=True)
    logf = rvc_root / "logs" / exp / "preprocess.log"
    logf.write_text("", encoding="utf-8")
    cmd = '"%s" infer/modules/train/preprocess.py "%s" %s %s "%s/logs/%s" %s %.1f' % (
        config.python_cmd,
        p.trainset_dir,
        sr,
        p.num_processes,
        rvc_root,
        exp,
        config.noparallel,
        config.preprocess_per,
    )
    code = _run(cmd, rvc_root)
    if code != 0:
        raise RuntimeError("preprocess.py exited with %s" % code)


def step_extract_f0_and_features(rvc_root: Path, config, p: TrainingParams) -> None:
    exp = p.experiment_name
    os.makedirs(rvc_root / "logs" / exp, exist_ok=True)
    log_path = rvc_root / "logs" / exp / "extract_f0_feature.log"
    log_path.write_text("", encoding="utf-8")
    gpus = p.gpu_devices_train.split("-") if p.gpu_devices_train else ["0"]

    if p.if_f0:
        if p.f0_method != "rmvpe_gpu":
            cmd = (
                '"%s" infer/modules/train/extract/extract_f0_print.py "%s/logs/%s" %s %s'
                % (
                    config.python_cmd,
                    rvc_root,
                    exp,
                    p.num_processes,
                    p.f0_method,
                )
            )
            code = _run(cmd, rvc_root)
            if code != 0:
                raise RuntimeError("extract_f0_print.py exited with %s" % code)
        else:
            if p.gpus_for_rmvpe != "-":
                ids = p.gpus_for_rmvpe.split("-")
                leng = len(ids)
                procs = []
                for idx, n_g in enumerate(ids):
                    cmd = (
                        '"%s" infer/modules/train/extract/extract_f0_rmvpe.py %s %s %s "%s/logs/%s" %s '
                        % (
                            config.python_cmd,
                            leng,
                            idx,
                            n_g,
                            rvc_root,
                            exp,
                            config.is_half,
                        )
                    )
                    procs.append(subprocess.Popen(cmd, shell=True, cwd=str(rvc_root)))
                for x in procs:
                    x.wait()
                    if x.returncode not in (0, None):
                        raise RuntimeError("extract_f0_rmvpe.py failed")
            else:
                cmd = (
                    config.python_cmd
                    + ' infer/modules/train/extract/extract_f0_rmvpe_dml.py "%s/logs/%s" '
                    % (rvc_root, exp)
                )
                code = _run(cmd, rvc_root)
                if code != 0:
                    raise RuntimeError("extract_f0_rmvpe_dml.py exited with %s" % code)

    leng = len(gpus)
    for idx, n_g in enumerate(gpus):
        cmd = (
            '"%s" infer/modules/train/extract_feature_print.py %s %s %s %s "%s/logs/%s" %s %s'
            % (
                config.python_cmd,
                config.device,
                leng,
                idx,
                n_g,
                rvc_root,
                exp,
                p.version,
                config.is_half,
            )
        )
        code = _run(cmd, rvc_root)
        if code != 0:
            raise RuntimeError(
                "extract_feature_print.py thất bại (exit %s). Đọc output phía trên và "
                "file logs/%s/extract_f0_feature.log — thường gặp: Hubert hỏng/thiếu, "
                "fairseq/torch lỗi, hoặc VRAM không đủ khi đưa model lên GPU."
                % (code, exp)
            )


def _write_filelist(rvc_root: Path, config, p: TrainingParams) -> None:
    import json
    import pathlib
    from random import shuffle

    now_dir = str(rvc_root)
    exp_dir1 = p.experiment_name
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if p.version == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if p.if_f0:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    spk_id5 = str(p.speaker_id)
    for name in names:
        if p.if_f0:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if p.version == "v1" else 768
    sr2 = p.sample_rate_label
    if p.if_f0:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w", encoding="utf-8") as f:
        f.write("\n".join(opt))

    if p.version == "v1" or sr2 == "40k":
        config_path = "v1/%s.json" % sr2
    else:
        config_path = "v2/%s.json" % sr2
    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(
                config.json_config[config_path],
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            f.write("\n")


def step_train(rvc_root: Path, config, p: TrainingParams) -> None:
    _write_filelist(rvc_root, config, p)
    path_suffix_v2 = "" if p.version == "v1" else "_v2"
    if not p.pretrained_g and not p.pretrained_d:
        pretrained_G14, pretrained_D15 = resolve_pretrained_paths(p, path_suffix_v2)
    else:
        pretrained_G14, pretrained_D15 = p.pretrained_g, p.pretrained_d

    l = 1 if p.save_only_latest else 0
    c = 1 if p.cache_dataset_in_gpu else 0
    sw = 1 if p.save_weights_every_epoch else 0
    sr2 = p.sample_rate_label
    exp = p.experiment_name
    pg = "-pg %s" % pretrained_G14 if pretrained_G14 else ""
    pd = "-pd %s" % pretrained_D15 if pretrained_D15 else ""

    if p.gpu_devices_train:
        cmd = (
            '"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
            % (
                config.python_cmd,
                exp,
                sr2,
                1 if p.if_f0 else 0,
                p.batch_size,
                p.gpu_devices_train,
                p.total_epochs,
                p.save_every_epoch,
                pg,
                pd,
                l,
                c,
                sw,
                p.version,
            )
        )
    else:
        cmd = (
            '"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
            % (
                config.python_cmd,
                exp,
                sr2,
                1 if p.if_f0 else 0,
                p.batch_size,
                p.total_epochs,
                p.save_every_epoch,
                pg,
                pd,
                l,
                c,
                sw,
                p.version,
            )
        )
    code = _run(cmd, rvc_root)
    if code != 0:
        raise RuntimeError("train.py exited with %s" % code)


def step_train_index(rvc_root: Path, config, p: TrainingParams) -> Iterable[str]:
    import faiss

    exp_dir1 = p.experiment_name
    exp_dir = "logs/%s" % (exp_dir1)
    exp_path = rvc_root / exp_dir
    exp_path.mkdir(parents=True, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if p.version == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    fdir = rvc_root / feature_dir.replace("/", os.sep)
    if not fdir.is_dir() or not any(fdir.iterdir()):
        yield "请先进行特征提取!"
        return

    infos = []
    npys = []
    for name in sorted(os.listdir(fdir)):
        phone = np.load(str(fdir / name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    if big_npy.shape[0] > 2e5:
        infos.append("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
        yield "\n".join(infos)
        try:
            n_cpu = config.n_cpu or os.cpu_count() or 4
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except Exception:
            infos.append(traceback.format_exc())
            yield "\n".join(infos)
            return

    np.save(str(exp_path / "total_fea.npy"), big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append("%s,%s" % (big_npy.shape, n_ivf))
    yield "\n".join(infos)
    version19 = p.version
    index = faiss.index_factory(
        256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf
    )
    infos.append("training")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        str(
            exp_path
            / (
                "trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
                % (n_ivf, index_ivf.nprobe, exp_dir1, version19)
            )
        ),
    )
    infos.append("adding")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    added_name = "added_IVF%s_Flat_nprobe_%s_%s_%s.index" % (
        n_ivf,
        index_ivf.nprobe,
        exp_dir1,
        version19,
    )
    faiss.write_index(index, str(exp_path / added_name))
    infos.append("成功构建索引 %s" % added_name)
    outside = os.getenv("outside_index_root")
    if outside:
        try:
            dst = Path(outside) / ("%s_%s" % (exp_dir1, added_name))
            dst.parent.mkdir(parents=True, exist_ok=True)
            link = os.link if platform.system() == "Windows" else os.symlink
            try:
                link(str(exp_path / added_name), str(dst))
                infos.append("链接索引到外部-%s" % outside)
            except (OSError, NotImplementedError):
                import shutil

                shutil.copy2(str(exp_path / added_name), str(dst))
                infos.append("复制索引到外部-%s (copy)" % outside)
        except Exception as e:
            infos.append("链接/复制索引失败: %s" % e)
    yield "\n".join(infos)


def step_extract_small_weights(rvc_root: Path, p: TrainingParams) -> str:
    from infer.lib.train.process_ckpt import extract_small_model

    os.chdir(rvc_root)
    exp = p.experiment_name
    log_g = rvc_root / "logs" / exp
    ckpt = p.g_checkpoint_for_extract.strip()
    if not ckpt:
        cand = log_g / "G_2333333.pth"
        if cand.is_file():
            ckpt = str(cand)
        else:
            gs = sorted(
                log_g.glob("G_*.pth"), key=lambda x: x.stat().st_mtime, reverse=True
            )
            if not gs:
                raise FileNotFoundError("No G_*.pth in %s" % log_g)
            ckpt = str(gs[0])
    return extract_small_model(
        ckpt,
        p.infer_weight_name,
        p.sample_rate_label,
        int(p.if_f0),
        p.extract_info_str,
        p.version,
    )


def check_mute_template(rvc_root: Path) -> list[str]:
    p = rvc_root / "logs" / "mute"
    needed = [
        p / "0_gt_wavs",
        p / "3_feature256",
        p / "3_feature768",
        p / "2a_f0",
        p / "2b-f0nsf",
    ]
    missing = [str(x) for x in needed if not x.is_dir()]
    return missing


def run_all(rvc_root: Path, config, p: TrainingParams) -> None:
    miss = check_mute_template(rvc_root)
    if miss:
        logger.warning(
            "Thiếu thư mục mute (xem FAQ / bản RVC đầy đủ): %s",
            miss,
        )
    step_preprocess(rvc_root, config, p)
    step_extract_f0_and_features(rvc_root, config, p)
    step_train(rvc_root, config, p)
    if not p.skip_index:
        for line in step_train_index(rvc_root, config, p):
            print(line, flush=True)
    if p.extract_infer_pth:
        print(step_extract_small_weights(rvc_root, p), flush=True)
