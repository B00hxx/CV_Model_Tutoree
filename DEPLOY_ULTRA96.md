# Deploying the GTSRB CNN to the Ultra96-V2

This guide takes the trained `TrafficSignCNN` from `outputs/best_model.pt` to an **INT8 model running on the DPU of an Avnet Ultra96-V2** (Zynq UltraScale+ MPSoC **ZU3EG**), using **Vitis AI**. The architecture is described in [README §4](README.md#4-model-architecture).

Every step lists what to run, where to run it, and what you should see.

---

## Contents

0. [Overview](#0-overview)
1. [What you need](#1-what-you-need)
2. [Pick matching versions](#2-pick-matching-versions)
3. [Prepare the SD card and boot the board](#3-prepare-the-sd-card-and-boot-the-board)
4. [Read the DPU fingerprint, write `arch.json`](#4-read-the-dpu-fingerprint-write-archjson)
5. [Export the trained weights](#5-export-the-trained-weights)
6. [Start the Vitis AI container](#6-start-the-vitis-ai-container)
7. [Check the float deploy model](#7-check-the-float-deploy-model)
8. [(Optional) Inspect DPU compatibility](#8-optional-inspect-dpu-compatibility)
9. [Quantize: calibrate](#9-quantize-calibrate)
10. [Quantize: measure INT8 accuracy](#10-quantize-measure-int8-accuracy)
11. [Quantize: export the xmodel](#11-quantize-export-the-xmodel)
12. [Compile for the DPU](#12-compile-for-the-dpu)
13. [Check the compiled model](#13-check-the-compiled-model)
14. [Copy everything to the board](#14-copy-everything-to-the-board)
15. [Run inference on the board](#15-run-inference-on-the-board)
16. [Measure accuracy and speed on the board](#16-measure-accuracy-and-speed-on-the-board)
17. [Results log](#17-results-log)
18. [If INT8 accuracy drops](#18-if-int8-accuracy-drops)
19. [Troubleshooting](#19-troubleshooting)
20. [Checklist](#20-checklist)

---

## 0. Overview

```
 PC: training image (torch 2.14)            PC: Vitis AI 3.0 container (CPU)                       Ultra96-V2
 ───────────────────────────────            ─────────────────────────────────                      ──────────
 best_model.pt                               quantize.py --mode calib   (activation ranges)
   │ deploy/export_weights.py                quantize.py --mode test    (INT8 accuracy)
   ▼                                         quantize.py --mode test --deploy
 outputs/deploy/gtsrb_float.npz  ─────────▶    ▼
                                             TrafficSignCNN_int.xmodel   (quantized, target-independent)
                                               │ vai_c_xir  + arch.json  (fingerprint of *your* DPU)
                                               ▼
                                             gtsrb_cnn.xmodel            (DPU instructions)  ──scp──▶  gtsrb_dpu.py (VART)
                                                                                                         ▼
                                                                                                       predictions
```

**What runs where on the board**

| Work | Runs on |
|---|---|
| Load image, ROI crop, resize to 48×48, normalise, convert to int8 (NHWC) | ARM Cortex-A53 (Python/OpenCV) |
| All 6 convolutions (BatchNorm folded in), ReLU, max/avg pooling, both FC layers | **DPU** (FPGA fabric), one subgraph |
| Dequantise logits, softmax, top-k | ARM Cortex-A53 |

**Files in this repo used for deployment**

| File | Runs in | Purpose |
|---|---|---|
| `deploy/export_weights.py` | training Docker image | `best_model.pt` → `outputs/deploy/gtsrb_float.npz` |
| `deploy/quantize.py` | Vitis AI container | float check, calibration, INT8 test, xmodel export |
| `deploy/gtsrb_dpu.py` | Ultra96-V2 | inference with VART; accuracy and timing on a GTSRB CSV |
| `deploy/arch.json` | Vitis AI container | DPU fingerprint for the compiler (you create it in step 4) |

**Already checked on the PC.** The deploy model (`AvgPool2d(2)` head, weights from the `.npz`) scores **99.43%** on the test set, the same as the original. The board's OpenCV preprocessing with int8 input rounding (scale 2⁵) also scores **99.43%** with the float model. So any accuracy you lose on the board comes from weight and activation quantization, not from the pipeline.

---

## 1. What you need

**Hardware**
- Avnet Ultra96-V2 board, with its 12 V / 2 A+ power supply
- microSD card, 16 GB or more
- A way to reach the board's console and network:
  - the Ultra96 **JTAG/UART adapter** (micro-USB serial console at 115200 baud), *or* a USB keyboard and DisplayPort monitor
  - Wi-Fi (the board has no Ethernet port), or a USB-to-Ethernet adapter on one of the USB-A ports

**Software on the PC**
- This repo, with a trained `outputs/best_model.pt` and the dataset in `./archive`
- Docker, the same setup used for training. The Vitis AI CPU container works under WSL2.
- About 15 GB of free disk space for the Vitis AI image
- An SD-card writer tool: [balenaEtcher](https://etcher.balena.io/) on Windows, or `dd` on Linux

---

## 2. Pick matching versions

> **The rule:** the Vitis AI version you quantize and compile with must match the **VART runtime and DPU IP in the board image**. Most "it compiled but won't run" problems come from a version mismatch.

The Ultra96-V2 isn't an official AMD/Xilinx evaluation board. Board images with a DPU come from **Avnet**, which publishes Vitis platforms and pre-built Vitis AI SD images for `u96v2_sbc`.

| Choice | Recommended | Notes |
|---|---|---|
| Vitis AI | **3.0** | Last release with full Zynq UltraScale+ (DPUCZDX8G) support. 3.5 focuses on Versal and only carries the 3.0 MPSoC IP forward. 2.5 also works, but then use 2.5 everywhere. |
| Board image | Avnet `u96v2_sbc` image built for **Vitis AI 3.0** (tools 2022.2) | Includes the DPU in the bitstream, VART, `xdputil`, Python 3, NumPy and OpenCV. |
| Docker image | `xilinx/vitis-ai-pytorch-cpu`, a **3.0.x** tag (e.g. `ubuntu2004-3.0.0.106`) | Don't use `:latest`, which may pull a newer release. Check the tags on Docker Hub. |
| DPU | Whatever the image contains (often B2304 or B1600 on the ZU3EG) | You never hard-code this. Step 4 reads it from the board. |

Write down the versions you end up with. They go in the [results log](#17-results-log).

---

## 3. Prepare the SD card and boot the board

1. **Download** the Avnet `u96v2_sbc` Vitis AI 3.0 SD image (`.img` or `.wic`, often zipped). Avnet links it from its Vitis AI tutorials and from the Ultra96-V2 page on avnet.com and hackster.io. If you'd rather build it yourself, see the [build-your-own note](#building-your-own-image) at the end of this step.
2. **Flash** it to the microSD card:
   - Windows: balenaEtcher → *Flash from file* → choose the card → *Flash*.
   - Linux (**double-check `/dev/sdX`**, because `dd` overwrites whatever you point it at):
     ```bash
     lsblk                                    # identify the SD card
     unzip -p u96v2_sbc_*.zip | sudo dd of=/dev/sdX bs=4M status=progress conv=fsync
     ```
3. **Set the boot mode switch (SW3) to SD-card boot.** Check the position in the Avnet Ultra96-V2 hardware user guide.
4. Insert the card, attach the UART adapter (or keyboard and monitor), connect power, and press the **power button (SW4)**.
5. **Open the console.** Use `screen /dev/ttyUSB1 115200` on Linux, or PuTTY on the adapter's COM port at 115200 on Windows. Log in; on PetaLinux images the user is usually `root`, and the image's README gives the password.
6. **Get on the network.** Wi-Fi example (the exact tool depends on the image; some use `connmanctl` instead):
   ```bash
   wpa_passphrase "MySSID" "MyPassword" > /etc/wpa_supplicant.conf
   wpa_supplicant -B -i wlan0 -c /etc/wpa_supplicant.conf
   udhcpc -i wlan0            # or: dhclient wlan0
   ip addr show wlan0         # note the board's IP, used as <board-ip> below
   ```
7. From the PC, check that SSH works: `ssh root@<board-ip>`.

### Building your own image

If no pre-built image matches your Vitis AI version, build one with Avnet's 2022.2 repositories (`Avnet/bdf`, `Avnet/hdl`, `Avnet/petalinux`, `Avnet/vitis` on GitHub) and the DPU TRD for Vitis AI 3.0. That means Vivado/Vitis/PetaLinux 2022.2 on a Linux machine and a few hours of build time. Everything after step 4 stays the same.

---

## 4. Read the DPU fingerprint, write `arch.json`

The compiler has to know exactly which DPU configuration is in your bitstream. It identifies it by a **fingerprint**.

On the **board**:
```bash
xdputil query
```
Expected output (abridged; the values depend on your image):
```json
{
    "DPU IP Spec": { "DPU Core Count": 1, "IP version": "v4.1.0", ... },
    "kernels": [{
        "DPU Arch": "DPUCZDX8G_ISA1_B2304",
        "DPU Frequency (MHz)": 300,
        "fingerprint": "0x101000016010406",
        ...
    }]
}
```

On the **PC**, create `deploy/arch.json` using **your** fingerprint (the one above is only an example):
```json
{
    "fingerprint": "0x101000016010406"
}
```

Also write down the `DPU Arch` (for example B2304), the frequency and the core count. You'll want them when comparing speed.

> If `xdputil` doesn't exist, or reports no DPU, the image has no DPU in its bitstream, or the DPU wasn't loaded at boot. Fix that before going further (see [troubleshooting](#19-troubleshooting)).

---

## 5. Export the trained weights

Vitis AI 3.0 ships an older PyTorch (1.x) and Python 3.8. Instead of depending on a checkpoint written by torch 2.14, export the weights as plain NumPy arrays.

On the **PC**, in this repo, using the training image:
```bash
docker compose run --rm -v "$PWD/deploy:/app/deploy" --entrypoint python train deploy/export_weights.py
```
Expected:
```
epoch 21, val acc 99.42% -> outputs/deploy/gtsrb_float.npz
```
The file is about 2.4 MB. `outputs/deploy/` is git-ignored like the other run outputs.

---

## 6. Start the Vitis AI container

On the **PC**:
```bash
docker pull xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106      # use the 3.0.x tag you chose
git clone -b v3.0 https://github.com/Xilinx/Vitis-AI.git ~/Vitis-AI  # only needed for docker_run.sh

cd ~/projects/tut                                                    # this repo
~/Vitis-AI/docker_run.sh xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106
```
`docker_run.sh` mounts the **current directory** as `/workspace`, so start it from the repo root. Accept the license prompt. Then, inside the container:
```bash
conda activate vitis-ai-pytorch
cd /workspace
python -c "import torch, pytorch_nndct; print(torch.__version__)"
```
Every command from step 7 to step 13 runs **inside this container**, from `/workspace`. The dataset is available at `archive/`, which is the default `--data-dir` of `deploy/quantize.py`.

> The CPU image is enough. Calibration covers 1,000 small images, and the INT8 test pass runs on CPU in minutes. The GPU image only helps with QAT ([step 18](#18-if-int8-accuracy-drops)).

---

## 7. Check the float deploy model

`deploy/quantize.py` builds the **deploy version** of the network. It's the same as the trained model, except `AdaptiveAvgPool2d((3,3))` is replaced by `AvgPool2d(2)`. On the 6×6 map that block 3 produces at 48×48 input, the two are mathematically identical (README §4.5). The plain 2×2 pool maps cleanly onto the DPU.

```bash
python deploy/quantize.py --mode float
```
Expected:
```
float deploy model, test acc: 99.43%
```
This needs no Vitis AI. It's a sanity check that the weights loaded and the preprocessing matches training. If it isn't 99.43%, stop and fix that first, because quantization will only make things worse.

> `deploy/quantize.py` imports only `src/model.py`. `src/data.py` uses Python 3.9+ type hints (`list[dict]`), which fail on the container's Python 3.8, so the script has its own copy of the eval preprocessing.

---

## 8. (Optional) Inspect DPU compatibility

The Vitis AI *inspector* reports, before quantizing, which operators would run on the DPU and which on the CPU. Pass the fingerprint from step 4:
```bash
python - <<'EOF'
import sys, torch
sys.path.insert(0, "deploy")
from quantize import build_deploy_model
from pytorch_nndct.apis import Inspector

model = build_deploy_model("outputs/deploy/gtsrb_float.npz")
Inspector("0x101000016010406").inspect(           # your fingerprint (or DPU name)
    model, (torch.randn(1, 3, 48, 48),), device=torch.device("cpu"),
    output_dir="outputs/deploy/inspect", image_format="png")
EOF
```
Expected: every node is assigned to the **DPU**. Read `outputs/deploy/inspect/inspect_*.txt` for the details.

---

## 9. Quantize: calibrate

Post-training quantization (PTQ) runs about 1,000 images through the network to measure the range of every activation. It then chooses a power-of-two scale (`fix_point`) per tensor so the values fit in int8.

```bash
python deploy/quantize.py --mode calib
```
- Calibration images are a fixed random sample (seed 0) of **`Train.csv`**. The test set isn't used, so the INT8 test score in the next step stays honest.
- Use `--calib-images 2000` (or more) if accuracy drops more than expected.

Output in `outputs/deploy/quantize_result/`:
- `quant_info.json`: the chosen bit widths and fix points
- `TrafficSignCNN.py`: the quantizer's traced version of the model

---

## 10. Quantize: measure INT8 accuracy

```bash
python deploy/quantize.py --mode test
```
This runs the full 12,630-image test set through the **simulated INT8 model** on the CPU, so it takes a few minutes. Expected: within about 0.5 percentage points of 99.43%. A model this size with BatchNorm and ReLU usually quantizes well, but **write down the number you actually get**. It's your target for the board.

If the drop is larger than 0.5 pt, see [step 18](#18-if-int8-accuracy-drops) before moving on.

---

## 11. Quantize: export the xmodel

```bash
python deploy/quantize.py --mode test --deploy
```
xmodel export needs a forward pass with batch size 1 in test mode, so this runs a single image and then exports. Expected:
```
exported outputs/deploy/quantize_result/TrafficSignCNN_int.xmodel
```
This xmodel is quantized but **not yet tied to a DPU**. It works for any DPU size.

---

## 12. Compile for the DPU

```bash
vai_c_xir \
    -x outputs/deploy/quantize_result/TrafficSignCNN_int.xmodel \
    -a deploy/arch.json \
    -o outputs/deploy/compiled \
    -n gtsrb_cnn
```
In the compiler log, look for:
```
[UNILOG][INFO] Total device subgraph number 3, DPU subgraph number 1
[UNILOG][INFO] Compile done.
```
**"DPU subgraph number 1" is the line that matters.** It means the whole network runs on the DPU in one piece. (The other subgraphs are the input/output placeholders.) If it reports more than one DPU subgraph, or any CPU subgraph with real ops in it, an operator didn't map to the DPU. See [troubleshooting](#19-troubleshooting).

Output: `outputs/deploy/compiled/gtsrb_cnn.xmodel` (plus `md5sum.txt` and `meta.json`).

---

## 13. Check the compiled model

Still in the container, print the DPU subgraph's input and output tensors:
```bash
python - <<'EOF'
import xir
g = xir.Graph.deserialize("outputs/deploy/compiled/gtsrb_cnn.xmodel")
dpu = [s for s in g.get_root_subgraph().toposort_child_subgraph()
       if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]
print("DPU subgraphs:", len(dpu))
for t in list(dpu[0].get_input_tensors()) + list(dpu[0].get_output_tensors()):
    print(f"{t.name:60s} dims={t.dims} dtype={t.dtype} fix_point={t.get_attr('fix_point')}")
EOF
```
What to expect:
- **1** DPU subgraph.
- Input dims `[1, 48, 48, 3]`. That's **NHWC with batch 1**. The DPU wants channels last, which `gtsrb_dpu.py` already produces.
- Input **`fix_point` = 5**, meaning int8 value = round(x · 2⁵). The normalised inputs range from about −1.25 to +2.6, and 2⁵ is the scale that covers that range. With 2⁶, values above 1.98 would clip and cost about 0.1 pt. The board script reads `fix_point` from the model, so it adapts either way, but 5 is the expected value.
- Output dims `[1, 43]` (or `[1, 1, 1, 43]`): the logits.

Optional: `xir png outputs/deploy/compiled/gtsrb_cnn.xmodel graph.png` draws the graph.

You can now leave the container.

---

## 14. Copy everything to the board

From the **PC** (repo root):
```bash
ssh root@<board-ip> "mkdir -p ~/gtsrb"
scp outputs/deploy/compiled/gtsrb_cnn.xmodel deploy/gtsrb_dpu.py src/classes.py root@<board-ip>:~/gtsrb/

# Test set for the accuracy run (~150 MB, 12,630 PNGs)
scp -r archive/Test.csv archive/Test root@<board-ip>:~/gtsrb/
```
If `scp -r` of 12k files is slow over Wi-Fi, pack them first:
```bash
tar -C archive -czf /tmp/gtsrb_test.tgz Test.csv Test
scp /tmp/gtsrb_test.tgz root@<board-ip>:~/gtsrb/ && ssh root@<board-ip> "cd ~/gtsrb && tar xzf gtsrb_test.tgz"
```

Board folder afterwards:
```
~/gtsrb/
├── gtsrb_cnn.xmodel
├── gtsrb_dpu.py
├── classes.py
├── Test.csv
└── Test/00000.png ...
```

---

## 15. Run inference on the board

On the **board**:
```bash
cd ~/gtsrb
python3 gtsrb_dpu.py gtsrb_cnn.xmodel Test/00000.png Test/00001.png
```
Expected, similar to `predict.py` on the PC (the two files are class 16 and class 1):
```
Test/00000.png
   9x.xx%  [16] Vehicles >3.5t prohibited
   ...
Test/00001.png
   9x.xx%  [ 1] Speed limit 30
   ...
```
Image paths passed this way are used **as they are, with no ROI crop**, so give it tightly cropped signs. The GTSRB test images are already close crops, which is why the example works.

What `gtsrb_dpu.py` does for each image:
1. `cv2.imread` (BGR), then crop to the ROI if one is known, then convert BGR → **RGB**.
2. Resize to 48×48: `INTER_AREA` when shrinking, `INTER_LINEAR` when enlarging. On the test set this matches PIL's resize used in training (99.43% vs 99.32% with `INTER_LINEAR` everywhere).
3. Normalise with the GTSRB mean/std, multiply by 2^`fix_point`, round, clip to [−128, 127] → int8 NHWC.
4. `runner.execute_async` and `runner.wait` on the DPU.
5. Multiply the int8 logits by 2^−`fix_point(output)`, then softmax on the CPU.

---

## 16. Measure accuracy and speed on the board

```bash
python3 gtsrb_dpu.py gtsrb_cnn.xmodel --csv Test.csv --limit 2000   # quick run
python3 gtsrb_dpu.py gtsrb_cnn.xmodel --csv Test.csv                # all 12,630 images
```
Output format:
```
accuracy 12xxx/12630 = 99.xx%
per image: load+preprocess x.xx ms, DPU+softmax x.xx ms (~xxxx FPS single-thread, DPU only)
```
- **Accuracy** should be within about 0.1 pt of the INT8 number from step 10. A bigger gap means the board preprocessing differs from `quantize.py` (see troubleshooting).
- **Speed.** The model needs 0.174 GOP per image. A B2304 DPU at 300 MHz peaks at 2304 × 0.3 G ≈ 690 GOPS, so the hard upper bound is about 4,000 images/s. Small models use the DPU far below peak, and each single-image call has a fixed overhead, so the real number will be noticeably lower. Measure it, don't assume it.
- On the A53 cores, the Python preprocessing (PNG decode and resize) will likely cost more time than the DPU. For a real camera pipeline, move preprocessing to C++, or run several threads, each with its own runner.
- For a raw DPU throughput number without Python, try `xdputil benchmark gtsrb_cnn.xmodel -i -1 4` (4 threads; check `xdputil --help` on your image).

---

## 17. Results log

Fill this in as you go:

| Stage | Where | Test accuracy | Notes |
|---|---|---|---|
| Float, original model | PC, `evaluate.py` | **99.43%** | 72 errors |
| Float, deploy model (`AvgPool2d(2)`, `.npz`) | PC, `quantize.py --mode float` | **99.43%** | verified |
| Float + board-style OpenCV preprocessing + int8 input | PC | **99.43%** | verified |
| INT8, simulated | Vitis AI, `quantize.py --mode test` | | |
| INT8, on DPU | Ultra96-V2, `gtsrb_dpu.py --csv` | | |

| Setting | Value |
|---|---|
| Vitis AI version (Docker tag) | |
| Board image (name/date) | |
| DPU arch / fingerprint / frequency | |
| Calibration images | 1000 |
| Board latency, DPU + softmax (ms) | |
| Board latency, load + preprocess (ms) | |

---

## 18. If INT8 accuracy drops

Try these in order of effort:

1. **More calibration data.** Use `--calib-images 3000`, then rerun steps 9–12. Cheap, and often enough.
2. **Fast finetune (AdaQuant).** `vai_q_pytorch` can adjust weights layer by layer during calibration, using a small unlabeled set. Call `quantizer.fast_finetune(evaluate_fn, (qmodel, loader))` in calib mode and `quantizer.load_ft_param()` in test mode. See the [vai_q_pytorch docs](https://docs.amd.com/r/3.0-English/ug1414-vitis-ai/vai_q_pytorch-Fast-Finetuning) and the `resnet18_quant.py` example in the Vitis AI repo.
3. **Quantization-aware training (QAT).** Retrain for a few epochs with fake-quant nodes using `pytorch_nndct`'s `QatProcessor`, starting from the trained weights. This needs the training loop ported to the container's PyTorch version. It's the most work, and almost never needed for a network this size.

Look at *which* classes drop, too. With 48×48 inputs, the speed-limit digits (60 vs 80) are the most fragile part, as in README §6.3.

---

## 19. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `TypeError: 'type' object is not subscriptable` in the Vitis AI container | Something imported `src/data.py` or `src/utils.py` (Python 3.9+ type hints on Python 3.8). Use only `deploy/quantize.py`, which imports just `src/model.py`. |
| `float deploy model` accuracy isn't 99.43% | Wrong `--weights` file, or the dataset isn't at `archive/` (pass `--data-dir`). |
| DataLoader "bus error" / shared-memory crash in the container | The Vitis AI container has a small `/dev/shm`. Add `--num-workers 0` to the `quantize.py` commands. |
| `ModuleNotFoundError: pytorch_nndct` | `conda activate vitis-ai-pytorch` wasn't run, or this is the wrong Docker image. |
| Compiler reports more than one DPU subgraph, or CPU ops | An op didn't map to the DPU. The usual culprit is exporting with `AdaptiveAvgPool2d`, or adding an op such as softmax inside the model. Run the [inspector](#8-optional-inspect-dpu-compatibility). |
| On the board: fingerprint / "target mismatch" error when creating the runner | `arch.json` doesn't match the DPU in the bitstream. Re-read `xdputil query` and recompile (step 12). |
| On the board: error loading the xmodel (version / unknown attribute) | xmodel compiled with a newer Vitis AI than the board's VART. Use the same version on both sides (step 2). |
| `xdputil query` shows no DPU, or `/dev/dri` / `zocl` errors | Image without a DPU, or the DPU overlay/xclbin wasn't loaded. Check the image README; some images need a `xmutil loadapp ...` or a `dpu.xclbin` path in `/etc/vart.conf`. |
| Board accuracy much lower than the INT8 simulation | Preprocessing mismatch. Check RGB vs BGR, ROI crop, the mean/std values, NHWC layout, and that you multiply by 2^`fix_point`. `gtsrb_dpu.py` handles all of these; check any changes you made. |
| Predictions on your own photos are bad | Training used tight ROI crops, so crop the sign before classifying. |
| `ImportError: vart` / `xir` on the board | Not a Vitis AI image, or the Python bindings weren't installed. Use the Avnet Vitis AI image. |

---

## 20. Checklist

- [ ] Versions chosen: Vitis AI 3.0 container tag, matching Avnet `u96v2_sbc` image
- [ ] SD card flashed, board boots from SD, SSH works
- [ ] `xdputil query` shows the DPU → `deploy/arch.json` written with **your** fingerprint
- [ ] `export_weights.py` → `outputs/deploy/gtsrb_float.npz`
- [ ] `quantize.py --mode float` → 99.43%
- [ ] `quantize.py --mode calib` → `quant_info.json`
- [ ] `quantize.py --mode test` → INT8 accuracy written down
- [ ] `quantize.py --mode test --deploy` → `TrafficSignCNN_int.xmodel`
- [ ] `vai_c_xir` → `gtsrb_cnn.xmodel`, **DPU subgraph number 1**
- [ ] Input tensor `[1,48,48,3]`, `fix_point` 5
- [ ] Files copied to the board, single-image prediction works
- [ ] Full test-set accuracy on the board close to the INT8 simulation; latency recorded
