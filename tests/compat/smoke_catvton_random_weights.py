"""Compatibility smoke test: CatVTON code + OUR engine wrapper with RANDOM weights (CPU, no downloads
from Hugging Face).

Purpose: prove the code paths run on a given Python/torch/diffusers stack before spending a Colab
session on it: detectron2/DensePose -> SCHP (LIP+ATR) -> mask fusion -> CatVTON diffusion loop with
spatially concatenated latents -> our progress counting, mask blur, repaint and timings.
The output images are meaningless by design (random weights). This is NOT a quality test.

Usage (needs a clone of CatVTON and the two DensePose yaml configs from detectron2/projects/DensePose/configs):
    python tests/compat/smoke_catvton_random_weights.py <path/to/CatVTON> <dir/with/densepose/yamls>
"""
import os, sys, pickle, shutil, time, tempfile
import numpy as np, torch
from collections import OrderedDict
from PIL import Image
REPO = os.path.abspath(sys.argv[1]); CFG = sys.argv[2]
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))  # project root
torch.manual_seed(0)
work = tempfile.mkdtemp()
# ---- fake DensePose checkpoint ----
dp_dir = os.path.join(work, "DensePose"); os.makedirs(dp_dir)
for f in os.listdir(CFG): shutil.copy(os.path.join(CFG, f), dp_dir)
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from densepose import add_densepose_config
cfg = get_cfg(); add_densepose_config(cfg); cfg.merge_from_file(os.path.join(dp_dir, "densepose_rcnn_R_50_FPN_s1x.yaml")); cfg.MODEL.DEVICE = "cpu"
m = build_model(cfg)
with open(os.path.join(dp_dir, "model_final_162be9.pkl"), "wb") as fh:
    pickle.dump({"model": {k: v.numpy() for k, v in m.state_dict().items()}, "__author__": "smoke"}, fh)
del m
# ---- fake SCHP checkpoints ----
from model.SCHP import networks
schp_dir = os.path.join(work, "SCHP"); os.makedirs(schp_dir)
for name, ncls in [("exp-schp-201908301523-atr.pth", 18), ("exp-schp-201908261155-lip.pth", 20)]:
    net = networks.init_model("resnet101", num_classes=ncls, pretrained=None)
    torch.save({"state_dict": OrderedDict(("module." + k, v) for k, v in net.state_dict().items())}, os.path.join(schp_dir, name))
    del net
from detectron2.config.defaults import _C as D2DEFAULTS
D2DEFAULTS.MODEL.DEVICE = "cpu"  # sandbox has no GPU
from model.cloth_masker import AutoMasker
t = time.time()
masker = AutoMasker(densepose_ckpt=dp_dir, schp_ckpt=schp_dir, device="cpu")
print(f"AutoMasker constructed in {time.time()-t:.1f}s")
person = Image.fromarray((np.random.rand(1024, 768, 3) * 255).astype(np.uint8))
t = time.time()
try:
    out = masker(person, "upper")
    print("AutoMasker call OK:", {k: (v.size if hasattr(v, "size") else type(v)) for k, v in out.items()}, f"{time.time()-t:.1f}s")
except Exception as e:
    import traceback; traceback.print_exc(); print("AutoMasker call FAILED (random weights may give no detections):", repr(e))
# ---- tiny diffusion pipeline ----
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from model.pipeline import CatVTONPipeline
from model.utils import init_adapter, get_trainable_module
from model.attn_processor import SkipAttnProcessor
pipe = CatVTONPipeline.__new__(CatVTONPipeline)
pipe.device = "cpu"; pipe.weight_dtype = torch.float32; pipe.skip_safety_check = True
pipe.noise_scheduler = DDIMScheduler()
pipe.vae = AutoencoderKL(block_out_channels=(32, 64), down_block_types=("DownEncoderBlock2D",)*2, up_block_types=("UpDecoderBlock2D",)*2, latent_channels=4, norm_num_groups=32)
pipe.unet = UNet2DConditionModel(sample_size=16, in_channels=9, out_channels=4, block_out_channels=(32, 64), layers_per_block=1,
    down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"), up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=32, norm_num_groups=32)
init_adapter(pipe.unet, cross_attn_cls=SkipAttnProcessor)
pipe.attn_modules = get_trainable_module(pipe.unet, "attention")
print("trainable attention params (tiny):", sum(p.numel() for p in pipe.attn_modules.parameters()))
from utils import resize_and_crop, resize_and_padding
W, H = 96, 128
p = resize_and_crop(person, (W, H)); c = resize_and_padding(person, (W, H))
mask = Image.new("L", (W, H), 0); mask.paste(255, (20, 30, 76, 100))
res = pipe(image=p, condition_image=c, mask=mask, num_inference_steps=3, guidance_scale=2.5, height=H, width=W, generator=torch.Generator().manual_seed(0))
print("pipeline OK, result:", res[0].size, res[0].mode)

# ---- OUR engine wrapper on top of the same random-weight components ----
from tryon_server.catvton_engine import CatVTONEngine, EngineConfig, TryOnParams
from diffusers.image_processor import VaeImageProcessor
import utils as catvton_utils
engine = CatVTONEngine.__new__(CatVTONEngine)
engine.config = EngineConfig(catvton_repo=REPO, width=W, height=H, device="cpu")
engine._lock = __import__("threading").Lock()
engine.dtype = torch.float32
engine.pipeline, engine.automasker = pipe, masker
engine.mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
engine._resize_and_crop, engine._resize_and_padding, engine._repaint_result = (
    catvton_utils.resize_and_crop, catvton_utils.resize_and_padding, catvton_utils.repaint_result)
events = []
user_mask = Image.new("L", (W, H), 0); user_mask.paste(255, (20, 30, 76, 100))
out = engine.generate(person, person, TryOnParams(steps=5, seed=7), user_mask=user_mask,
                      progress=lambda st, i, n: events.append((st, i, n)))
assert out.result.size == (W, H) and out.seed_used == 7
assert [e for e in events if e[0] == "diffusion"] == [("diffusion", i, 5) for i in range(1, 6)], events
assert events[-1] == ("done", 5, 5)
# repaint: pixels far outside the (blurred) mask must equal the input exactly
assert np.array_equal(np.array(out.result)[0:10, 0:10], np.array(out.person_input)[0:10, 0:10])
print("engine.generate OK:", out.timings, "progress events:", len(events))
