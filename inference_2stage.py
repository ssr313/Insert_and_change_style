# Copyright 2024 InstantX Team. All rights reserved.

import os
import cv2
import numpy as np
from PIL import Image

import diffusers
from diffusers.utils import load_image
from diffusers import DDIMScheduler, ControlNetModel
from transformers import CLIPVisionModelWithProjection
from transformers import AutoProcessor, Blip2ForConditionalGeneration

import torch
import torchvision
from torchvision import transforms

from src.eunms import Model_Type, Scheduler_Type
from src.utils.enums_utils import get_pipes
from src.config import RunConfig

from inversion import run as invert
from CSD_Score.model import CSD_CLIP, convert_state_dict
from pipeline_controlnet_sd_xl_img2img import StableDiffusionXLControlNetImg2ImgPipeline
import os
import math
import gradio as gr
import numpy as np
import torch
import safetensors.torch as sf
import db_examples

from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from briarmbg import BriaRMBG
from enum import Enum
from torch.hub import download_url_to_file


# 'stablediffusionapi/realistic-vision-v51'
# 'runwayml/stable-diffusion-v1-5'
sd15_name = 'stablediffusionapi/realistic-vision-v51'
tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")
rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

# Change UNet

with torch.no_grad():
    new_conv_in = torch.nn.Conv2d(12, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
    new_conv_in.weight.zero_()
    new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
    new_conv_in.bias = unet.conv_in.bias
    unet.conv_in = new_conv_in

unet_original_forward = unet.forward


def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
    c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
    c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
    new_sample = torch.cat([sample, c_concat], dim=1)
    kwargs['cross_attention_kwargs'] = {}
    return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)


unet.forward = hooked_unet_forward

# Load

model_path = './models/iclight_sd15_fbc.safetensors'

if not os.path.exists(model_path):
    download_url_to_file(url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fbc.safetensors', dst=model_path)

sd_offset = sf.load_file(model_path)
sd_origin = unet.state_dict()
keys = sd_origin.keys()
sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
unet.load_state_dict(sd_merged, strict=True)
del sd_offset, sd_origin, sd_merged, keys

# Device

device = torch.device('cuda')
text_encoder = text_encoder.to(device=device, dtype=torch.float16)
vae = vae.to(device=device, dtype=torch.bfloat16)
unet = unet.to(device=device, dtype=torch.float16)
rmbg = rmbg.to(device=device, dtype=torch.float32)

# SDP

unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

# Samplers

ddim_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

euler_a_scheduler = EulerAncestralDiscreteScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    steps_offset=1
)

dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=True,
    steps_offset=1
)

# Pipelines

t2i_pipe = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)

i2i_pipe = StableDiffusionImg2ImgPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)


@torch.inference_mode()
def encode_prompt_inner(txt: str):
    max_length = tokenizer.model_max_length
    chunk_length = tokenizer.model_max_length - 2
    id_start = tokenizer.bos_token_id
    id_end = tokenizer.eos_token_id
    id_pad = id_end

    def pad(x, p, i):
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = [[id_start] + tokens[i: i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
    chunks = [pad(ck, id_pad, max_length) for ck in chunks]

    token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
    conds = text_encoder(token_ids).last_hidden_state

    return conds


@torch.inference_mode()
def encode_prompt_pair(positive_prompt, negative_prompt):
    c = encode_prompt_inner(positive_prompt)
    uc = encode_prompt_inner(negative_prompt)

    c_len = float(len(c))
    uc_len = float(len(uc))
    max_count = max(c_len, uc_len)
    c_repeat = int(math.ceil(max_count / c_len))
    uc_repeat = int(math.ceil(max_count / uc_len))
    max_chunk = max(len(c), len(uc))

    c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
    uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

    c = torch.cat([p[None, ...] for p in c], dim=1)
    uc = torch.cat([p[None, ...] for p in uc], dim=1)

    return c, uc


@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)

        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h


def resize_and_center_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    
    # 计算缩放比例，取最小值以确保图像完全显示
    scale_factor = min(target_width / original_width, target_height / original_height)
    
    # 计算新的尺寸
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    # 调整图像大小
    resized_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
    
    # 创建目标尺寸的空白图像
    result_image = Image.new('RGB', (target_width, target_height), (127, 127, 127))
    
    # 计算粘贴位置（居中）
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    
    # 粘贴调整后的图像
    result_image.paste(resized_image, (paste_x, paste_y))
    
    # 转换为numpy数组并返回
    return np.array(result_image)


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


@torch.inference_mode()
def run_rmbg(img, sigma=0.0):
    H, W, C = img.shape
    assert C == 3
    k = (256.0 / float(H * W)) ** 0.5
    feed = resize_without_crop(img, int(64 * round(W * k)), int(64 * round(H * k)))
    feed = numpy2pytorch([feed]).to(device=device, dtype=torch.float32)
    alpha = rmbg(feed)[0][0]
    alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
    alpha = alpha.movedim(1, -1)[0]
    alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
    result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
    return result.clip(0, 255).astype(np.uint8), alpha


@torch.inference_mode()
def process(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source):
    bg_source = BGSource(bg_source)

    if bg_source == BGSource.UPLOAD:
        pass
    elif bg_source == BGSource.UPLOAD_FLIP:
        input_bg = np.fliplr(input_bg)
    elif bg_source == BGSource.GREY:
        input_bg = np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8) + 64
    elif bg_source == BGSource.LEFT:
        gradient = np.linspace(224, 32, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.RIGHT:
        gradient = np.linspace(32, 224, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.TOP:
        gradient = np.linspace(224, 32, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == BGSource.BOTTOM:
        gradient = np.linspace(32, 224, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    else:
        raise 'Wrong background source!'

    rng = torch.Generator(device=device).manual_seed(seed)

    fg = resize_and_center_crop(input_fg, image_width, image_height)
    bg = resize_and_center_crop(input_bg, image_width, image_height)
    concat_conds = numpy2pytorch([fg, bg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
    concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

    conds, unconds = encode_prompt_pair(positive_prompt=prompt + ', ' + a_prompt, negative_prompt=n_prompt)

    latents = t2i_pipe(
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        width=image_width,
        height=image_height,
        num_inference_steps=steps,
        num_images_per_prompt=num_samples,
        generator=rng,
        output_type='latent',
        guidance_scale=cfg,
        cross_attention_kwargs={'concat_conds': concat_conds},
    ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels)
    pixels = [resize_without_crop(
        image=p,
        target_width=int(round(image_width * highres_scale / 64.0) * 64),
        target_height=int(round(image_height * highres_scale / 64.0) * 64))
    for p in pixels]

    pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
    latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
    latents = latents.to(device=unet.device, dtype=unet.dtype)

    image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8
    fg = resize_and_center_crop(input_fg, image_width, image_height)
    bg = resize_and_center_crop(input_bg, image_width, image_height)
    concat_conds = numpy2pytorch([fg, bg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
    concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

    latents = i2i_pipe(
        image=latents,
        strength=highres_denoise,
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        width=image_width,
        height=image_height,
        num_inference_steps=int(round(steps / highres_denoise)),
        num_images_per_prompt=num_samples,
        generator=rng,
        output_type='latent',
        guidance_scale=cfg,
        cross_attention_kwargs={'concat_conds': concat_conds},
    ).images.to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels, quant=False)

    return pixels, [fg, bg]


@torch.inference_mode()
def process_relight(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source):
    input_fg, matting = run_rmbg(input_fg)
    results, extra_images = process(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source)
    results = [(x * 255.0).clip(0, 255).astype(np.uint8) for x in results]
    return results + extra_images


@torch.inference_mode()
def process_normal(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source):
    input_fg, matting = run_rmbg(input_fg, sigma=16)

    print('left ...')
    left = process(input_fg, input_bg, prompt, image_width, image_height, 1, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, BGSource.LEFT.value)[0][0]

    print('right ...')
    right = process(input_fg, input_bg, prompt, image_width, image_height, 1, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, BGSource.RIGHT.value)[0][0]

    print('bottom ...')
    bottom = process(input_fg, input_bg, prompt, image_width, image_height, 1, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, BGSource.BOTTOM.value)[0][0]

    print('top ...')
    top = process(input_fg, input_bg, prompt, image_width, image_height, 1, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, BGSource.TOP.value)[0][0]

    inner_results = [left * 2.0 - 1.0, right * 2.0 - 1.0, bottom * 2.0 - 1.0, top * 2.0 - 1.0]

    ambient = (left + right + bottom + top) / 4.0
    h, w, _ = ambient.shape
    matting = resize_and_center_crop((matting[..., 0] * 255.0).clip(0, 255).astype(np.uint8), w, h).astype(np.float32)[..., None] / 255.0

    def safa_divide(a, b):
        e = 1e-5
        return ((a + e) / (b + e)) - 1.0

    left = safa_divide(left, ambient)
    right = safa_divide(right, ambient)
    bottom = safa_divide(bottom, ambient)
    top = safa_divide(top, ambient)

    u = (right - left) * 0.5
    v = (top - bottom) * 0.5

    sigma = 10.0
    u = np.mean(u, axis=2)
    v = np.mean(v, axis=2)
    h = (1.0 - u ** 2.0 - v ** 2.0).clip(0, 1e5) ** (0.5 * sigma)
    z = np.zeros_like(h)

    normal = np.stack([u, v, h], axis=2)
    normal /= np.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5
    normal = normal * matting + np.stack([z, z, 1 - z], axis=2) * (1 - matting)

    results = [normal, left, right, bottom, top] + inner_results
    results = [(x * 127.5 + 127.5).clip(0, 255).astype(np.uint8) for x in results]
    return results


quick_prompts = [
    'beautiful woman',
    'handsome man',
    'beautiful woman, cinematic lighting',
    'handsome man, cinematic lighting',
    'beautiful woman, natural lighting',
    'handsome man, natural lighting',
    'beautiful woman, neo punk lighting, cyberpunk',
    'handsome man, neo punk lighting, cyberpunk',
]
quick_prompts = [[x] for x in quick_prompts]


class BGSource(Enum):
    UPLOAD = "Use Background Image"
    UPLOAD_FLIP = "Use Flipped Background Image"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"
    GREY = "Ambient"


def generate_caption(
    image: Image.Image,
    text: str = None,
    decoding_method: str = "Nucleus sampling",
    temperature: float = 1.0,
    length_penalty: float = 1.0,
    repetition_penalty: float = 1.5,
    max_length: int = 50,
    min_length: int = 1,
    num_beams: int = 5,
    top_p: float = 0.9,
) -> str:
    
    if text is not None:
        inputs = processor(images=image, text=text, return_tensors="pt").to("cuda", torch.float16)
        generated_ids = model.generate(**inputs)
    else:
        inputs = processor(images=image, return_tensors="pt").to("cuda", torch.float16)
        generated_ids = model.generate(
            pixel_values=inputs.pixel_values,
            do_sample=decoding_method == "Nucleus sampling",
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            top_p=top_p,
        )
    result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return result

def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image


if __name__ == "__main__":

    if not os.path.exists("results"):
        os.makedirs("results")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # blip2
    MODEL_ID = "Salesforce/blip2-flan-t5-xl"
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast = True)
    model = Blip2ForConditionalGeneration.from_pretrained(MODEL_ID, device_map="cuda", load_in_8bit=False, torch_dtype=torch.float16)
    model.eval()

    # image dirs
    style_image_dir = "./data/style/bg1_resized.jpg"
    style_image = Image.open(style_image_dir).convert("RGB")

    content_image_dir = "./data/content/robot.jpg"
    content_image = Image.open(content_image_dir).convert("RGB")
    content_image = resize_img(content_image)
    content_image_prompt = generate_caption(content_image)
    # 使用BLIP模型生成内容图像的描述
    print(content_image_prompt)

    # init style clip model
    clip_model = CSD_CLIP("vit_large", "default", model_path="./CSD_Score/models/ViT-L-14.pt")
    model_path = "./CSD_Score/models/checkpoint.pth"
    checkpoint = torch.load(model_path, map_location="cpu", weights_only = False)
    state_dict = convert_state_dict(checkpoint['model_state_dict'])
    clip_model.load_state_dict(state_dict, strict=False)
    clip_model = clip_model.to(device)

    # preprocess
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    preprocess = transforms.Compose([
                    transforms.Resize(size=224, interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])

    # computer style embedding
    style_image_ = preprocess(Image.open(style_image_dir).convert("RGB")).unsqueeze(0).to(device) # torch.Size([1, 3, 224, 224])
    with torch.no_grad():
        _, __, style_output = clip_model(style_image_)

    # computer content embedding
    content_image_ = preprocess(Image.open(content_image_dir).convert("RGB")).unsqueeze(0).to(device) # torch.Size([1, 3, 224, 224])
    with torch.no_grad():
        _, content_output, __ = clip_model(content_image_)

    # inversion
    model_type = Model_Type.SDXL
    scheduler_type = Scheduler_Type.DDIM
    pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_type, device=device, model_name="./checkpoints/sdxlUnstableDiffusers_v8HeavensWrathVAE")

    config = RunConfig(model_type = model_type,
                       num_inference_steps = 50,
                       num_inversion_steps = 50,
                       num_renoise_steps = 1,
                       scheduler_type = scheduler_type,
                       perform_noise_correction = False,
                       seed = 7865
                      )
    
    # obtain content latent
    _, inv_latent, _, all_latents = invert(content_image,
                                           content_image_prompt,
                                           config,
                                           pipe_inversion=pipe_inversion,
                                           pipe_inference=pipe_inference,
                                           do_reconstruction=False) # torch.Size([1, 4, 128, 128])
    
    rec_image = pipe_inference(image = inv_latent,
                               prompt = content_image_prompt,
                               denoising_start=0.00001,
                               num_inference_steps = config.num_inference_steps,
                               guidance_scale = 1.0).images[0]

    rec_image.save(f"./results/result_rec.jpg")

    del pipe_inversion, pipe_inference, all_latents
    torch.cuda.empty_cache()
    
    control_type = "canny"
    if control_type == "tile":
        # condition image
        cond_image = load_image(content_image_dir)
        cond_image = resize_img(cond_image)
        
        controlnet_path = "./controlnet-tile-sdxl-1.0"
        controlnet = ControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to(device)
        
    elif control_type == "canny":
        # condition image
        input_image_cv2 = cv2.imread(content_image_dir)
        input_image_cv2 = np.array(input_image_cv2)
        input_image_cv2 = cv2.Canny(input_image_cv2, 100, 200)
        input_image_cv2 = input_image_cv2[:, :, None]
        input_image_cv2 = np.concatenate([input_image_cv2, input_image_cv2, input_image_cv2], axis=2)
        anyline_image = Image.fromarray(input_image_cv2)
        cond_image = resize_img(anyline_image)

        # load ControlNet
        controlnet_path = "./checkpoints/MistoLine"
        controlnet = ControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(device)

    # load pipeline
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "./checkpoints/IP-Adapter", subfolder="./models/image_encoder", torch_dtype=torch.float16
    ).to(device)

    pipe_inference = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                    "./checkpoints/sdxlUnstableDiffusers_v8HeavensWrathVAE",
                    controlnet=controlnet,
                    clip_model=clip_model,
                    image_encoder=image_encoder,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    #variant="fp16",
                ).to(device)
    pipe_inference.scheduler = DDIMScheduler.from_config(pipe_inference.scheduler.config) # works the best
    pipe_inference.unet.enable_gradient_checkpointing()

    # load multiple IPA
    pipe_inference.load_ip_adapter(
        ["./checkpoints/IP-Adapter", 
         "./checkpoints/IP-Adapter", 
        ],
        subfolder=["sdxl_models", "sdxl_models"],
        weight_name=[
            "ip-adapter_sdxl_vit-h.safetensors",
            "ip-adapter_sdxl_vit-h.safetensors",
        ],
        image_encoder_folder=None,
    )

    scale_global = 0.2 # high semantic content decrease style effect, lower it can benefit from textual or material style
    scale_style = {
        "up": {"block_0": [0.0, 1.2, 0.0]},
    }
    pipe_inference.set_ip_adapter_scale([scale_global, scale_style])

    # infer
    images = pipe_inference(
        prompt=content_image_prompt, # prompt used for inversion
        negative_prompt="lowres, low quality, worst quality, deformed, noisy, blurry",
        ip_adapter_image=[content_image, style_image], # IPA for semantic content, InstantStyle for style
        guidance_scale=5, # high cfg increase style
        num_inference_steps=config.num_inference_steps, # config.num_inference_steps achieves the best
        image=inv_latent, # init content latent
        #image=None, # init latent from noise
        control_image=cond_image, # ControlNet for spatial structure
        # 控制下面变量调整风格、内容之间的比例
        controlnet_conditioning_scale=0.5, # high control cond decrease style
        denoising_start=0.0001,
        style_embeddings_clip=style_output, # style guidance embedding
        content_embeddings_clip=content_output, # content guidance embedding
        style_guidance_scale=0, # enable style_guidance when style_guidance_scale > 0, cost high RAM, need optimization here
        content_guidance_scale=0, # enable content_guidance when style_guidance_scale > 0, cost high RAM, need optimization here
    ).images

    # computer style similarity score
    generated_image = preprocess(images[0]).unsqueeze(0).to(device)
    _, content_output1, style_output1 = clip_model(generated_image)

    style_sim = (style_output@style_output1.T).detach().cpu().numpy().mean()
    content_sim = (content_output@content_output1.T).detach().cpu().numpy().mean()

    print(style_sim, content_sim)

    prompt = ''
    images[0].save(f"./results/result_{style_sim}_{content_sim}.jpg")
    style_image_path = style_image_dir
    output_dir = "./output"
    content_image_path = f"./results/result_{style_sim}_{content_sim}.jpg"

    input_fg = np.array(Image.open(content_image_path))
    input_bg = np.array(Image.open(style_image_path))
    bg_height, bg_width = input_bg.shape[:2]
    params = {
        "prompt": prompt,
        "image_width": bg_width,
        "image_height": bg_height,
        "num_samples": 1,
        "seed": 12345,
        "steps": 20,
        "a_prompt": "best quality",
        "n_prompt": "lowres, bad anatomy, bad hands, cropped, worst quality",
        "cfg": 7.0,
        "highres_scale": 1.5,
        "highres_denoise": 0.5,
        "bg_source": BGSource.UPLOAD.value
    }
    results = process_relight(input_fg, input_bg, **params)
    
    # 保存结果
    for idx, img in enumerate(results):
        output_path = os.path.join(output_dir, f"result_{idx}.png")
        Image.fromarray(img).save(output_path)
        print(f"Saved result to {output_path}")