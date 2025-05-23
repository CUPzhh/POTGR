import torch
from diffusers import StableDiffusionPipeline, LCMScheduler
from torchvision import transforms
from PIL import Image

# 创建自定义安全检查函数
def dummy_safety_checker(images, clip_input):
    return images, [False] * len(images)

model_name = ""
adapter_id = ""

# 初始化并缓存 Stable Diffusion 模型（推荐在外部只加载一次）
def load_diffusion_model(model_name, device="cuda"):
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32)
    pipe = pipe.to(device)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = dummy_safety_checker

    pipe.load_lora_weights(adapter_id)
    pipe.fuse_lora()

    # 禁用 safety checker（避免拦截内容）
    pipe.safety_checker = lambda images, **kwargs: (images, False)

    return pipe


def get_clip_preprocess():
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])


def generate_images_and_labels(classnames, b, device="cuda", prompt_template="a photo of a {}", model=None):
    """
    使用Stable Diffusion生成图像并返回tensor和标签

    Args:
        classnames: list[str], 要回放的类别名称
        b: int, 要生成图像的数量（从前b个类别生成）
        device: str, 使用的设备
        prompt_template: str, 文本提示模版
        model: 可选StableDiffusionPipeline实例（可复用）

    Returns:
        images_tensor: torch.Tensor, shape=[b, 3, 224, 224]
        labels_tensor: torch.Tensor, shape=[b]
    """
    prompts = [prompt_template.format(name) for name in classnames[:b]]

    # 加载模型
    pipe = model if model is not None else load_diffusion_model(device=device)

    # 执行图像生成
    images = pipe(prompts, num_inference_steps=8).images  # list of PIL Images

    # 转换为 CLIP 所需格式
    transform = get_clip_preprocess()
    images_tensor = torch.stack([transform(img) for img in images]).to(device)
    labels_tensor = torch.tensor(list(range(len(prompts))), dtype=torch.long, device=device)

    return images_tensor, labels_tensor
