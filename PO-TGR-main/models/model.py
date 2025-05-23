from clip import clip
import torch
import torch.nn as nn
import pickle
from collections import OrderedDict
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # Context initialization
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # [n_ctx, ctx_dim]

        # TLCN_net (instance-conditioned prompt shift)
        self.TLCN_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, vis_dim // 8)),
            ("relu2", nn.ReLU(inplace=True)),
            ("linear3", nn.Linear(vis_dim // 8, ctx_dim))
        ]))

        if cfg.TRAINER.COOP.PREC == "fp16":
            self.TLCN_net.half()

        # Tokenization
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])       # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS + EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self, image_features=None):
        ctx = self.ctx  # [n_ctx, ctx_dim]
        prefix = self.token_prefix  # [n_cls, 1, ctx_dim]
        suffix = self.token_suffix  # [n_cls, *, ctx_dim]

        # Case 1: static prompt (task 0 or evaluation)
        if image_features is None:
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            return torch.cat([prefix, ctx, suffix], dim=1)  # [n_cls, prompt_len, ctx_dim]

        # Case 2: dynamic prompt (with TLCN)
        bias = self.TLCN_net(image_features)      # [B, ctx_dim]
        bias = bias.unsqueeze(1)                  # [B, 1, ctx_dim]
        ctx = ctx.unsqueeze(0)                    # [1, n_ctx, ctx_dim]
        ctx_shifted = ctx + bias                  # [B, n_ctx, ctx_dim]

        prompts = []
        for ctx_i in ctx_shifted:  # ctx_i: [n_ctx, ctx_dim]
            ctx_i = ctx_i.unsqueeze(0).expand(self.n_cls, -1, -1)  # [n_cls, n_ctx, ctx_dim]
            prompt_i = torch.cat([prefix, ctx_i, suffix], dim=1)   # [n_cls, prompt_len, ctx_dim]
            prompts.append(prompt_i)

        return torch.stack(prompts)  # [B, n_cls, prompt_len, ctx_dim]

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, pseudo_feat=None):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features)
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        logits = []
        for i in range(image_features.size(0)):
            text_features = self.text_encoder(prompts[i], tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit = logit_scale * image_features[i] @ text_features.t()
            logits.append(logit)

        logits = torch.stack(logits)

        if pseudo_feat is not None:
            pseudo_feat = pseudo_feat.half()
            logits_pseudo = logit_scale * pseudo_feat @ text_features.t()
            return logits, logits_pseudo

        return logits

    def forward_pseudo(self, pseudo_feat):
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * pseudo_feat @ text_features.t()
        return logits
