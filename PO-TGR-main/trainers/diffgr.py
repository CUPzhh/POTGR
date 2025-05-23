import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.transforms.transforms import build_transform

from datasets.cifar100 import CIFAR100, CLASSNAME_CFIAR100
from datasets.cub200 import CUB200, CLASSNAME_CUB200
from datasets.miniImageNet import MiniImageNet, CLASSNAME_miniImageNet

from models.model import load_clip_to_cpu, CustomCLIP
from trainers.SD import generate_images_and_labels

from tqdm import tqdm


@TRAINER_REGISTRY.register()
class DiffGR(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_data_loader(self):

        self.tfm_train = build_transform(self.cfg, is_train=True)
        self.tfm_test = build_transform(self.cfg, is_train=False)

        self.batch_size_train = self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        self.batch_size_test = self.cfg.DATALOADER.TEST.BATCH_SIZE
        self.num_workers = self.cfg.DATALOADER.NUM_WORKERS

        self.task_id = self.cfg.TRAINER.TASK_ID

        self.dataset_name = self.cfg.DATASET.NAME
        self.data_root = self.cfg.DATASET.ROOT
        self.num_classes = self.cfg.DATASET.NUM_CLASSES
        self.num_classes_base = self.cfg.DATASET.NUM_CLASSES_BASE
        self.class_per_task = self.cfg.DATASET.CLASS_PER_TASK
        self.shot = self.cfg.DATASET.NUM_SHOTS
        self.B = self.cfg.DATASET.B

        self.task_num = int((self.num_classes - self.num_classes_base) / self.class_per_task)

        if self.num_classes_base > 0:
            self.encounter_class_id = self.num_classes_base + self.class_per_task * self.task_id
        else:
            self.encounter_class_id = self.class_per_task * (self.task_id + 1)

        if self.dataset_name == 'CIFAR100':
            train_set_task0 = CIFAR100(shot=self.shot,
                                       tfm=self.tfm_train,
                                       task_id=self.task_id,
                                       mode='train',
                                       class_per_task=self.class_per_task)

            test_set_task0 = CIFAR100(tfm=self.tfm_test,
                                      task_id=self.task_id,
                                      mode='test',
                                      class_per_task=self.class_per_task)

            self.classnames = CLASSNAME_CFIAR100

        elif self.dataset_name == 'CUB200':
            train_set_task0 = CUB200(data_root=self.data_root,
                                     shot=self.shot,
                                     tfm=self.tfm_train,
                                     task_id=self.task_id,
                                     mode='train',
                                     class_per_task=self.class_per_task)

            test_set_task0 = CUB200(data_root=self.data_root,
                                    tfm=self.tfm_test,
                                    task_id=self.task_id,
                                    mode='test',
                                    class_per_task=self.class_per_task)

            self.classnames = CLASSNAME_CUB200

        elif self.dataset_name == 'miniImageNet':
            train_set_task0 = MiniImageNet(data_root=self.data_root,
                                           tfm=self.tfm_train,
                                           task_id=self.task_id,
                                           mode='train',
                                           class_per_task=self.class_per_task)

            test_set_task0 = MiniImageNet(data_root=self.data_root,
                                          tfm=self.tfm_test,
                                          task_id=self.task_id,
                                          mode='test',
                                          class_per_task=self.class_per_task)

            self.classnames = CLASSNAME_miniImageNet

        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        self.classnames_encountered = self.classnames[:self.encounter_class_id]

        self.train_loader_x = torch.utils.data.DataLoader(
            train_set_task0, batch_size=self.batch_size_train,
            num_workers=self.num_workers, drop_last=False, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            test_set_task0, batch_size=self.batch_size_test,
            num_workers=self.num_workers, drop_last=False)

        self.val_loader = self.test_loader

        self.lab2cname = {
            x: self.classnames_encountered[x]
            for x in range(len(self.classnames_encountered))
        }

    def build_model(self):
        cfg = self.cfg

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC in ["fp32", "amp"]:
            clip_model.float()
        self.clip_model = clip_model

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, self.classnames_encountered, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        if torch.cuda.device_count() > 1:
            print("Multiple GPUs detected, using DataParallel")
            self.model = nn.DataParallel(self.model)

        self.lambda_o = self.cfg.TRAINER.LAMBDA_O

    def forward_backward(self, batch):
        image, label = batch[0].to(self.device), batch[1].to(self.device)

        if self.task_id == 0:
            prec = self.cfg.TRAINER.COOP.PREC
            if prec == "amp":
                with autocast():
                    output = self.model(image)
                    loss = F.cross_entropy(output, label)
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                output = self.model(image)
                loss = F.cross_entropy(output, label)
                self.model_backward_and_update(loss)
        else:
            pseudo_images, pseudo_label = generate_images_and_labels(self.classnames_encountered[:self.num_classes_base], self.B, device=self.device)
            pseudo_feat = self.model.image_encoder(pseudo_images.type(self.model.dtype))
            pseudo_feat = pseudo_feat / pseudo_feat.norm(dim=-1, keepdim=True)
            pseudo_label = pseudo_label.to(self.device)

            output, output_pseudo = self.model(image, pseudo_feat)
            loss = F.cross_entropy(torch.cat((output, output_pseudo)), torch.cat((label, pseudo_label)), reduction='none')

            weight_n = torch.ones((image.shape[0]))
            weight_o = torch.ones((pseudo_feat.shape[0]))

            #  loss = weight_n * lambda_o + weight_o * (1 - lambda_o)
            weight_n = weight_n * self.lambda_o
            weight_o = weight_o * (1 - self.lambda_o)
            weight = torch.cat((weight_n, weight_o)).half()

            loss = loss * (weight.to(loss.device).detach())
            loss = loss.mean()
            self.model_backward_and_update(loss)

        return {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

    @torch.no_grad()
    def test(self, split=None):
        self.set_model_mode("eval")
        self.evaluator.reset()
        split = split or self.cfg.TEST.SPLIT
        data_loader = self.val_loader if split == "val" else self.test_loader

        print(f"Evaluate on the *{split}* set")
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            image, label = batch
            image = image.to(self.device)
            label = label.to(self.device)
            output = self.model_inference(image)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        for k, v in results.items():
            self.write_scalar(f"{split}/{k}", v, self.epoch)

        return list(results.values())[0]




    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)











