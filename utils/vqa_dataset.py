import json
import os
import random

import cv2
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from model.segment_anything.utils.transforms import ResizeLongestSide

from .conversation import get_default_conv_template
from .utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
)


class VQADataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        vqa_data="llava_instruct_150k",
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        DATA_DIR = os.path.join(base_image_dir, "llava_dataset")
        self.vqa_image_root = os.path.join(base_image_dir, "coco/train2017")
        with open(os.path.join(DATA_DIR, "{}.json".format(vqa_data))) as f:
            vqa_data = json.load(f)
        self.vqa_data = vqa_data

        print("vqa_data: ", len(self.vqa_data))

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.vqa_data) - 1)
        item = self.vqa_data[idx]
        image_path = os.path.join(self.vqa_image_root, item["image"])
        img = cv2.imread(image_path)
        images = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ori_size = images.shape[:2]
        images_clip = self.clip_image_processor.preprocess(images, return_tensors="pt")[
            "pixel_values"
        ][
            0
        ]  # preprocess images for clip
        image_token_len = (images_clip.shape[1] // 14) * (
            images_clip.shape[2] // 14
        )  # FIXME: 14 is hardcoded patch size

        images = self.transform.apply_image(images)  # preprocess images for sam
        resize = images.shape[:2]
        source = item["conversations"]
        conv = get_default_conv_template(
            "vicuna"
        ).copy()  # conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        conversations = []
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

        questions = conversations
        sampled_classes = conversations

        # replace <image> token
        for i in range(len(conversations)):
            replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversations[i] = conversations[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )

        images = self.preprocess(torch.from_numpy(images).permute(2, 0, 1).contiguous())

        masks = torch.rand(0, *ori_size)
        label = torch.ones(ori_size) * self.ignore_label

        return (
            image_path,
            images,
            images_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_classes,
        )
