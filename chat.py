import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, CLIPImageProcessor

from model.LISA import LISA
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.conversation import get_default_conv_template


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA chat")
    parser.add_argument("--version", default="xinlai/LISA-13B-llama2-v0")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image-size", default=1024, type=int, help="image size")
    parser.add_argument("--model-max-length", default=512, type=int)
    parser.add_argument("--lora-r", default=-1, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[SEG]")
    ret_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids
    args.seg_token_idx = ret_token_idx[0]

    model = LISA(
        args.local_rank,
        args.seg_token_idx,
        tokenizer,
        args.version,
        args.lora_r,
        args.precision,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )

    weight = {}
    visual_model_weight = torch.load(
        os.path.join(args.version, "pytorch_model-visual_model.bin")
    )
    text_hidden_fcs_weight = torch.load(
        os.path.join(args.version, "pytorch_model-text_hidden_fcs.bin")
    )
    weight.update(visual_model_weight)
    weight.update(text_hidden_fcs_weight)
    missing_keys, unexpected_keys = model.load_state_dict(weight, strict=False)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif args.precision == "fp16":
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
    else:
        model = model.float().cuda()

    DEFAULT_IMAGE_TOKEN = "<image>"
    DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
    DEFAULT_IM_START_TOKEN = "<im_start>"
    DEFAULT_IM_END_TOKEN = "<im_end>"
    image_token_len = 256

    clip_image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    while True:
        conv = get_default_conv_template("vicuna").copy()
        conv.messages = []

        prompt = input("Please input your prompt: ")
        prompt = DEFAULT_IMAGE_TOKEN + " " + prompt
        replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
        replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        image_path = input("Please input the image path: ")
        if not os.path.exists(image_path):
            print("File not found in {}".format(image_path))
            continue

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size_list = [image.shape[:2]]
        if args.precision == "bf16":
            images_clip = (
                clip_image_processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][0]
                .unsqueeze(0)
                .cuda()
                .bfloat16()
            )
        elif args.precision == "fp16":
            images_clip = (
                clip_image_processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][0]
                .unsqueeze(0)
                .cuda()
                .half()
            )
        else:
            images_clip = (
                clip_image_processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][0]
                .unsqueeze(0)
                .cuda()
                .float()
            )
        images = transform.apply_image(image)
        resize_list = [images.shape[:2]]
        if args.precision == "bf16":
            images = (
                preprocess(torch.from_numpy(images).permute(2, 0, 1).contiguous())
                .unsqueeze(0)
                .cuda()
                .bfloat16()
            )
        elif args.precision == "fp16":
            images = (
                preprocess(torch.from_numpy(images).permute(2, 0, 1).contiguous())
                .unsqueeze(0)
                .cuda()
                .half()
            )
        else:
            images = (
                preprocess(torch.from_numpy(images).permute(2, 0, 1).contiguous())
                .unsqueeze(0)
                .cuda()
                .float()
            )

        input_ids = tokenizer(prompt).input_ids
        input_ids = torch.LongTensor(input_ids).unsqueeze(0).cuda()
        output_ids, pred_masks = model.evaluate(
            images_clip,
            images,
            input_ids,
            resize_list,
            original_size_list,
            max_new_tokens=512,
            tokenizer=tokenizer,
        )
        text_output = tokenizer.decode(output_ids[0], skip_special_tokens=False)
        text_output = (
            text_output.replace(DEFAULT_IMAGE_PATCH_TOKEN, "")
            .replace("\n", "")
            .replace("  ", "")
        )

        print("text_output: ", text_output)
        for i, pred_mask in enumerate(pred_masks):
            if pred_mask.shape[0] == 0:
                continue

            pred_mask = pred_mask.detach().cpu().numpy()[0]
            pred_mask = pred_mask > 0

            save_path = "{}/{}_mask_{}.jpg".format(
                args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
            )
            cv2.imwrite(save_path, pred_mask * 100)
            print("{} has been saved.".format(save_path))

            save_path = "{}/{}_masked_img_{}.jpg".format(
                args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
            )
            save_img = image.copy()
            save_img[pred_mask] = (
                image * 0.5
                + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
            )[pred_mask]
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, save_img)
            print("{} has been saved.".format(save_path))


if __name__ == "__main__":
    main(sys.argv[1:])
