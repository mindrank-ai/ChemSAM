from functools import partial
from pathlib import Path
import urllib.request
import torch
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
from .modeling import (
    ImageEncoderViT,
    MaskDecoder,
    PromptEncoder,
    Sam,
    TwoWayTransformer,
)
def build_sam_vit_h(args = None, checkpoint=None):
    return _build_sam(
        args,
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )
build_sam = build_sam_vit_h
def build_sam_vit_l(args, checkpoint=None):
    return _build_sam(
        args,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )
def build_sam_vit_b(args, checkpoint=None):
    return _build_sam(
        args,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )
sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}
def _build_sam(
    args,
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = args.image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        args,
        image_encoder=ImageEncoderViT(
            args = args,
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is None:
        return sam
    checkpoint = Path(checkpoint)
    if checkpoint.name == "sam_vit_b_01ec64.pth" and not checkpoint.exists():
        cmd = input("Download sam_vit_b_01ec64.pth from facebook AI? [y]/n: ")
        if len(cmd) == 0 or cmd.lower() == 'y':
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading SAM ViT-B checkpoint...")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                checkpoint,
            )
            print(checkpoint.name, " is downloaded!")
    elif checkpoint.name == "sam_vit_h_4b8939.pth" and not checkpoint.exists():
        cmd = input("Download sam_vit_h_4b8939.pth from facebook AI? [y]/n: ")
        if len(cmd) == 0 or cmd.lower() == 'y':
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading SAM ViT-H checkpoint...")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                checkpoint,
            )
            print(checkpoint.name, " is downloaded!")
    elif checkpoint.name == "sam_vit_l_0b3195.pth" and not checkpoint.exists():
        cmd = input("Download sam_vit_l_0b3195.pth from facebook AI? [y]/n: ")
        if len(cmd) == 0 or cmd.lower() == 'y':
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading SAM ViT-L checkpoint...")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                checkpoint,
            )
            print(checkpoint.name, " is downloaded!")
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            loaded_dict = torch.load(f)
        current_model_dict=sam.state_dict()
        if 'epoch' in loaded_dict.keys():
            print(loaded_dict['epoch'],'<---epoch !!!')
            loaded_state_dict=loaded_dict['state_dict']
        else:
            loaded_state_dict=loaded_dict
        key_ori=loaded_state_dict.keys()
        key_cur=current_model_dict.keys()
        diff_=[k for k in key_ori if k not in key_cur]
        same_=[k for k in key_cur if k in key_ori]
        assert len(diff_)==0, f'make sure loaded model keys: {len(key_ori)} are all included in current build model keys: {len(key_cur)}'
        new_state_dict=dict()
        i=0;j=0
        for k in same_:
            if loaded_state_dict[k].size()==current_model_dict[k].size():
                new_state_dict[k]=loaded_state_dict[k]
                i+=1
            else:
                new_state_dict[k]=current_model_dict[k] 
                j+=1
        sam.load_state_dict(new_state_dict, strict=False)
        print(f'{bcolors.WARNING}have partly load the match paramters>>number of:{len(new_state_dict.keys())}, all:{len(sam.state_dict().keys())}. {bcolors.ENDC}')
        print(f' {bcolors.WARNING} loaded keys {i}, skip mismatch size keys {j}')
    return sam