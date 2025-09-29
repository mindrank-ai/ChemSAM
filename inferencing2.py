import os
import numpy as np
import imageio
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
import torch
import cfg
from utils import get_network
import random
import time
import numpy as np
import scipy
from scipy.ndimage import label
import skimage
from skimage.morphology import binary_dilation
from skimage.morphology import binary_erosion
from complete_structure import expand_masks
import fitz
import file_handler
import torchvision.transforms as transforms
import numbers
import warnings
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
warnings.filterwarnings('ignore')
def scale_round(x):
    return int(round(x))
def get_PIL_size(arr, size):
    if isinstance(size, numbers.Number):
        h, w = arr.shape
        size = scale_round(w * size), scale_round(h * size) 
        return size
    elif (isinstance(size, tuple) or isinstance(size, list)) and len(size) == 2:
        return (size[1], size[0]) 
    raise ValueError("Size is something weird %s" % size)
def scale_array(img, scale, interp=Image.BICUBIC):
    size = get_PIL_size(img, scale)
    img_scaled = Image.fromarray(img).resize(size, resample=interp)
    img_scaled = np.array(img_scaled)
    return img_scaled
def pdf2PIL(pdf_path, start_page=None, end_page=None,gray_scale=False):
    fileler=file_handler.FileHandler(support_pdfs=True, use_pdfbox=False)
    page_range = file_handler.format_page_range_str(start_page=start_page,
                                        end_page=end_page)
    return [(page_idx, img)  for page_idx, img in fileler.page2PIL(
                pdf_path, page_range,gray_scale=gray_scale)]
def page2imagePIL(doc,page_ix=2,resolution=(300,300),mode='RGB'):
    page = doc[page_ix]
    zoom_x=2.0
    zoom_y=2.0
    mat=fitz.Matrix(zoom_x,zoom_y)
    pix = page.get_pixmap(dpi=300)  
    if resolution:
        pix.set_dpi(*resolution)
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    return img
def np2pil(image_array):
    blurred_array = image_array.astype('int8') * 255
    blurred_array_image=Image.fromarray(np.uint8(blurred_array))
    return blurred_array_image
def seedpix(image_array,contour_mask,pg, debug=False):
    y_coordinates, x_coordinates = np.nonzero(contour_mask)
    x_center = int((x_coordinates.max() + x_coordinates.min()) / 2)
    y_center = int((y_coordinates.max() + y_coordinates.min()) / 2)
    mask_array=1-contour_mask
    seed_pixels = []
    up, down, right, left = True, True, True, True
    terminal_=np.min(image_array.shape)
    for n in range(1, terminal_):
        if up:
            if x_center + n < image_array.shape[1]:
                if not mask_array[y_center, x_center + n]:
                    up = False
                if not image_array[y_center, x_center + n]:
                    seed_pixels.append((x_center + n, y_center))
                    up = False
                    if debug:
                        print(f'up::{n}')
        if down:
            if x_center - n >= 0:
                if not mask_array[y_center, x_center - n]:
                    down = False
                if not image_array[y_center, x_center - n]:
                    seed_pixels.append((x_center - n, y_center))
                    down = False
                    if debug:
                        print(f'down::{n}')
        if left:
            if y_center + n < image_array.shape[0]:
                if not mask_array[y_center + n, x_center]:
                    left = False
                if not image_array[y_center + n, x_center]:
                    seed_pixels.append((x_center, y_center + n))
                    left = False
                    if debug:
                        print(f'left::{n}')
        if right:
            if y_center - n >= 0:
                if not mask_array[y_center - n, x_center]:
                    right = False
                if not image_array[y_center - n, x_center]:
                    seed_pixels.append((x_center, y_center - n))
                    right = False
                    if debug:
                        print(f'right::{n}')
    return seed_pixels
def get_bounding_box(contour):
    new_tl_y = int(np.min(contour[:, 0]))
    new_tl_x = int(np.min(contour[:, 1]))
    new_br_y = int(np.max(contour[:, 0]))
    new_br_x = int(np.max(contour[:, 1]))
    bbox = [new_tl_y, new_tl_x, new_br_y, new_br_x]
    return bbox
def bbox_filter(bboxes,w_=400, h_=400, area_size_threshold=8050*3):
    unique_list = [list(x) for x in set( tuple(x) for x in bboxes)   ]
    filtered_bbox_list = []
    outer_bbox=[]
    used_labels=[]
    for i, bbox1 in enumerate(unique_list):
        h=bbox1[2]-bbox1[0]
        w=bbox1[3]-bbox1[1]
        if w<w_ and h <h_:
            continue
        elif  h*w < area_size_threshold:
            print(h*w - area_size_threshold,'h*w--cutoff')
            continue
        is_inner_box = False
        for j, bbox2 in enumerate(unique_list):
            if i == j:
                continue
            if bbox1[0] >= bbox2[0] and bbox1[2] <= bbox2[2] and               bbox1[1] >= bbox2[1] and bbox1[3] <= bbox2[3]:
                is_inner_box = True
                if bbox2 not in outer_bbox:
                    outer_bbox.append(bbox2)
                break
        filtered_bbox_list.append(bbox1)
        used_labels.append(i+1)
    filtered_bbox_list=[box for box in filtered_bbox_list if box not in outer_bbox]
    return filtered_bbox_list,outer_bbox
def get_scalar(ori_image,new_size):
    ori_w, ori_h=ori_image.size
    y_scaling_factor=ori_h/new_size[0]
    x_scaling_factor=ori_w/new_size[1]
    ori_size=(ori_h,ori_w)
    return y_scaling_factor,x_scaling_factor,ori_size
def boxscalar(boxes_list,ori_size,new_size):
    y_scaling_factor=ori_size[0]/new_size[0]
    x_scaling_factor=ori_size[1]/new_size[1]
    scaled_boxes = []
    for box in boxes_list:
        scaled_box = [
            int(box[0] * y_scaling_factor),
            int(box[1] * x_scaling_factor),
            int(box[2] * y_scaling_factor),
            int(box[3] * x_scaling_factor),
        ]
        scaled_boxes.append(scaled_box)
    return scaled_boxes
def chemsam(file_path='./WO2021092153A1.pdf'):
    args = cfg.parse_args()
    args.gpu=False
    if args.gpu:
        GPUdevice = torch.device('cuda', args.gpu_device)
    else:
        GPUdevice = torch.device('cpu')
    args.image_size= 512 
    args.out_size= 128 
    new_size=(args.image_size,args.image_size)
    transform_train = transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)),
        transforms.ToTensor(),
    ])
    args.net= 'sam_adaptered'
    args.loadSaved_point= 'logs/chemseg_pix_sdg_2023_07_10_17_34_25/Model/last_660_checkpoint.pth'
    if args.loadSaved_point:
        args.sam_ckpt=None
    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
    net.to(GPUdevice)
    tot=[]
    s = np.ones((3,3))
    usebinary_dilation=True
    ori_blur=True
    debug=False
    boxviewer=True
    seg_figs=False
    start_epoch=0
    if args.loadSaved_point:
        if debug:
            print(f'=> resuming from {args.loadSaved_point}')
        assert os.path.exists(args.loadSaved_point)
        checkpoint_file = os.path.join(args.loadSaved_point)
        assert os.path.exists(checkpoint_file)
        if args.gpu:
            loc = 'cuda:{}'.format(args.gpu_device)
        else:
            loc='cpu'
        checkpoint = torch.load(checkpoint_file, map_location=loc)
        start_epoch = checkpoint['epoch']
        best_tol = checkpoint['best_tol']
        net.load_state_dict(checkpoint['state_dict'],strict=True)
        if debug:
            print(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
    name, ext = os.path.splitext(file_path)
    pix_cut=75
    base_fname=name.split('/')[-1]+f'pixcut_{pix_cut}'
    os.makedirs(base_fname, exist_ok = True)
    with open(base_fname+'/'+'page_cropedImage.csv','a+') as wf:
            wf.write('page,structures,sum\n')
    id_imar = pdf2PIL(file_path, gray_scale=False)
    for pg, image_pil in id_imar:
        time_start = time.time()
        ori_w, ori_h=image_pil.size
        gray_image=image_pil.convert('L')
        scaled_image = gray_image.resize(new_size)
        threshold = np.mean(np.array(scaled_image))
        binary_image = scaled_image.point(lambda x: 255 if x > threshold else 0, mode='1')
        binary_array = np.array(binary_image)
        if ori_blur:
            blur_factor = (int(binary_array.shape[1] / 185) if binary_array.shape[1] / 185 >= 2 else 2)
            kernel = np.ones((blur_factor, blur_factor))
            bool_arry = binary_erosion(binary_array, footprint=kernel)
        else:
            bool_arry=binary_array
        x1=transform_train(image_pil)
        x1=x1.unsqueeze(0).to(dtype = torch.float32,device = GPUdevice)
        out_= net.image_encoder(x1)
        bs=out_.size()[0]
        sparse_embeddings = torch.empty((bs, 0, net.prompt_encoder.embed_dim), device=net.prompt_encoder._get_device())
        dense_embeddings = net.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                        bs, -1, net.prompt_encoder.image_embedding_size[0], net.prompt_encoder.image_embedding_size[1]
                    )
        pred_masks, iou_pred_ = net.mask_decoder(
            image_embeddings=out_,
            image_pe=net.prompt_encoder.get_dense_pe(), 
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings, 
            multimask_output=False,
            )
        pred_sigs = torch.sigmoid(pred_masks)
        pred_bhw = pred_sigs.cpu().detach().numpy()[:, 0, :, :]
        pix_cut=75
        rescaled_mask=scale_array(pred_bhw[0], new_size)
        predicted_mask = rescaled_mask*255
        predicted_mask[predicted_mask >=pix_cut] = 255
        predicted_mask[predicted_mask < pix_cut] = 0
        if debug:
            print(np.mean(predicted_mask),np.max(predicted_mask))
        if usebinary_dilation:
            dilatedd_mask = binary_dilation(predicted_mask,footprint=np.ones((5,5)))
            predicted_mask = dilatedd_mask.astype('int8') * 255
        predicted_mask_image=Image.fromarray(np.uint8(predicted_mask))
        if usebinary_dilation:
            predicted_mask_image.save(f'{base_fname}/binary_dilation_page_{pg+1}.png')
        labels, num_labels = scipy.ndimage.label(predicted_mask, structure=s)
        ar_masks=[]
        for label in range(1, num_labels + 1):
            contour_mask = labels == label
            contour_pixel = np.argwhere(contour_mask)
            ar_masks.append((contour_pixel.shape[0],label, contour_mask))
        l2s_masks=sorted(ar_masks,key=lambda a: a[0],reverse=True)
        contour_masks=[x[2] for x in l2s_masks]
        expanded_split_mask_arrays=[]
        filtered=[]
        for i, contour_mask in enumerate(contour_masks):
            seed_pixels=seedpix(bool_arry,contour_mask,pg)
            if seed_pixels != []:
                mask_array2 = expand_masks(bool_arry, seed_pixels, contour_mask) 
                expanded_split_mask_arrays.append(mask_array2)
            else:
                filtered.append(contour_mask)
        bboxes=[]
        for mask_array2 in expanded_split_mask_arrays:
            mask_bool=mask_array2.astype('int8')  !=0
            contour_pixel = np.argwhere(mask_bool)
            bbox=get_bounding_box(contour_pixel)
            bboxes.append(bbox)
        unique_list = [list(x) for x in set( tuple(x) for x in bboxes)   ]
        y_scaling_factor,x_scaling_factor,ori_size =get_scalar(image_pil,new_size)
        area_size_threshold=8050/(y_scaling_factor *x_scaling_factor)
        w_=400/x_scaling_factor
        h_=400/y_scaling_factor
        filtered_bbox_list,outer_bbox=bbox_filter(bboxes,w_,h_,area_size_threshold=area_size_threshold)
        if debug:
            print('area:unit with 1x1 pixcel', [(b[2]-b[0])*(b[3]-b[1]) for b in filtered_bbox_list ])
            print('h,w:unit with 1 pixcel',[[(b[2]-b[0]),(b[3]-b[1])] for b in filtered_bbox_list ])
        scaled_boxes= boxscalar(filtered_bbox_list,ori_size,new_size)
        i=0
        image_array=np.array(gray_image)
        if seg_figs:
            i=0
            for i, bbox in enumerate(scaled_boxes):
                cropped_region = image_array[bbox[0]:bbox[2],bbox[1]:bbox[3]]
                cropped_image = Image.fromarray(np.uint8(cropped_region))
                i+=1
                fname = f'{base_fname}_pg{pg+1}_segm{i}.png'
                cropped_image.save(base_fname+'/'+fname)
        if boxviewer:
            i=0
            image_cp=copy.deepcopy(image_pil)
            draw = ImageDraw.Draw(image_cp)
            for box in scaled_boxes:
                pil_box=[box[1],box[0],box[3],box[2]]
                x1,y1,x2,y2=pil_box
                area=(x2-x1 )* (y2-y1)
                if area>8050:
                    i+=1
                    draw.rectangle(pil_box, outline='red',width=3)
            original_width, original_height = image_cp.size
            new_width = original_width // 10
            new_height = original_height // 10
            zoomed_out_image = image_cp.resize((new_width, new_height))
            fname = f'{pg+1}_boxvier.png'
            zoomed_out_image.save(base_fname+'/'+fname)
        if i:
            line_=f'{pg+1}, {i}\n'
            tot.append(i)
            with open(base_fname+'/'+'page_cropedImage.csv','a+') as wf:
                wf.write(line_)
        time_end_inf = time.time()
    with open(base_fname+'/'+'page_cropedImage.csv','a+') as wf:
            wf.write(f'total,cropedStrcut,{sum(tot)}\n')
if __name__=='__main__':
    f1_png="pdf_list.txt"
    path = os.getcwd()
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    task_Number = os.path.basename(grandparent_dir)
    pdfs=[x for x in os.listdir('./') if 'pdf' in x]
    numproces=len(pdfs)
    print(pdfs)
    chemsam(pdfs[0])