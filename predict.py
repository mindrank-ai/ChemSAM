import os
import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
import torch
import cfg
import time
import scipy
import skimage
from skimage.morphology import binary_dilation,binary_erosion
import fitz
import file_handler
import torchvision.transforms as transforms
import numbers
import copy
from PIL import Image
import matplotlib.pyplot as plt
from models.chemsam import sam_model_registry
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
import rdkit.Chem as Chem
from Filter import dataset
from Filter.constants import RGROUP_SYMBOLS, ABBREVIATIONS
from getfilter import get_filter

SKEW_TOLERANCE = 2
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

def np2pil(image_array):
    blurred_array = image_array.astype('int8') * 255
    blurred_array_image=Image.fromarray(np.uint8(blurred_array))
    return blurred_array_image

def seedpix(image_array,contour_mask,debug=False):
    y_coordinates, x_coordinates = np.nonzero(contour_mask)
    x_center = int((x_coordinates.max() + x_coordinates.min()) / 2)
    y_center = int((y_coordinates.max() + y_coordinates.min()) / 2)
    mask_array=1-contour_mask
    seed_pixels = []
    up, down, right, left = True, True, True, True
    terminal_=np.min(image_array.shape)
    terminal_=100
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
        if w<w_ or h <h_:
            continue
        elif  h*w < area_size_threshold:
            print(h*w - area_size_threshold,'h*w--cutoff')
            continue
        filtered_bbox_list.append(bbox1)
    return filtered_bbox_list, outer_bbox

def labels2pixel(mask_boolarray,blurred_image_array):
    seed_pixels=[]
    yx_label=np.argwhere(mask_boolarray)
    for y,x in yx_label:
        if blurred_image_array[y,x]==0:
            seed_pixels.append((y,x))
    return seed_pixels

def mask2update(predicted_mask,blurred_image_array):
    seed_pixels=[]
    update_mask=np.zeros(predicted_mask.shape).astype('float32')
    yx_label=np.argwhere(predicted_mask)
    for y,x in yx_label:
        if blurred_image_array[y,x]==0:
            seed_pixels.append((y,x))
            update_mask[y,x]=1
    return update_mask, seed_pixels

def get_neighbour_pixels(seed_pixel,image_shape):
    neighbour_pixels = []
    y,x = seed_pixel
    for new_x in range(x - 1, x + 2):
        if new_x in range(image_shape[1]):
            for new_y in range(y - 1, y + 2):
                if new_y in range(image_shape[0]):
                    if (x, y) != (new_x, new_y):
                        neighbour_pixels.append((new_y,new_x))
    return neighbour_pixels

def expand_masks(blurred_image_array,seed_pixels):
    mask_array=np.zeros(blurred_image_array.shape).astype('bool')
    for seed_pixel in seed_pixels:
        neighbour_pixels = get_neighbour_pixels(seed_pixel, blurred_image_array.shape)
        for neighbour_pixel in neighbour_pixels:
            y,x = neighbour_pixel
            if not mask_array[y, x]:
                if not blurred_image_array[y, x]:
                    mask_array[y, x] = True
                    seed_pixels.append((y, x))
    return mask_array

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

def contours_plt(binary_image, contours):
    fig, ax = plt.subplots()
    ax.imshow(binary_image, cmap='gray')
    colors = plt.cm.get_cmap('tab10', len(contours))
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1)
    ax.axis('image')
    plt.show()

def get_polygon_area(corners):
    n = len(corners)  
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

def get_hough_lines(img_src, min_line_length):
    hough_threshold = min_line_length
    binary = (img_src < 250)
    lines = skimage.transform.probabilistic_hough_line(binary,
                                               threshold=hough_threshold,
                                               line_length=min_line_length,
                                               line_gap=0)
    img_bw_inverted = skimage.morphology.skeletonize(binary)
    lines += skimage.transform.probabilistic_hough_line(img_bw_inverted,
                                                threshold=hough_threshold,
                                                line_length=min_line_length,
                                                line_gap=0)
    coords_list = []
    for l in lines:
        p1 = l[0]
        p2 = l[1]
        if (abs(p1[1] - p2[1]) < SKEW_TOLERANCE) or               (abs(p1[0] - p2[0]) < SKEW_TOLERANCE):
            coords_list.append((p1[1], p1[0], p2[1], p2[0]))
    return coords_list

def remove_lines(img, mask):
    hough_scale = 1  
    if hough_scale != 1:
        img_for_hough = scale_array(img, hough_scale)
    else:
        img_for_hough = img.copy()
    mask = mask.copy()
    img = img.copy()
    img_for_hough[img_for_hough < np.mean(img_for_hough)] = 0
    img_for_hough[img_for_hough >= np.mean(img_for_hough)] = 255
    detected_lines = get_hough_lines(img_for_hough, 32 * 5)
    thickness = int(3 / hough_scale)
    if thickness % 2 == 0:
        thickness += 1  
    for coords in detected_lines:
        is_horizontal = abs(coords[2] - coords[0]) < abs(coords[3] - coords[1])
        coords = np.array(coords)
        if coords[0] != coords[2] and is_horizontal:
            avg = np.mean([coords[0], coords[2]])
            coords[0] = avg
            coords[2] = avg
        if coords[1] != coords[3] and not is_horizontal:
            avg = np.mean([coords[1], coords[3]])
            coords[1] = avg
            coords[3] = avg
        for i in range(thickness * 2):
            shift_val = (i - int(thickness // 2)) * 0.5
            x_shift = shift_val if not is_horizontal else 0
            y_shift = shift_val if is_horizontal else 0
            xy=(scale_round(coords[0] / hough_scale + y_shift),
                scale_round(coords[1] / hough_scale + x_shift),
                scale_round(coords[2] / hough_scale + y_shift),
                scale_round(coords[3] / hough_scale + x_shift))
            rr, cc = skimage.draw.line(*xy)
            rr = np.clip(rr, 0, img.shape[0] - 1)
            cc = np.clip(cc, 0, img.shape[1] - 1)
            mask[rr, cc] = 0
            img[rr, cc] = 255
    return img, mask

def clip2ori(bboxes_np_dict, filtered_bbox_list,image_pil,new_size):
    gray_image=image_pil.convert('L')
    gray_np=np.array(gray_image)
    color_np=np.array(image_pil)
    ori_size=gray_np.shape
    cropped_arrs=[]
    cropped_arrs_col=[]
    corpneed=[bboxes_np_dict[tuple(box)] for box in filtered_bbox_list]
    scaled_boxes= boxscalar(filtered_bbox_list,ori_size,new_size)
    for i, box in enumerate(scaled_boxes):
        croped_mask=corpneed[i]
        reversed=255-croped_mask[0]*255
        reversed=scale_array(reversed.astype('int8'), ori_size)
        page_col_arr=copy.deepcopy(color_np)
        page_gray_arr=copy.deepcopy(gray_np)
        page_col_arr[reversed==255,:] = 255.
        page_gray_arr[reversed==255] = 255.
        y_min,x_min,y_max,x_max=box
        cropped_arr_col = page_col_arr[int(y_min)-1:int(y_max)+1, int(x_min)-1:int(x_max)+1]
        cropped_arr_gray = page_gray_arr[int(y_min)-1:int(y_max)+1, int(x_min)-1:int(x_max)+1]
        cropped_arrs.append(cropped_arr_gray)
        cropped_arrs_col.append(cropped_arr_col)
    return cropped_arrs,cropped_arrs_col,scaled_boxes

def lables2masks(binary_array):
    labels, num_labels = scipy.ndimage.label(binary_array, structure= np.ones((3,3)))
    ar_masks=[]
    for label in range(1, num_labels + 1):
        contour_mask = labels == label
        contour_pixel = np.argwhere(contour_mask)
        ar_masks.append((contour_pixel.shape[0],label, contour_mask))
    l2s_masks=sorted(ar_masks,key=lambda a: a[0],reverse=True)
    contour_masks=[x[2] for x in l2s_masks]
    return ar_masks,l2s_masks,contour_masks

def expandmaskFromBlur(contour_masks,blurred_image_array):
    expanded_split_mask_arrays=[]
    filtered=[]
    for i, contour_mask in enumerate(contour_masks):
        seed_pixels=labels2pixel(contour_mask,blurred_image_array)
        if seed_pixels != []:
            mask_array2 = expand_masks(blurred_image_array, seed_pixels) 
            expanded_split_mask_arrays.append(mask_array2)
        else:
            filtered.append(contour_mask)
    return expanded_split_mask_arrays,filtered

def boxNp(expanded_split_mask_arrays):
    bboxes=[]
    bboxes_np_dict=defaultdict(list)
    for mask_array2 in expanded_split_mask_arrays:
        mask_bool=mask_array2.astype('int8')  !=0
        contour_pixel = np.argwhere(mask_bool)
        bbox=get_bounding_box(contour_pixel)
        bboxes.append(bbox)
        bboxes_np_dict[tuple(bbox)].append(mask_array2)
    return bboxes,bboxes_np_dict

def hwboxfilter(unique_list,w_,h_,area_size_threshold=None):
    filtered_bbox_list = []
    if not area_size_threshold:
        area_size_threshold=w_*h_
    for i, bbox1 in enumerate(unique_list):
        h=bbox1[2]-bbox1[0]
        w=bbox1[3]-bbox1[1]
        if w<w_ or h <h_:
            continue
        elif  h*w < area_size_threshold:
            print(h*w - area_size_threshold,'h*w--cutoff')
            continue
        filtered_bbox_list.append(bbox1)    
    return filtered_bbox_list

def expanded2boxeslist(expanded_split_mask_arrays):
    bboxes=[]
    bboxes_np_dict=defaultdict(list)
    for mask_array2 in expanded_split_mask_arrays:
        mask_bool=mask_array2.astype('int8')  !=0
        contour_pixel = np.argwhere(mask_bool)
        bbox=get_bounding_box(contour_pixel)
        bboxes.append(bbox)
        bboxes_np_dict[tuple(bbox)].append(mask_array2)
    unique_list = [list(x) for x in set( tuple(x) for x in bboxes)   ]
    return unique_list,bboxes_np_dict

def check_ccc(ccc):
    ccc_height=[x.width  for x in ccc ]
    ccc_height=[x.height  for x in ccc ]
    height= max(ccc_height)
    width=sum([x.width  for x in ccc ])
    concatenated_image = Image.new("L", (width, height))
    concatenated_image.paste(ccc[0], (0, 0))
    concatenated_image.paste(ccc[1], (ccc[0].width, 0))
    concatenated_image.paste(ccc[2], (ccc[0].width+ccc[1].width, 0))
    return concatenated_image

def filter_stucture(list_png):
    device = torch.device('cpu')
    path = './logs/chemseg_pix_sdg_2023_09_08_19_02_14/Model/params.pth'
    filter = get_filter(path,device)
    filter.decoder.compute_confidence=True    
    m_np=[]
    for imf in list_png:
        image = imf
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        abl_transform=dataset.get_transforms(384, augment=False)
        images=[abl_transform(image=image, keypoints=[])['image']]
        images = torch.stack(images, dim=0)
        features, hiddens=filter.encoder(images)
        batch_predictions = filter.decoder.decode(features)
        smiles=batch_predictions[0]['chartok_coords']['smiles']
        coords, symbols,atom_scores=(batch_predictions[0]['chartok_coords']['coords'],batch_predictions[0]['chartok_coords']['symbols'],batch_predictions[0]['chartok_coords']['atom_scores'])
        edges=batch_predictions[0]['edges']
        mol = Chem.RWMol()
        n = len(symbols)
        ids = []
        for i in range(n):
            symbol = symbols[i]
            if symbol[0] == '[':
                symbol = symbol[1:-1]
            if symbol in RGROUP_SYMBOLS:
                atom = Chem.Atom("*")
                if symbol[0] == 'R' and symbol[1:].isdigit():
                    atom.SetIsotope(int(symbol[1:]))
                Chem.SetAtomAlias(atom, symbol)
            elif symbol in ABBREVIATIONS:
                atom = Chem.Atom("*")
                Chem.SetAtomAlias(atom, symbol)
            else:
                try:  
                    atom = Chem.AtomFromSmiles(symbols[i])
                    atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
                except:  
                    atom = Chem.Atom("*")
                    Chem.SetAtomAlias(atom, symbol)
            if atom.GetSymbol() == '*':
                atom.SetProp('molFileAlias', symbol)
            idx = mol.AddAtom(atom)
            assert idx == i
            ids.append(idx)
        for i in range(n):
            for j in range(i + 1, n):
                if edges[i][j] == 1:
                    mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
                elif edges[i][j] == 2:
                    mol.AddBond(ids[i], ids[j], Chem.BondType.DOUBLE)
                elif edges[i][j] == 3:
                    mol.AddBond(ids[i], ids[j], Chem.BondType.TRIPLE)
                elif edges[i][j] == 4:
                    mol.AddBond(ids[i], ids[j], Chem.BondType.AROMATIC)
                elif edges[i][j] == 5:
                    mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
                    mol.GetBondBetweenAtoms(ids[i], ids[j]).SetBondDir(Chem.BondDir.BEGINWEDGE)
                elif edges[i][j] == 6:
                    mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
                    mol.GetBondBetweenAtoms(ids[i], ids[j]).SetBondDir(Chem.BondDir.BEGINDASH)
        pred_smiles = '<invalid>'
        id_bonds=[(i, bond) for i, bond in enumerate(mol.GetBonds())]
        if not (np.array(edges)>0).any():
            continue 
        else:
            x_ys={f"{x},{y}" for x,y in coords}
            ys={f"{y}" for x,y in coords}
            xs={f"{x}" for x,y in coords}
            sy_unique={}
            if len(set(symbols))==1:  
                if 'H' in list(set(symbols))[0]:
                    continue 
                elif 'C' in list(set(symbols))[0]:
                    continue 
            else:
                if len(id_bonds)==1:
                    continue 
                if len(set(xs))<=5 or len(set(ys))<=5:
                    continue 
                try:
                    sm = Chem.rdmolfiles.MolToSmiles(mol)
                    frags=sm.split('.')
                    if len(frags)>=5:
                        print(f' molecules are too frags ???, may be not molecule at all. mv to letter')
                        continue 
                except Exception as e:
                    print(f"Error converting molecule to SMILES: {e}")
                    sm = '<invalid>'
                else:
                     m_np.append(imf)
    return(m_np)

def chemsam(file_path):
    args = cfg.parse_args()
    args.gpu=False
    if args.gpu:
        GPUdevice = torch.device('cuda', args.gpu_device)
    else:
        GPUdevice = torch.device('cpu')
    args.net= 'sam_adaptered'
    args.loadSaved_point="./logs/chemseg_pix_sdg_2023_09_08_19_02_14/Model/save_pointEpoch.pth"
    args.image_size= 512
    args.out_size= 128
    transform_train = transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)),
        transforms.ToTensor(),
    ])
    if args.loadSaved_point:
        args.sam_ckpt=None
    net = sam_model_registry['vit_b'](args,checkpoint=args.sam_ckpt)
    net.to(GPUdevice)
    if args.loadSaved_point:
        assert os.path.exists(args.loadSaved_point)
        checkpoint_file = os.path.join(args.loadSaved_point)
        assert os.path.exists(checkpoint_file)
        if args.gpu:
            loc = 'cuda:{}'.format(args.gpu_device)
        else:
            loc='cpu'
        checkpoint = torch.load(checkpoint_file, map_location=loc)
        net.load_state_dict(checkpoint['state_dict'],strict=True)
    tot=[]
    pix_cut=10 
    seg_figs=True
    triple_view=True
    time_display=True
    usebinary_dilation=False
    debug=False
    name, ext = os.path.splitext(file_path)
    base_fname=name.split('/')[-1]
    os.makedirs(base_fname, exist_ok = True)
    with open(base_fname+'/'+'page_cropedImage.csv','a+') as wf:
            wf.write('page,structures,sum\n')
    id_imar = pdf2PIL(file_path, 
                                    start_page = 0, 
                                    end_page = -1,
                                    gray_scale=False)
    for pg, image_pil in id_imar:
        time_start = time.time()
        ori_w, ori_h=image_pil.size
        gray_image=image_pil.convert('L')
        new_size=(512,512)
        scaled_image = gray_image.resize(new_size)
        scaled_image_array=np.array(scaled_image)
        threshold = np.mean(scaled_image_array)
        if threshold<250:  threshold+=5     
        scaled_image_array[scaled_image_array >=threshold] = 255
        scaled_image_array[scaled_image_array < threshold] = 0
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
        image_array=np.array(gray_image)
        if time_display:
            time_end_inf = time.time()
            print(f'time_for_SAMinferencing: {time_end_inf - time_start}s  @page_{pg+1}' )
        rescaled_mask=scale_array(pred_bhw[0], new_size)
        predicted_mask = rescaled_mask*255
        predicted_mask[predicted_mask >=pix_cut] = 255
        predicted_mask[predicted_mask < pix_cut] = 0
        blur_factor = (int(scaled_image_array.shape[1] / 185) if scaled_image_array.shape[1] / 185 >= 2 else 2)
        kernel = np.ones((blur_factor, blur_factor))
        blurred_image_array_line = binary_erosion(scaled_image_array, footprint=kernel)
        scaled_array_deline,predicted_mask_deline=remove_lines(scaled_image_array,predicted_mask)
        blurred_image_array = binary_erosion(scaled_array_deline, footprint=kernel)
        blurred_image = np2pil(blurred_image_array)
        deline_image=Image.fromarray(scaled_array_deline)
        usebinary_dilation=False
        if usebinary_dilation:
            dilatedd_mask = binary_dilation(predicted_mask,footprint=np.ones((5,5)))
            predicted_mask = dilatedd_mask.astype('int8') * 255
            bb_np=255-np.array(blurred_image)
            dilatedd_bb_np = binary_dilation(bb_np,footprint=np.ones((3,3)))
        update_mask,seed_pixels_total=mask2update(predicted_mask,blurred_image_array)
        ar_masks,l2s_masks,contour_mask_list=lables2masks(update_mask)
        expanded_split_mask_arrays=[]
        filtered=[]
        for i, contour_mask in enumerate(contour_mask_list):
            seed_pixels=labels2pixel(contour_mask,blurred_image_array)
            if seed_pixels != []:
                mask_array2 = expand_masks(blurred_image_array, seed_pixels)
                expanded_split_mask_arrays.append(mask_array2)
            else:
                filtered.append(contour_mask)
        unique_list,bboxes_np_dict=expanded2boxeslist(expanded_split_mask_arrays)
        y_scaling_factor,x_scaling_factor,ori_size =get_scalar(image_pil,new_size)
        filtered_bbox_list=hwboxfilter(unique_list,w_=10,h_=10,area_size_threshold=100) 
        debug=False
        if debug:
            print(y_scaling_factor,x_scaling_factor,ori_size)
            print('area:unit with 1x1 pixcel', [(b[2]-b[0])*(b[3]-b[1]) for b in unique_list ])
            h_sort_unique_list=sorted(unique_list, key=lambda b:b[2]-b[0],reverse=True)
            w_sort_unique_list=sorted(unique_list, key=lambda b:b[3]-b[1],reverse=True)
            print('H  :unit with 1 pixcel',[[(b[2]-b[0]),(b[3]-b[1])] for b in h_sort_unique_list ])
            print('W  :unit with 1 pixcel',[[(b[2]-b[0]),(b[3]-b[1])] for b in w_sort_unique_list ])
        if triple_view:
            imagecc=copy.deepcopy(blurred_image)
            draw = ImageDraw.Draw(imagecc)
            imagecc_np=np.array(imagecc)
            contours_len=[]
            for box in filtered_bbox_list:
                pil_box=[box[1],box[0],box[3],box[2]]
                draw.rectangle(pil_box, outline='green',width=1)
            predicted_mask_image=Image.fromarray(np.uint8(predicted_mask))
            ccc=[predicted_mask_image,scaled_image,imagecc]
            ccc_height=[x.width  for x in ccc ]
            ccc_height=[x.height  for x in ccc ]
            height= max(ccc_height)
            width=sum([x.width  for x in ccc ])
            concatenated_image = Image.new("L", (width, height))
            concatenated_image.paste(ccc[0], (0, 0))
            concatenated_image.paste(ccc[1], (ccc[0].width, 0))
            concatenated_image.paste(ccc[2], (ccc[0].width+ccc[1].width, 0))
            fname = f'{base_fname}/{pg+1}_boxvier.jpg'
            concatenated_image.save(fname)
            check_view=True
            if check_view:
                ccc=[scaled_image,blurred_image,deline_image]
                concatenated_imageccc=check_ccc(ccc)
                predicted_mask_image=Image.fromarray(np.uint8(predicted_mask))
                predicted_mask_delineima=Image.fromarray(np.uint8(predicted_mask_deline))
                update_mask_image=Image.fromarray(np.uint8(update_mask*255))
                ccc=[predicted_mask_image,predicted_mask_delineima,update_mask_image]
                concatenated_imagemask=check_ccc(ccc)
                fname2 = f'{base_fname}/{pg+1}_mask.jpg'
                concatenated_imagemask.save(fname2)
                concatenated_imagemask
        seg_figs=True
        if seg_figs:
            expanded_split_mask_arrays=[]
            filtered=[]
            for i, contour_mask in enumerate(contour_mask_list):
                seed_pixels=labels2pixel(contour_mask,blurred_image_array)
                if seed_pixels != []:
                    mask_array2 = expand_masks(blurred_image_array, seed_pixels)
                    mask_array1 = expand_masks(blurred_image_array_line, seed_pixels) 
                    mask_array2=np.logical_or(mask_array1,mask_array2)
                    expanded_split_mask_arrays.append(mask_array2)
                else:
                    filtered.append(contour_mask)
            area_cur=blurred_image_array.shape[0]*blurred_image_array.shape[1]
            unique_list_seg,bboxes_np_dict_seg=expanded2boxeslist(expanded_split_mask_arrays)
            for k,v in bboxes_np_dict_seg.items():
                if k not in bboxes_np_dict.keys():
                    area=(k[2]-k[0])*(k[3]-k[1])
                    if area*2>=area_cur:
                        continue
                    else:
                        yx_index=np.argwhere(v)
                        h_yx=np.max(yx_index[:,0])-np.min(yx_index[:,0])
                        w_yx=np.max(yx_index[:,1])-np.min(yx_index[:,1])
                        if h_yx >20 and w_yx >20:
                            bboxes_np_dict[k].extend(v)
                            filtered_bbox_list.append(k)
                            print(f'box::{k} added')
            i=0
            for bboxes, nparray in bboxes_np_dict.items():
                if len(nparray)>1:
                    nparray=sorted(nparray, key= lambda a: a.shape[0],reverse=True)
                    nparray=nparray[0]
                else:
                    nparray=nparray[0]
                bboxes_np_dict[bboxes]=[nparray]
            cropped_arrs,cropped_arrs_col,scaled_boxes=clip2ori(bboxes_np_dict, filtered_bbox_list, image_pil,new_size)
            cronp_box=sorted(zip(cropped_arrs,cropped_arrs_col,scaled_boxes), key= lambda ar_box: ar_box[2][0] )
            m_np=[]
            for crop,crop_np,box_sca in cronp_box:
                crop_ima=Image.fromarray(crop)
                p=np.mean(crop)
                crop[crop >p]=255
                crop[crop <=p]=0
                crop_dilated=binary_dilation(255-crop,footprint=np.ones((20,20)) )
                labels, num_labels = scipy.ndimage.label(crop_dilated, structure= np.ones((3,3)))
                if num_labels>1:
                    ar_masks,l2s_masks,contour_masks=lables2masks(crop_dilated)
                    expanded_split_mask_arrays,filtered=expandmaskFromBlur(contour_masks,np.invert(crop_dilated))
                    bboxes2,bboxes_np_dict2=boxNp(expanded_split_mask_arrays)
                    filtered_bbox_list2 = [list(x) for x in set( tuple(x) for x in bboxes2)   ]
                    filtered_bbox_list2=hwboxfilter(filtered_bbox_list2,20,20)
                    corpneed2=[bboxes_np_dict2[tuple(box)] for box in filtered_bbox_list2]
                    for j, box in enumerate(filtered_bbox_list2):
                        croped_mask=corpneed2[j]
                        reversed=(255-croped_mask[0]*255).astype('int8')
                        page_image_arr=copy.deepcopy(crop_np)
                        page_image_arr[reversed==255] = 255.
                        y_min,x_min,y_max,x_max=box
                        if (y_max-y_min)>100 and (x_max-x_min)>100:
                            c_r2 = page_image_arr[int(y_min):int(y_max), int(x_min):int(x_max)]
                            m_np.append(c_r2)
                            i+=1
                elif num_labels==1:
                    c_h,c_w=crop.shape
                    if c_h>100 and c_w>100:
                        m_np.append(crop_np)
                        i+=1
                else:
                    print(f'which state u are ????!',num_labels)
            m_np=filter_stucture(m_np)
            for jj,ima_np in enumerate(m_np):
                fname = f'{base_fname}_page_{pg+1}_segm{jj}.png'
                crop_ima=Image.fromarray(ima_np)
                crop_ima.save(fname)
            if i:
                line_=f'{pg+1}, {i}\n'
                tot.append(i)
                with open(base_fname+'/'+'page_cropedImage.csv','a+') as wf:
                    wf.write(line_)
        if time_display:
            time_end_segedima = time.time()
            print(f'time_for_SAMimage segment saved: {time_end_segedima - time_start}s  @page_{pg+1}' )
    with open(base_fname+'/'+'page_cropedImage.csv','a+') as wf:
                wf.write(f'total,cropedStrcut,{sum(tot)}\n')
                
if __name__=='__main__':

    pdfs='./testpdf/acs.jmedchem.0c00456.pdf'
    chemsam(pdfs)
