import argparse
import IPython
from IPython import get_ipython
def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   
        elif shell == 'TerminalInteractiveShell':
            return False  
        else:
            return False  
    except NameError:
        return False
def parse_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='chemsam', help='net type:sam_adaptered or sam_ori')
    parser.add_argument('-mod', type=str, default='chemsam_adpt', help='mod type:seg,cls,val_ad')
    parser.add_argument('-exp_name', type=str, default='chemseg_mc', help='net type used for log dir name')
    parser.add_argument('-vis', type=int, default=100, help='visualization')
    parser.add_argument('-reverse', type=bool, default=False, help='adversary reverse')
    parser.add_argument('-pretrain', type=bool, default=False, help='pretrain model')
    parser.add_argument('-val_freq',type=int,default=50,help='interval between each validation')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-gpu_device', type=int, default=0, help='use which gpu')
    parser.add_argument('-sim_gpu', type=int, default=0, help='split sim to this gpu')
    parser.add_argument('-epoch_ini', type=int, default=1, help='start epoch')
    parser.add_argument('-image_size', type=int, default=512, help='image_size')
    parser.add_argument('-out_size', type=int, default=128, help='output_size')
    parser.add_argument('-w', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=4, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('-imp_lr', type=float, default=3e-4, help='implicit learning rate')
    parser.add_argument('-base_weights', type=str, default = 0, help='the weights baseline')
    parser.add_argument('-sim_weights', type=str, default = 0, help='the weights sim')
    parser.add_argument('-distributed', default=False,type=bool,help='DDP or not')
    parser.add_argument('--local-rank', default = 0, type=int,help='DDP multi GPU ids to use')
    parser.add_argument('--local_rank', type=int, default=-1,  help='local rank passed from distributed launcher')
    parser.add_argument("--deepspeed", action="store_true", help="use deepspeed or not")
    parser.add_argument("--deepspeed_config", type=str, default=None, help="deepspeed config")   
    parser.add_argument('-dataset', default='segchem' ,type=str,help='segchem or isic dataset name')
    parser.add_argument('-loadSaved_point', type=str, default = None, help='the saved_point file you want to test,conflict with -sam_ckpt')
    parser.add_argument('-sam_ckpt', default='./logs/chemseg_pix_sdg_2023_09_08_19_02_14/Model/checkpoint_best.pth' , help='chemsam checkpoint address')
    parser.add_argument('-chunk', type=int, default=96 , help='crop volume depth')
    parser.add_argument('-evl_chunk', type=int, default=None , help='evaluation chunk')
    parser.add_argument('-thd', type=bool, default=False , help='3d or not')
    parser.add_argument('-train_dat', type=str, default='segchem_mcsmall.csv' , help='3d or not')
    parser.add_argument('-val_dat', type=str, default='segchem_mcsmall_val.csv' , help='3d or not')
    parser.add_argument(
    '-data_path',
    type=str,
    default='./data/isic',
    help='The path of segmentation data')
    parser.add_argument('-epochs', type=int, default=30000, help='total epochs ')
    if is_notebook():
        args = parser.parse_args(args=[]) 
    else:
        args = parser.parse_args() 
    return args