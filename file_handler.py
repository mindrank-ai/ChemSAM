import os
import signal
import subprocess
from PIL import Image
import numpy as np
import fitz
def format_page_range_str(start_page=None, end_page=None):
    if start_page == None:
        return None
    if end_page == None:
        return str(start_page)
    end_page = '' if end_page == -1 else end_page
    return "%s:%s" % (start_page, end_page)
class TimeoutException(Exception):
    def __init__(self, *args, **kwargs):
        pass
def signal_handler(signum, frame):
    raise TimeoutException()
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
class FileHandler:
    TYPE_OPTIONS = ('pdf', 'png', 'jpeg', 'tiff', 'jpg', 'tif')
    def __init__(self, support_pdfs=True, use_pdfbox=False):
        self.support_pdfs = support_pdfs
        self.use_pdfbox = use_pdfbox
        self.pid = None
    def page2PIL(self,
                  file_path,
                  page_range=None,
                  resolution=300,
                  gray_scale=True):
        paths = []
        if os.path.isdir(file_path):
            for item in os.listdir(file_path):
                if file_path[0] != '.':
                    if os.path.splitext(
                            file_path)[1].lower()[1:] not in TYPE_OPTIONS[1:]:
                        raise ValueError("Batching not supported for PDFs")
                    paths.append(os.path.join(file_path, item))
        def load_image(path):
            img = Image.open(path).convert('L')
            return np.array(img)
        if self._determine_file_type(file_path) == 'pdf':
            print("Processing PDF...")
            if not self.support_pdfs:
                raise ValueError("File is a PDF though 'support_pdfs' == False")
            doc = fitz.open(file_path)
            page_count=doc.page_count
            if not page_range:
                page_range = range(page_count)
            else:
                if isinstance(page_range, list):
                    pass
                else:
                    if ':' in page_range:
                        page_range_arg = page_range.split(':')
                        if page_range_arg[1].strip() == '':
                            pr_start = int(page_range_arg[0])
                            page_range = range(page_count)[pr_start:]
                        else:
                            pr_start = int(page_range_arg[0])
                            pr_end = int(page_range_arg[1])
                            page_range = range(page_count)[pr_start:pr_end+1]
                    else:
                        page_range = range(page_count)[int(page_range):
                                                       int(page_range) + 1]
            for page_ix in page_range:
                try:
                    if type(resolution) is int or float:
                        resolution=(resolution,resolution)
                    img=page2imagePIL(doc,page_ix=page_ix,resolution=None,mode='RGB')
                except:
                    print(
                        'PDF caused an error. This is most often due to OCRed scans. Please send the PDF to the developers for debugging.'
                    )
                    continue
                if gray_scale:
                    img = img.convert('L')
                yield page_ix, img
    def load_file(self,
                  file_path,
                  page_range=None,
                  resolution=300,
                  gray_scale=True):
        paths = []
        if os.path.isdir(file_path):
            for item in os.listdir(file_path):
                if file_path[0] != '.':
                    if os.path.splitext(
                            file_path)[1].lower()[1:] not in TYPE_OPTIONS[1:]:
                        raise ValueError("Batching not supported for PDFs")
                    paths.append(os.path.join(file_path, item))
        def load_image(path):
            img = Image.open(path).convert('L')
            return np.array(img)
        if self._determine_file_type(file_path) == 'pdf':
            print("Processing PDF...")
            if not self.support_pdfs:
                raise ValueError("File is a PDF though 'support_pdfs' == False")
            doc = fitz.open(file_path)
            page_count=doc.page_count
            if not page_range:
                page_range = range(page_count)
            else:
                if isinstance(page_range, list):
                    pass
                else:
                    if ':' in page_range:
                        page_range_arg = page_range.split(':')
                        if page_range_arg[1].strip() == '':
                            pr_start = int(page_range_arg[0])
                            page_range = range(page_count)[pr_start:]
                        else:
                            pr_start = int(page_range_arg[0])
                            pr_end = int(page_range_arg[1])
                            page_range = range(page_count)[pr_start:pr_end+1]
                    else:
                        page_range = range(page_count)[int(page_range):
                                                       int(page_range) + 1]
            for page_ix in page_range:
                try:
                    if type(resolution) is int or float:
                        resolution=(resolution,resolution)
                    img=page2imagePIL(doc,page_ix=page_ix,resolution=None,mode='RGB')
                except:
                    print(
                        'PDF caused an error. This is most often due to OCRed scans. Please send the PDF to the developers for debugging.'
                    )
                    signal.alarm(0)
                    continue
                if gray_scale:
                    img = img.convert('L')
                yield page_ix, np.array(img)
        elif paths:
            print("Processing image batch...")
            for ix in range(len(paths)):
                yield ix, load_image(paths[ix])
        else:
            print("Processing %s..." % self._determine_file_type(file_path))
            yield 0, load_image(file_path)
    def get_num_pages(self, file_path):
        if not self.support_pdfs:
            raise ValueError("File is a PDF though 'support_pdfs' == False")
        pdfbox = PdfBox(file_path)
        page_count = pdfbox.number_of_pages
        return page_count
    def _determine_file_type(self, file_path):
        ext = os.path.splitext(file_path)[1]
        return ext.strip('.').lower()
def user_page_range_str_to_list(page_range, num_pages):
    output = []
    page_range = "".join(page_range.split())
    if page_range == '':
        return list(range(num_pages))
    page_range_list = page_range.split(',')
    for page_range_token in page_range_list:
        if '-' in page_range_token:
            pr_start, pr_end = page_range_token.split('-')
            pr_start = int(pr_start)
            pr_end = num_pages if pr_end.lower() == 'last' else int(pr_end)
            pr_list = list(range(num_pages)[pr_start - 1:pr_end])
            output.extend(pr_list)
        elif page_range_token.lower() == 'last':
            output.append(num_pages - 1)
        else:
            output.append(int(page_range_token) - 1)
    return sorted(list(set(output)))
def java_pid():
    cmd = ["ps", "aux"]
    output = subprocess.check_output(cmd).decode()
    procs = output.split('\n')
    for proc in procs:
        if 'pdfbox' in proc.lower():
            return proc.split()[1]
    return None
def kill_process(pid):
    kill_command = ['kill', '-9', pid]
    subprocess.call(kill_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE)
    return None
def kill_java_process():
    pid = java_pid()
    if pid:
        kill_process(pid)
class FileTypeNotSupported(Exception):
    pass