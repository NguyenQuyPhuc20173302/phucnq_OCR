import matplotlib.pyplot as plt
from PIL import Image
from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import torch

# #train model

config = Cfg.load_config_from_name('vgg_transformer')

config['vocab'] = 'Ẳ\'ẹỄỮỜỡđ́ẢẰỉ$ÍÔếờÃỢÁ8̃ỐỞebUậX,o|ạÊẴƠÌ_}ồAíJỏỰwẳÐƯCKÕỎỂẦữ3ìMx1ấưẠ\u200bŨẶô7(ỚsẸkẽĐZ]4dểàỈÈệỵỴứảềý Úẻ!ỌỗặẬỨỲp;Ừ?tỤộỹ^<Â~\xa0âÓẫẵNụ5ẺỖ@[/Ộăi>#ỶỊĨYIừ̉j°Fqủ:{ễỆúB2"èẨòỔrỀổT6`áẼRọởvÝ*ÒÉịựmãQÀĩfhĂGốHợầẩ̀êằỬyOaPỷõ+ơ&=éóớuử’ẤWắS.LzVẪ9̣Ỹũ0g%ẾÙỦỒẮ)ElỠùc-n\\ỳD'
home = "/data/phucnq/data/OCR/"
dataset_params = {
    'name':'hw',
    'data_root': "/data/phucnq/data/OCR/data/dataLine/InkData_line_processed",
    'train_annotation':'/data/phucnq/data/OCR/data/data_line/train.txt',
    'valid_annotation':'/data/phucnq/data/OCR/data/data_line/test.txt'
}

params = {
         'print_every':10,
         'valid_every':10,
          'iters':500000,
          'checkpoint': home + 'weights/transformerocr_checkpoint.pth',
          'export': home + 'weights/transformerocr_train.pth',
          'metrics': 10000,
          'log': home + 'weights/train.log'
         }

config['trainer'].update(params)
config['dataset'].update(dataset_params)
config['device'] = 'cuda:1'
config['weights'] = home + "weights/vgg_transformer.pth"

with torch.cuda.device(1):
    trainer = Trainer(config, pretrained=True)


trainer.config.save('config.yml')


with torch.cuda.device(1):
    trainer.train()


with torch.cuda.device(1):
    trainer.precision()


import torch
torch.cuda.empty_cache()


