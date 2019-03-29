
from os.path import join as pjoin
import os
from datetime import datetime
import time

from fastai.vision import ImageDataBunch,ImageList,get_transforms,models,cnn_learner,accuracy,imagenet_stats
from fastai.vision import ShowGraph,partial
from pathlib import Path
import torch


start = time.time()
# vars for models logging
TRAIN_LOG_DIR = 'train_log'
#NB_NAME = 'densenet-fastai.ipynb'
NB_NAME = 'resnet18-fastai.ipynb'
MODEL_NAME = NB_NAME.split('.')[0]

os.makedirs(pjoin(TRAIN_LOG_DIR,MODEL_NAME),exist_ok=True)

find_version = lambda x : int(x.split('.')[0]) if len(x.split('.')[0]) >0 else 0
list_versions = sorted(list(map(find_version,os.listdir(pjoin(TRAIN_LOG_DIR,MODEL_NAME)))))
version = list_versions[-1] + 1 if len(list_versions)>0 else 0

date = datetime.today().strftime('%d-%m-%Y-%H-%M')
save_folder = f"{version:0>3d}" +'.'+date
model_save_dir = pjoin(TRAIN_LOG_DIR,MODEL_NAME,save_folder)
os.makedirs(model_save_dir,exist_ok=True)
print(model_save_dir)

NOTE = f"""
Note for version {version} model {MODEL_NAME}:
Train for a long time
"""
print(NOTE)

data_path = '/home/qnkhuat/data/emotion_compilation_split'
tfms = get_transforms(do_flip=True,
                      flip_vert=False,
                      max_rotate=20,
                      max_zoom=1.1,
                     )
# ran this get erro in THcroe
data = (ImageDataBunch.from_folder(data_path,test='test',size=48,ds_tfms=tfms,bs=256).normalize(imagenet_stats))
print(data)



model = models.resnet18
#model = models.densenet121

learn = cnn_learner(data, model,callback_fns=[ShowGraph])
### THE DIRECTORY TO SAVE CHECKPOINTS
learn.path = Path(model_save_dir)
learn.model_dir = model_save_dir
learn.metrics = [accuracy]



### START TRAINING
lr=5e-2
learn.fit_one_cycle(12,max_lr = slice(1e-4,lr))

# Unfreeze
learn.unfreeze()
learn.fit(15)
# Refereeze
learn.fit_one_cycle(9,slice(1e-6,1e-3))
tta = accuracy(*learn.TTA()).item()*100
print('Done round 1')
print(tta)


print('Start Round 2')
data_path = '/home/qnkhuat/data/emotion_compilation_split'
tfms = get_transforms(do_flip=True,
                      flip_vert=False,
                      max_rotate=20,
                      max_zoom=1.1,
                     )
# ran this get erro in THcroe
data = (ImageDataBunch.from_folder(data_path,test='test',size=96,ds_tfms=tfms,bs=256).normalize(imagenet_stats))

learn.data = data
lr = 1e-2


learn.fit_one_cycle(12,max_lr = slice(1e-4,lr))

# Unfreeze
learn.unfreeze()
learn.fit(15)
# Refereeze
learn.fit_one_cycle(9,slice(1e-6,1e-3))
tta = accuracy(*learn.TTA()).item()*100
print('Done round 2')
print(tta)
learn.export(f'{MODEL_NAME}.pkl')

print(f'Total time {time.time() - start}')

