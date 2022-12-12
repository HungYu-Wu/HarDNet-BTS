import argparse
import os
import pathlib
import random
from datetime import datetime
from types import SimpleNamespace
import torch.nn as nn

import SimpleITK as sitk
import numpy as np
import torch
import torch.optim
import torch.utils.data
import yaml
from torch.cuda.amp import autocast

import models
from dataset import get_datasets
from dataset.batch_utils import pad_batch1_to_compatible_size
from models import get_norm_layer
from tta import apply_simple_tta
from utils import reload_ckpt_bis

parser = argparse.ArgumentParser(description='Brats validation and testing dataset inference')
parser.add_argument('--config', default='', type=str, metavar='PATH',
                    help='path(s) to the trained models config yaml you want to use', nargs="+")
parser.add_argument('--devices', required=True, type=str,
                    help='Set the CUDA_VISIBLE_DEVICES env var from this string')
parser.add_argument('--on', default="train", choices=["train","val","test"])
parser.add_argument('--tta', action="store_true")
parser.add_argument('--seed', default=16111990)


def main(args):
    # setup
    random.seed(args.seed)
    ngpus = torch.cuda.device_count()
    if ngpus == 0:
        raise RuntimeWarning("This will not be able to run on CPU only")
    print(f"Working with {ngpus} GPUs")
    print(args.config)

    current_experiment_time = datetime.now().strftime('%Y%m%d_%T').replace(":", "")
    save_folder = pathlib.Path(f"./preds/{current_experiment_time}")
    save_folder.mkdir(parents=True, exist_ok=True)

    with (save_folder / 'args.txt').open('w') as f:
        print(vars(args), file=f)

    args_list = []
    argsconfig =[]
    #argsconfig.append('runs/20210626_000036__foldFULL_hardnet_48_batch6_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse/20210626_000036__foldFULL_hardnet_48_batch6_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append('runs/20210626_000116__foldFULL_hardnet_48_batch4_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse/20210626_000116__foldFULL_hardnet_48_batch4_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append('runs/20210627_222251__foldFULL_hardnet_48_batch6_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse/20210627_222251__foldFULL_hardnet_48_batch6_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append('runs/20210627_222445__foldFULL_hardnet_48_batch4_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse/20210627_222445__foldFULL_hardnet_48_batch4_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append('runs/20210520_135739__foldFULL_hardnet_48_batch6_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse/20210520_135739__foldFULL_hardnet_48_batch6_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append('runs/20210526_144847__fold0_hardnet_48_batch4_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse/20210526_144847__fold0_hardnet_48_batch4_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append('runs/20210703_235346__foldFULL_hardnet_48_batch6_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse/20210703_235346__foldFULL_hardnet_48_batch6_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse.yaml')
    
    #argsconfig.append('runs/20210627_222445__foldFULL_hardnet_48_batch4_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse/20210627_222445__foldFULL_hardnet_48_batch4_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse.yaml')

    
    
    #argsconfig.append('runs/20210707_225141__fold0_hardnet4_48_batch6_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse/20210707_225141__fold0_hardnet4_48_batch6_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append('runs/20210707_225901__fold1_hardnet4_48_batch6_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse/20210707_225901__fold1_hardnet4_48_batch6_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append('runs/20210707_230111__fold2_hardnet4_48_batch6_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse/20210707_230111__fold2_hardnet4_48_batch6_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append('runs/20210707_230225__fold3_hardnet4_48_batch6_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse/20210707_230225__fold3_hardnet4_48_batch6_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append('runs/20210707_230327__fold4_hardnet4_48_batch6_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse/20210707_230327__fold4_hardnet4_48_batch6_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse.yaml')
    
    #argsconfig.append('runs/20210712_233216__fold0_hardnet4_48_batch6_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse/20210712_233216__fold0_hardnet4_48_batch6_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append('runs/20210712_234018__fold0_hardnet4_48_batch6_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse/20210712_234018__fold0_hardnet4_48_batch6_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append('runs/20210712_234520__fold0_hardnet4_48_batch6_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse/20210712_234520__fold0_hardnet4_48_batch6_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse.yaml')
    
    
    
    #argsconfig.append('runs/20210718_124008__fold0_hardnet4_48_batch11_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse/20210718_124008__fold0_hardnet4_48_batch11_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append('runs/20210726_154943__fold0_hardnet4_48_batch6_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse/20210726_154943__fold0_hardnet4_48_batch6_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse.yaml')
    
    #argsconfig.append('runs/20210728_005809__foldFULL_hardnet4_48_batch10_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse/20210728_005809__foldFULL_hardnet4_48_batch10_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append('runs/20210724_005400__foldFULL_hardnet4_48_batch8_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse/20210724_005400__foldFULL_hardnet4_48_batch8_optimranger_ranger_lr0.0001-wd0.0_epochs1000_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse.yaml')
    
    #argsconfig.append('runs/20210813_150418__foldFULL_hardnet4_48_batch16_optimranger_ranger_lr0.0001-wd0.0_epochs200_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse/20210813_150418__foldFULL_hardnet4_48_batch16_optimranger_ranger_lr0.0001-wd0.0_epochs200_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append('runs/20210813_152120__foldFULL_hardnet4_48_batch13_optimranger_ranger_lr0.0001-wd0.0_epochs200_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse/20210813_152120__foldFULL_hardnet4_48_batch13_optimranger_ranger_lr0.0001-wd0.0_epochs200_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append('runs/20210815_132306__foldFULL_hardnet4_48_batch16_optimranger_ranger_lr0.0001-wd0.0_epochs200_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse/20210815_132306__foldFULL_hardnet4_48_batch16_optimranger_ranger_lr0.0001-wd0.0_epochs200_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append('runs/20210815_133223__foldFULL_hardnet4_48_batch13_optimranger_ranger_lr0.0001-wd0.0_epochs200_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse/20210815_133223__foldFULL_hardnet4_48_batch13_optimranger_ranger_lr0.0001-wd0.0_epochs200_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse.yaml')
    
    #argsconfig.append('runs/20211219_203554__foldFULL_hardnet4_48_batch6_optimranger_ranger_lr0.0001-wd0.0_epochs600_deepsupTrue_fp16_warm0__normgroup_dropout0.0_warm_restartFalse/20211219_203554__foldFULL_hardnet4_48_batch6_optimranger_ranger_lr0.0001-wd0.0_epochs600_deepsupTrue_fp16_warm0__normgroup_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append('runs/20210905_221403__foldFULL_hardnet4_48_batch16_optimranger_ranger_lr0.0001-wd0.0_epochs600_deepsupTrue_fp32_warm0__normgroup_swa5_dropout0.0_warm_restartFalse/4.yaml')
    
    #STORKE
    #argsconfig.append('runs/20220119_163428__fold0_hardnet3_48_batch6_optimadam_adam_lr0.0001-wd0.0_epochs600_deepsupTrue_fp16_warm3__normgroup_dropout0.0_warm_restartFalse/20220119_163428__fold0_hardnet3_48_batch6_optimadam_adam_lr0.0001-wd0.0_epochs600_deepsupTrue_fp16_warm3__normgroup_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append('runs/20220615_120217__fold0_hardnet4_48_batch8_optimranger_ranger_lr0.0001-wd0.0_epochs2000_deepsupTrue_fp16_warm0__normgroup_dropout0.0_warm_restartFalse/20220615_120217__fold0_hardnet4_48_batch8_optimranger_ranger_lr0.0001-wd0.0_epochs2000_deepsupTrue_fp16_warm0__normgroup_dropout0.0_warm_restartFalse.yaml')
    argsconfig.append('runs/20220711_113148__fold0_hardnet4_48_batch8_optimranger_ranger_lr0.0001-wd0.0_epochs2000_deepsupTrue_fp16_warm0__normgroup_dropout0.0_warm_restartFalse/20220711_113148__fold0_hardnet4_48_batch8_optimranger_ranger_lr0.0001-wd0.0_epochs2000_deepsupTrue_fp16_warm0__normgroup_dropout0.0_warm_restartFalse.yaml') 

    for config in argsconfig:
        config_file = pathlib.Path(config).resolve()
        ckpt = config_file.with_name("model_best.pth.tar")
        with config_file.open("r") as file:
            old_args = yaml.safe_load(file)
            old_args = SimpleNamespace(**old_args, ckpt=ckpt)
            # set default normalisation
            if not hasattr(old_args, "normalisation"):
                old_args.normalisation = "minmax"
        args_list.append(old_args)

    if args.on == "test":
        args.pred_folder = save_folder / f"test_segs_tta{args.tta}"
        args.pred_folder.mkdir(exist_ok=True)
    elif args.on == "val":
        args.pred_folder = save_folder / f"validation_segs_tta{args.tta}"
        args.pred_folder.mkdir(exist_ok=True)
    else:
        args.pred_folder = save_folder / f"training_segs_tta{args.tta}"
        args.pred_folder.mkdir(exist_ok=True)

    # Create model

    models_list = []
    normalisations_list = []
    for model_args in args_list:
        print(model_args.arch)
        model_maker = getattr(models, model_args.arch)
        if ngpus >1:
            model = nn.DataParallel(model_maker())
        else :
            model = model_maker()

        reload_ckpt_bis(str(model_args.ckpt), model)
        models_list.append(model)
        normalisations_list.append(model_args.normalisation)
        print("reload best weights")

    dataset_minmax = get_datasets(args.seed, False, no_seg=True,
                                  on=args.on, normalisation="minmax")

    dataset_zscore = get_datasets(args.seed, False, no_seg=True,
                                  on=args.on, normalisation="zscore")

    loader_minmax = torch.utils.data.DataLoader(
        dataset_minmax, batch_size=1, num_workers=2)

    loader_zscore = torch.utils.data.DataLoader(
        dataset_zscore, batch_size=1, num_workers=2)

    print("Val dataset number of batch:", len(loader_minmax))
    generate_segmentations((loader_minmax, loader_zscore), models_list, normalisations_list, args)


def generate_segmentations(data_loaders, models, normalisations, args):
    # TODO: try reuse the function used for train...
    for i, (batch_minmax, batch_zscore) in enumerate(zip(data_loaders[0], data_loaders[1])):
        patient_id = batch_minmax["patient_id"][0]
        ref_img_path = batch_minmax["seg_path"][0]
        crops_idx_minmax = batch_minmax["crop_indexes"]
        crops_idx_zscore = batch_zscore["crop_indexes"]
        inputs_minmax = batch_minmax["image"]
        inputs_zscore = batch_zscore["image"]
        inputs_minmax, pads_minmax = pad_batch1_to_compatible_size(inputs_minmax)
        inputs_zscore, pads_zscore = pad_batch1_to_compatible_size(inputs_zscore)
        model_preds = []
        last_norm = None
        
        for model, normalisation in zip(models, normalisations):
            if normalisation == last_norm:
                pass
            elif normalisation == "minmax":
                inputs = inputs_minmax.cuda()
                pads = pads_minmax
                crops_idx = crops_idx_minmax
            elif normalisation == "zscore":
                inputs = inputs_zscore.cuda()
                pads = pads_zscore
                crops_idx = crops_idx_zscore
            model.cuda()  # go to gpu
            with autocast():
                with torch.no_grad():
                    if args.tta:
                        pre_segs = apply_simple_tta(model, inputs, True)
                        pre_segs1, _ = model(inputs)
                        pre_segs1 = pre_segs1.sigmoid_().cpu()
                    else:
                        if model.deep_supervision:
                            pre_segs, _ = model(inputs)
                        else:
                            pre_segs = model(inputs)
                        pre_segs = pre_segs.sigmoid_().cpu()
                    # remove pads
                    
                    maxz, maxy, maxx = pre_segs.size(2) - pads[0], pre_segs.size(3) - pads[1], pre_segs.size(4) - \
                                       pads[2]
                    pre_segs = pre_segs[:, :, 0:maxz, 0:maxy, 0:maxx].cpu()
                    print("pre_segs size", pre_segs.shape[1])
                    segs = torch.zeros((pre_segs.shape[0],pre_segs.shape[1]-1,pre_segs.shape[2],pre_segs.shape[3],pre_segs.shape[4]))
                    segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = pre_segs[0][0:3]
                    

                    model_preds.append(segs)
                    
            model.cpu()  # free for the next one
        
        pre_segs = torch.stack(model_preds).mean(dim=0)

        segs1 = pre_segs[0].numpy() >0.5
        segs2 = pre_segs[0].numpy() >0.5
        segs3 = pre_segs[0].numpy() >0.5
        et = segs1[0]
        net = np.logical_and(segs2[1], np.logical_not(et))
        ed = np.logical_and(segs3[2], np.logical_not(segs2[1]))
        labelmap = np.zeros(segs1[0].shape)
        labelmap[et] = 4
        labelmap[net] = 1
        labelmap[ed] = 2
        labelmap = sitk.GetImageFromArray(labelmap)
        ref_img = sitk.ReadImage(ref_img_path)
        labelmap.CopyInformation(ref_img)
        
        print(f"Writing {str(args.pred_folder)}/{patient_id}.nii.gz")
        sitk.WriteImage(labelmap, f"{str(args.pred_folder)}/{patient_id}.nii.gz")

if __name__ == '__main__':
    arguments = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    main(arguments)




