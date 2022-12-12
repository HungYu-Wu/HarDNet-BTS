import argparse
import os
import pathlib
import random
from datetime import datetime
from types import SimpleNamespace
import torch.nn as nn
import time
import yaml
from matplotlib import pyplot as plt
from dataset.batch_utils import determinist_collate
from numpy import logical_and as l_and, logical_not as l_not
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

import SimpleITK as sitk
import numpy as np
import torch
import torch.optim
import torch.utils.data
import yaml
from torch.cuda.amp import autocast

import models
from dataset import get_datasets
from dataset.batch_utils import pad_batch1_to_compatible_size, pad_batch_to_max_shape
from models import get_norm_layer
from tta import apply_simple_tta
from utils import save_args, AverageMeter, ProgressMeter, reload_ckpt, save_checkpoint,save_checkpoint1,save_checkpoint2, reload_ckpt_bis, \
    count_parameters, WeightSWA, save_metrics, generate_segmentations
from loss import EDiceLoss
import pprint
from scipy.spatial.distance import directed_hausdorff

parser = argparse.ArgumentParser(description='Brats validation and testing dataset inference')
parser.add_argument('--config', default='', type=str, metavar='PATH',
                    help='path(s) to the trained models config yaml you want to use', nargs="+")
parser.add_argument('--devices', required=True, type=str,
                    help='Set the CUDA_VISIBLE_DEVICES env var from this string')
parser.add_argument('--on', default="val", choices=["train","val","test"])
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

    args_list = []
    argsconfig =[]
    #argsconfig.append('runs/002/20220616_204501__fold0_hardnet4_48_batch16_optimadam_adam_lr0.0001-wd0.0_epochs2000_deepsupFalse_fp16_warm3__normgroup_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append('runs/003/20220616_214100__fold0_hardnet_48_batch60_optimadam_adam_lr0.0001-wd0.0_epochs2000_deepsupFalse_fp16_warm3__normgroup_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append('runs/004/20220616_221011__fold0_hardnet4_48_batch8_optimranger_ranger_lr0.0001-wd0.0_epochs2000_deepsupFalse_fp16_warm0__normgroup_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append('runs/005/20220617_112700__fold0_Unet_48_batch8_optimadam_adam_lr0.0001-wd0.0_epochs2000_deepsupFalse_fp16_warm3__normgroup_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append('runs/006/20220617_113008__fold0_Unet2_48_batch8_optimadam_adam_lr0.0001-wd0.0_epochs2000_deepsupFalse_fp16_warm3__normgroup_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append('runs/007/20220622_145808__fold0_hardnet68_48_batch16_optimadam_adam_lr0.0001-wd0.0_epochs600_deepsupFalse_fp16_warm3__normgroup_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append('runs/008/20220622_154647__fold0_hardnet68_48_batch16_optimadam_adam_lr0.0001-wd0.0_epochs600_deepsupFalse_fp16_warm3__normgroup_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append('runs/009/20220622_155932__fold0_hardnet68_48_batch16_optimadam_adam_lr0.0001-wd0.0_epochs600_deepsupFalse_fp16_warm3__normgroup_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append('runs/10/20220623_180506__fold0_hardnet69_48_batch16_optimadam_adam_lr0.0001-wd0.0_epochs600_deepsupFalse_fp16_warm3__normgroup_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append('runs/11/20220623_180726__fold0_hardnet69_48_batch16_optimadam_adam_lr0.0001-wd0.0_epochs600_deepsupFalse_fp16_warm3__normgroup_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append('runs/12/20220624_105048__fold0_hardnet69_48_batch8_optimranger_ranger_lr0.0001-wd0.0_epochs600_deepsupFalse_fp16_warm0__normgroup_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append('runs/13/20220627_155013__fold0_hardnet4_48_batch16_optimadam_adam_lr0.0001-wd0.0_epochs1200_deepsupFalse_fp16_warm3__normgroup_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append('runs/14/20220629_172627__fold0_hardnet4_48_batch16_optimadam_adam_lr0.0001-wd0.0_epochs1200_deepsupFalse_fp16_warm3__normgroup_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append('runs/16/20220629_183048__fold0_hardnet4_48_batch8_optimranger_ranger_lr0.0001-wd0.0_epochs1200_deepsupFalse_fp16_warm0__normgroup_dropout0.0_warm_restartFalse.yaml')
    #argsconfig.append("runs/17/20220630_220645__fold0_hardnet4_48_batch16_optimranger_ranger_lr0.0001-wd0.0_epochs1200_deepsupFalse_fp16_warm0__normgroup_dropout0.0_warm_restartFalse.yaml")
    #argsconfig.append("runs/022/20220711_114941__fold0_hardnet3_48_batch8_optimranger_ranger_lr0.0001-wd0.0_epochs2000_deepsupTrue_fp16_warm0__normgroup_dropout0.0_warm_restartFalse.yaml")
    #argsconfig.append("runs/023/20220711_115142__fold0_hardnet3_48_batch8_optimranger_ranger_lr0.0001-wd0.0_epochs2000_deepsupTrue_fp16_warm0__normgroup_dropout0.0_warm_restartFalse.yaml")
    #argsconfig.append("runs/024/20220711_115317__fold0_hardnet3_48_batch8_optimranger_ranger_lr0.0001-wd0.0_epochs2000_deepsupTrue_fp16_warm0__normgroup_dropout0.0_warm_restartFalse.yaml")
    #argsconfig.append('runs/18/20220711_113148__fold0_hardnet4_48_batch8_optimranger_ranger_lr0.0001-wd0.0_epochs2000_deepsupTrue_fp16_warm0__normgroup_dropout0.0_warm_restartFalse.yaml')
    argsconfig.append("runs/19/20220711_113922__fold0_hardnet4_48_batch8_optimranger_ranger_lr0.0001-wd0.0_epochs2000_deepsupTrue_fp16_warm0__normgroup_dropout0.0_warm_restartFalse.yaml")
    #argsconfig.append("runs/20/20220711_114244__fold0_hardnet4_48_batch6_optimranger_ranger_lr0.0001-wd0.0_epochs2000_deepsupTrue_fp16_warm0__normgroup_dropout0.0_warm_restartFalse.yaml")
    #argsconfig.append("runs/021/20220711_114451__fold0_hardnet4_48_batch6_optimranger_ranger_lr0.0001-wd0.0_epochs2000_deepsupTrue_fp16_warm0__normgroup_dropout0.0_warm_restartFalse.yaml")
    #argsconfig.append('runs/20220711_113148__fold0_hardnet4_48_batch8_optimranger_ranger_lr0.0001-wd0.0_epochs2000_deepsupTrue_fp16_warm0__normgroup_dropout0.0_warm_restartFalse/20220711_113148__fold0_hardnet4_48_batch8_optimranger_ranger_lr0.0001-wd0.0_epochs2000_deepsupTrue_fp16_warm0__normgroup_dropout0.0_warm_restartFalse.yaml') 
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
        
    # Create model
    
    models_list = []
    normalisations_list = []
    for model_args in args_list:
        print(model_args.arch)
        model_maker = getattr(models, model_args.arch)
        if ngpus >1:
            model = nn.DataParallel(model_maker()).cuda()
        else :
            model = model_maker()
        model.eval()
        reload_ckpt_bis(str(model_args.ckpt), model)
  
        models_list.append(model)
        normalisations_list.append(model_args.normalisation)
        print("reload best weights")
    train_dataset, val_dataset, bench_dataset = get_datasets(16111990, False, fold_number=0)
        
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=True,
            num_workers=2, pin_memory=False, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=max(1, 1)
            , shuffle=False, pin_memory=False, num_workers=2)
       
    bench_loader = torch.utils.data.DataLoader(
            bench_dataset, batch_size=1, num_workers=2)
    dataset_minmax = get_datasets(args.seed, False, no_seg=True,
                                  on=args.on, normalisation="minmax")

    dataset_zscore = get_datasets(args.seed, False, no_seg=True,
                                  on=args.on, normalisation="zscore")

    loader_minmax = torch.utils.data.DataLoader(
        dataset_minmax, batch_size=1, num_workers=2)

    loader_zscore = torch.utils.data.DataLoader(
        dataset_zscore, batch_size=1, num_workers=2)

    print("Val dataset number of batch:", len(loader_minmax))
    criterion = EDiceLoss().cuda()
    metric = criterion.metric
    generate_segmentations1((bench_loader, loader_zscore), models_list, normalisations_list, args, criterion, metric)
    #models_list


def generate_segmentations1(data_loaders, models, normalisations, args, criterion, metric):
    metrics_list = []
    for i, (batch_minmax, batch_zscore) in enumerate(zip(data_loaders[0], data_loaders[1])):
        targets = batch_minmax["label"].cuda(non_blocking=True)
        patient_id = batch_minmax["patient_id"][0]
        ref_img_path = batch_minmax["seg_path"][0]
        ref_path = batch_minmax["seg_path"][0]
        crops_idx_minmax = batch_minmax["crop_indexes"]
        crops_idx_zscore = batch_zscore["crop_indexes"]
        crops_idx = batch_minmax["crop_indexes"]
        crops_idx_zscore = batch_zscore["crop_indexes"]
        inputs_minmax = batch_minmax["image"]
        inputs_zscore = batch_zscore["image"]
        last_norm = None
        model_preds = []
        c=0
        inputs_minmax, pads_minmax = pad_batch1_to_compatible_size(inputs_minmax)
        inputs_zscore, pads_zscore = pad_batch1_to_compatible_size(inputs_zscore)
        #pre_segs = model(inputs)
        
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
            model.cuda()
            with autocast():
                with torch.no_grad():
                    if 1==1:
                        pre_segs, _ = model(inputs)
                    else:
                        pre_segs = model(inputs)
                    pre_segs = torch.sigmoid(pre_segs)
        
                    maxz, maxy, maxx = pre_segs.size(2) - pads[0], pre_segs.size(3) - pads[1], pre_segs.size(4) - pads[2]
                    pre_segs = pre_segs[:, :, 0:maxz, 0:maxy, 0:maxx].cpu()
                    segs = torch.zeros((1, 3, 155, 240, 240))
                    
                    segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = pre_segs[0]
                    model_preds.append(segs)
            model.cpu()
        segs = torch.stack(model_preds).mean(dim=0)
        segs = segs[0].numpy() > 0.5
        et = segs[0]
        net = np.logical_and(segs[1], np.logical_not(et))
        ed = np.logical_and(segs[2], np.logical_not(segs[1]))
        labelmap = np.zeros(segs[0].shape)
        labelmap[et] = 4
        labelmap[net] = 1
        labelmap[ed] = 2
        labelmap = sitk.GetImageFromArray(labelmap)
        ref_seg_img = sitk.ReadImage(ref_path)
        ref_seg = sitk.GetArrayFromImage(ref_seg_img)
        refmap_et, refmap_tc, refmap_wt = [np.zeros_like(ref_seg) for i in range(3)]
        refmap_et = ref_seg == 4
        refmap_tc = np.logical_or(refmap_et, ref_seg == 1)
        refmap_wt = np.logical_or(refmap_tc, ref_seg == 2)
        refmap = np.stack([refmap_et, refmap_tc, refmap_wt])
        patient_metric_list = calculate_metrics(segs, refmap, patient_id)
        metrics_list.append(patient_metric_list)
        labelmap.CopyInformation(ref_seg_img)
    val_metrics = [item for sublist in metrics_list for item in sublist]
    df = pd.DataFrame(val_metrics)
    overlap = df.boxplot(METRICS[1:], by="label", return_type="axes")
    overlap_figure = overlap[0].get_figure()
    haussdorf_figure = df.boxplot(METRICS[0], by="label").get_figure()
    grouped_df = df.groupby("label")[METRICS]
    summary = grouped_df.mean().to_dict()



def calculate_metrics(preds, targets, patient, tta=False):
    """

    Parameters
    ----------
    preds:
        torch tensor of size 1*C*Z*Y*X
    targets:
        torch tensor of same shape
    patient :
        The patient ID
    tta:
        is tta performed for this run
    """
    pp = pprint.PrettyPrinter(indent=4)
    assert preds.shape == targets.shape, "Preds and targets do not have the same size"

    labels = ["ET", "TC", "WT"]

    metrics_list = []

    for i, label in enumerate(labels):
        metrics = dict(
            patient_id=patient,
            label=label,
            tta=tta,
        )
        if np.sum(targets[i]) == 0:
            sens = np.nan
            dice = 1 if np.sum(preds[i]) == 0 else 0
            tn = np.sum(l_and(l_not(preds[i]), l_not(targets[i])))
            fp = np.sum(l_and(preds[i], l_not(targets[i])))
            spec = tn / (tn + fp)
            haussdorf_dist = np.nan

        else:
            preds_coords = np.argwhere(preds[i])
            targets_coords = np.argwhere(targets[i])
            haussdorf_dist = directed_hausdorff(preds_coords, targets_coords)[0]

            tp = np.sum(l_and(preds[i], targets[i]))
            tn = np.sum(l_and(l_not(preds[i]), l_not(targets[i])))
            fp = np.sum(l_and(preds[i], l_not(targets[i])))
            fn = np.sum(l_and(l_not(preds[i]), targets[i]))

            sens = tp / (tp + fn)
            spec = tn / (tn + fp)
            dice = 2 * tp / (2 * tp + fp + fn)
        if str(haussdorf_dist) == 'nan':
            haussdorf_dist = 373
        metrics[HAUSSDORF] = haussdorf_dist
        metrics[DICE] = dice
        metrics[SENS] = sens
        metrics[SPEC] = spec
        metrics_list.append(metrics)
    #print(metrics_list[0]['patient_id'], metrics_list[0]['haussdorf'], metrics_list[1]['haussdorf'],metrics_list[2]['haussdorf'])
    print(metrics_list[0]['patient_id'], metrics_list[0]['dice'], metrics_list[1]['dice'],metrics_list[2]['dice'])
    return metrics_list


HAUSSDORF = "haussdorf"
DICE = "dice"
SENS = "sens"
SPEC = "spec"
METRICS = [HAUSSDORF, DICE, SENS, SPEC]

if __name__ == '__main__':
    arguments = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    main(arguments)
