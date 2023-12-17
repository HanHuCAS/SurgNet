import os
import os.path as osp
from .builder import DATASETS
from .custom import CustomDataset
import os.path as osp
import imgviz
from functools import reduce
import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
from torch.utils.data import Dataset
from mmseg.core import eval_metrics
from mmseg.utils import get_root_logger
from .pipelines import Compose
from PIL import Image
import numpy as np
@DATASETS.register_module()
class PascalVOCDataset(CustomDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor')

    PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
               [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
               [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
               [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]

    def __init__(self, split, **kwargs):
        super(PascalVOCDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', split=split, **kwargs)
        
        assert osp.exists(self.img_dir) and self.split is not None

@DATASETS.register_module()
class LMY1800VOCDataset(CustomDataset):
    """LMY1800VOC.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    def __init__(self, split, **kwargs):
        super(LMY1800VOCDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', split=split, **kwargs)
        classes_ = []
        for i, line in enumerate(open(osp.join(self.data_root,'class_names.txt')).readlines()):
            class_name_ = line.strip()
            classes_.append(class_name_)
       
        classes_ = tuple(classes_)
        palette_ = imgviz.label_colormap(len(classes_))
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes_, palette_.tolist())
        assert osp.exists(self.img_dir) and self.split is not None
    """def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps(efficient_test)
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)
        for i in range(len(self.img_infos)):
            metrics = eval_metrics(
            [results[i]],
            [gt_seg_maps[i]],
            num_classes,
            self.ignore_index,
            metric,
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label)
            #print(metrics[2][1])
            #print(osp.join(self.data_root,self.ann_dir,self.img_infos[i]['ann']['seg_map']))
            #os.rename ('results/' + self.img_infos[i]['filename'], 'results/' + self.img_infos[i]['filename'][:-4] + '_IoU_' + str(int(metrics[2][1] * 100)) + '.jpg')
            img_result = Image.open('results/' + self.img_infos[i]['filename'])
            img_target = Image.open(osp.join(self.data_root,'SegmentationClassVisualization',self.img_infos[i]['filename']))
            result = Image.new(img_result.mode, (img_result.width, img_result.height + img_target.height))
            result.paste(img_result, box=(0, 0 * img_result.height))
            result.paste(img_target, box=(0, 1 * img_result.height))
            if np.isnan(metrics[2][1]):
                result.save('results/merge_' + self.img_infos[i]['filename'][:-4] + '_IoU_' + str(0) + '.jpg')
            else:
                result.save('results/merge_' + self.img_infos[i]['filename'][:-4] + '_IoU_' + str(int(metrics[2][1] * 100)) + '.jpg')

        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metric,
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label)
        print(ret_metrics)
        class_table_data = [['Class'] + [m[1:] for m in metric] + ['Acc']]
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES
        ret_metrics_round = [
            np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
        ]
        for i in range(num_classes):
            class_table_data.append([class_names[i]] +
                                    [m[i] for m in ret_metrics_round[2:]] +
                                    [ret_metrics_round[1][i]])
        summary_table_data = [['Scope'] +
                              ['m' + head
                               for head in class_table_data[0][1:]] + ['aAcc']]
        ret_metrics_mean = [
            np.round(np.nanmean(ret_metric) * 100, 2)
            for ret_metric in ret_metrics
        ]
        summary_table_data.append(['global'] + ret_metrics_mean[2:] +
                                  [ret_metrics_mean[1]] +
                                  [ret_metrics_mean[0]])
        print_log('per class results:', logger)
        table = AsciiTable(class_table_data)
        print_log('\n' + table.table, logger=logger)
        print_log('Summary:', logger)
        table = AsciiTable(summary_table_data)
        print_log('\n' + table.table, logger=logger)

        for i in range(1, len(summary_table_data[0])):
            eval_results[summary_table_data[0]
                         [i]] = summary_table_data[1][i] / 100.0
        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
        return eval_results"""
@DATASETS.register_module()
class CholecSeg8kVOCDataset(CustomDataset):
    """CholecSeg8kVOCDataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    """CLASSES = ('_background_', 'AW', 'Liver', 'GT', 'Fat', 'Grasper',
               'CT', 'Blood', 'CD', 'Hook', 'Gallbladder', 'HV', 'LL')#13
    
    PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
               [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128]]"""

    def __init__(self, split, **kwargs):
        super(CholecSeg8kVOCDataset, self).__init__(
            img_suffix='.png', seg_map_suffix='.png', split=split, **kwargs)
        classes_ = ('_background_', 'AW', 'Liver', 'GT', 'Fat', 'Gallbladder',
               'Misc', 'Ins')#8
        palette_ = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128]]
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes_, palette_)
        #self.label_map = {0:0,1:1,2:2,3:3,4:4,5:7,6:6,7:6,8:6,9:7,10:5,11:6,12:6}
        re_label = {0:0,1:1,2:2,3:3,4:4,5:7,6:6,7:6,8:6,9:7,10:5,11:6,12:6}
        label_map = {50:0,11:1,21:2,13:3,12:4,31:5,23:6,24:7,25:8,32:9,22:10,33:11,5:12}
        self.label_map = {}
        for key in label_map.keys():
            self.label_map[key] = re_label[label_map[key]]
        #self.custom_classes = True
        assert osp.exists(self.img_dir) and self.split is not None

@DATASETS.register_module()
class EndoVisSub2017VOCDataset(CustomDataset):
    """CholecSeg8kVOCDataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    """CLASSES = ('_background_', 'AW', 'Liver', 'GT', 'Fat', 'Grasper',
               'CT', 'Blood', 'CD', 'Hook', 'Gallbladder', 'HV', 'LL')#13
    
    PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
               [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128]]"""

    def __init__(self, split, **kwargs):
        super(EndoVisSub2017VOCDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', split=split, **kwargs)
        classes_ = ('_background_', 'Bipolar_Forceps', 'Prograsp_Forceps', 'Large_Needle_Driver', 'Vessel_Sealer', 'Grasping_Retractor', 'Monopolar_Curved_Scissors', 'Other')#8
        palette_ = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128]]
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes_, palette_)
        #self.label_map = {0:0,32:1,64:2,96:3,128:4,160:5,192:6,224:7}
        """re_label = {0:0,1:1,2:2,3:3,4:4,5:7,6:6,7:6,8:6,9:7,10:5,11:6,12:6}
        label_map = {50:0,11:1,21:2,13:3,12:4,31:5,23:6,24:7,25:8,32:9,22:10,33:11,5:12}
        self.label_map = {}
        for key in label_map.keys():
            self.label_map[key] = re_label[label_map[key]]"""
        #self.custom_classes = True
        assert osp.exists(self.img_dir) and self.split is not None
