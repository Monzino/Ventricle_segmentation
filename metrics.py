import os
from glob import glob
import re
import argparse
import pandas as pd
import numpy as np

import scipy.stats as stats
import utils
import binary_metric as bm
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import logging

def conv_int(i):
    return int(i) if i.isdigit() else i


def natural_order(sord):
    """
    Sort a (list,tuple) of strings into natural order.
    Ex:
    ['1','10','2'] -> ['1','2','10']
    ['abc1def','ab10d','b2c','ab1d'] -> ['ab1d','ab10d', 'abc1def', 'b2c']
    """
    if isinstance(sord, tuple):
        sord = sord[0]
    return [conv_int(c) for c in re.split(r'(\d+)', sord)]


def print_latex_tables(df, eval_dir):
    """
    Report geometric measures in latex tables to be used in the ACDC challenge paper.
    Prints mean (+- std) values for Dice for all structures.
    :param df:
    :param eval_dir:
    :return:
    """

    out_file = os.path.join(eval_dir, 'latex_tables.txt')

    with open(out_file, "w") as text_file:

        text_file.write('\n\n-------------------------------------------------------------------------------------\n')
        text_file.write('table 1\n')
        text_file.write('-------------------------------------------------------------------------------------\n\n')
        # prints mean (+- std) values for Dice, all structures, averaged over both phases.

        header_string = ' & '
        line_string = 'METHOD '


        for s_idx, struc_name in enumerate(['LV', 'RV', 'Myo']):
            for measure in ['dice']:

                header_string += ' & {} ({}) '.format(measure, struc_name)

                dat = df.loc[df['struc'] == struc_name]

                if measure == 'dice':
                    line_string += ' & {:.3f}\,({:.3f}) '.format(np.mean(dat[measure]), np.std(dat[measure]))
                else:
                    line_string += ' & {:.2f}\,({:.2f}) '.format(np.mean(dat[measure]), np.std(dat[measure]))

            if s_idx < 2:
                header_string += ' & '
                line_string += ' & '

        header_string += ' \\\\ \n'
        line_string += ' \\\\ \n'

        text_file.write(header_string)
        text_file.write(line_string)


        text_file.write('\n\n-------------------------------------------------------------------------------------\n')
        text_file.write('table 2\n')
        text_file.write('-------------------------------------------------------------------------------------\n\n')
        # table 2: mean (+- std) values for Dice and HD, all structures, both phases separately


        for idx, struc_name in enumerate(['LV', 'RV', 'Myo']):
            # new line
            header_string = ' & '
            line_string = '({}) '.format(struc_name)

            for p_idx, phase in enumerate(['ED', 'ES']):
                for measure in ['dice', 'hd']:

                    header_string += ' & {} ({}) '.format(phase, measure)

                    dat = df.loc[(df['phase'] == phase) & (df['struc'] == struc_name)]

                    if measure == 'dice':
                        line_string += ' & {:.3f}\,({:.3f}) '.format(np.mean(dat[measure]), np.std(dat[measure]))
                    else:
                        line_string += ' & {:.2f}\,({:.2f}) '.format(np.mean(dat[measure]), np.std(dat[measure]))

                if p_idx == 0:
                    header_string += ' & '
                    line_string += ' & '

            header_string += ' \\\\ \n'
            line_string += ' \\\\ \n'

            if idx == 0:
                text_file.write(header_string)

            text_file.write(line_string)

    return 0


def compute_metrics_on_directories_raw(dir_gt, dir_pred):
    '''
    - Dice
    - Hausdorff distance
    - Average surface distance
    - Predicted volume
    - Volume error w.r.t. ground truth
    :param dir_gt: Directory of the ground truth segmentation maps.
    :param dir_pred: Directory of the predicted segmentation maps.
    :return: Pandas dataframe with all measures in a row for each prediction and each structure
    '''
    
    cardiac_phase = []
    file_names = []
    structure_names = []

    # measures per structure:
    dices_list = []
    hausdorff_list = []
    vol_list = []
    vol_err_list = []
    vol_gt_list = []
    
    structures_dict = {1: 'RV', 2: 'Myo', 3: 'LV'}
    
    for p_gt, p_pred in zip(sorted(os.listdir(dir_gt)), sorted(os.listdir(dir_pred))):
        if (p_gt != p_pred):
            raise ValueError("The two patients don't have the same name"
                             " {}, {}.".format(p_gt, p_pred))
        dir_p_gt = os.path.join(dir_gt, p_gt)
        dir_p_pred = os.path.join(dir_pred, p_pred)
        
        print(p_gt)
        
        for phase_gt, phase_pred in zip(sorted(os.listdir(dir_p_gt)), sorted(os.listdir(dir_p_pred))):
            if (phase_gt != phase_pred):
                raise ValueError("The two phases don't have the same name"
                                 " {}, {}.".format(phase_gt, phase_pred))
            dir_ph_gt = os.path.join(dir_p_gt, phase_gt)
            dir_ph_pred = os.path.join(dir_p_pred, phase_pred)
            
            pred_arr = []
            mask_arr = []
            for img_gt, img_pred in zip(sorted(glob.glob(os.path.join(dir_ph_gt, '*.png'))), sorted(glob.glob(os.path.join(dir_ph_pred, '*.png')))):
                if (img_gt.split(phase_gt + '/')[1] != img_pred.split(phase_pred + '/')[1]):
                    raise ValueError("The two images don't have the same name"
                                     " {}, {}.".format(img_gt, img_pred))
                    
                gt = cv2.imread(img_gt,0)
                pred = cv2.imread(img_pred,0)
                if (gt.shape != pred.shape):
                    raise ValueError("The two images don't have the same shape"
                                     " {}, {}.".format(gt.shape, pred.shape))
                
                pred_arr.append(pred)
                mask_arr.append(gt)

            pred_arr = np.transpose(np.asarray(pred_arr, dtype=np.uint8), (1,2,0))
            mask_arr = np.transpose(np.asarray(mask_arr, dtype=np.uint8), (1,2,0))
            print(pred_arr.shape)
            
            for struc in [3,1,2]:
                gt_binary = (mask_arr == struc) * 1
                pred_binary = (pred_arr == struc) * 1

                volpred = pred_binary.sum() * config.z_dim / 1000.
                volgt = gt_binary.sum() * config.z_dim / 1000.
            
                vol_list.append(volpred)
                vol_err_list.append(volpred - volgt)
                vol_gt_list.append(volgt)
                
                if np.sum(gt_binary) == 0 and np.sum(pred_binary) == 0:
                    dices_list.append(1)
                    hausdorff_list.append(0)
                elif np.sum(pred_binary) > 0 and np.sum(gt_binary) == 0 or np.sum(pred_binary) == 0 and np.sum(gt_binary) > 0:
                    logging.warning('Structure missing in either GT (x)or prediction. HD will not be accurate.')
                    dices_list.append(0)
                    hausdorff_list.append(1)
                else:
                    hausdorff_list.append(bm.hd(gt_binary, pred_binary, connectivity=1))
                    dices_list.append(bm.dc(gt_binary, pred_binary))

                cardiac_phase.append(phase_pred)
                file_names.append(p_pred)
                structure_names.append(structures_dict[struc])

    df = pd.DataFrame({'dice': dices_list, 'hd': hausdorff_list,
                       'vol': vol_list, 'vol_gt': vol_gt_list, 'vol_err': vol_err_list,
                      'phase': cardiac_phase, 'struc': structure_names, 'filename': file_names})
    
    return df
    

def print_stats(df, eval_dir):
    
    out_file = os.path.join(eval_dir, 'summary_report.txt')
    
    with open(out_file, "w") as text_file:

        text_file.write('\n\n-------------------------------------------------------------------------------------\n')
        text_file.write('Summary of geometric evaluation measures. \n')
        text_file.write('-------------------------------------------------------------------------------------\n\n')

        for struc_name in ['LV', 'RV', 'Myo']:

            text_file.write(struc_name)
            text_file.write('\n')

            for cardiac_phase in ['ED', 'ES']:

                text_file.write('    {}\n'.format(cardiac_phase))

                dat = df.loc[(df['phase'] == cardiac_phase) & (df['struc'] == struc_name)]

                for measure_name in ['dice', 'hd']:

                    text_file.write('       {} -- mean (std): {:.3f} ({:.3f}) \n'.format(measure_name,
                                                                         np.mean(dat[measure_name]), np.std(dat[measure_name])))

                    ind_med = np.argsort(dat[measure_name]).iloc[len(dat[measure_name])//2]
                    text_file.write('             median {}: {:.3f} ({})\n'.format(measure_name,
                                                                dat[measure_name].iloc[ind_med], dat['filename'].iloc[ind_med]))

                    ind_worst = np.argsort(dat[measure_name]).iloc[0]
                    text_file.write('             worst {}: {:.3f} ({})\n'.format(measure_name,
                                                                dat[measure_name].iloc[ind_worst], dat['filename'].iloc[ind_worst]))

                    ind_best = np.argsort(dat[measure_name]).iloc[-1]
                    text_file.write('             best {}: {:.3f} ({})\n'.format(measure_name,
                                                                dat[measure_name].iloc[ind_best], dat['filename'].iloc[ind_best]))


        text_file.write('\n\n-------------------------------------------------------------------------------------\n')
        text_file.write('Ejection fraction correlation between prediction and ground truth\n')
        text_file.write('-------------------------------------------------------------------------------------\n\n')

        for struc_name in ['LV', 'RV']:

            lv = df.loc[df['struc'] == struc_name]

            ED_vol = np.array(lv.loc[lv['phase'] == 'ED']['vol'])
            ES_vol = np.array(lv.loc[(lv['phase'] == 'ES')]['vol'])
            EF_pred = (ED_vol - ES_vol) / ED_vol

            ED_vol_gt = ED_vol - np.array(lv.loc[lv['phase'] == 'ED']['vol_err'])
            ES_vol_gt = ES_vol - np.array(lv.loc[(lv['phase'] == 'ES')]['vol_err'])

            EF_gt = (ED_vol_gt - ES_vol_gt) / ED_vol_gt

            LV_EF_corr = stats.pearsonr(EF_pred, EF_gt)
            text_file.write('{}, EF corr: {}\n\n'.format(struc_name, LV_EF_corr[0]))


def boxplot_metrics(df, eval_dir):
    """
    Create summary boxplots of all geometric measures.
    :param df:
    :param eval_dir:
    :return:
    """

    boxplots_file = os.path.join(eval_dir, 'boxplots.eps')

    fig, axes = plt.subplots(2, 1)
    fig.set_figheight(14)
    fig.set_figwidth(7)

    sns.boxplot(x='struc', y='dice', hue='phase', data=df, palette="PRGn", ax=axes[0])
    sns.boxplot(x='struc', y='hd', hue='phase', data=df, palette="PRGn", ax=axes[1])
    
    plt.savefig(boxplots_file)
    plt.close()

    return 0
    

def main(path_pred, path_gt, eval_dir):
    logging.info(path_gt)
    logging.info(path_pred)
    logging.info(eval_dir)
    
    if os.path.isdir(path_gt) and os.path.isdir(path_pred):
        
        df = compute_metrics_on_directories_raw(path_gt, path_pred)
        
        print_stats(df, eval_dir)
        print_latex_tables(df, eval_dir)
        boxplot_metrics(df, eval_dir)

        logging.info('------------Average Dice Figures----------')
        logging.info('Dice 1: %f' % np.mean(df.loc[df['struc'] == 'LV']['dice']))
        logging.info('Dice 2: %f' % np.mean(df.loc[df['struc'] == 'RV']['dice']))
        logging.info('Dice 3: %f' % np.mean(df.loc[df['struc'] == 'Myo']['dice']))
        logging.info('Mean dice: %f' % np.mean(np.mean(df['dice'])))
        logging.info('------------------------------------------')
    
    else:
        raise ValueError(
            "The paths given needs to be two directories or two files.")
