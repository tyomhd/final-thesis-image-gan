import xlrd
import numpy as np
import math
import matplotlib.pyplot as plt

def plot_loss_excel():

    # Matplotlib settings
    SMALL_SIZE = 16
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # For plot
    list_epoch_plot = []
    list_d_loss_fake_plot = []
    list_d_loss_real_plot = []
    list_g_loss_mse_vgg_plot = []
    list_g_loss_cxent_plot = []

    list_d_combined_loss = []
    list_g_combined_loss = []

    list_minmax = []
    list_minmax_goal = []
    list_minmax_d_wins = []
    list_minmax_g_wins = []

    excel_workbook = xlrd.open_workbook('D:/Project/itc-final-thesis-gan/data_final/training/final-16-50-22to24batch/loss-16_50_batch24-fixed.xls') 
    excel_sheet = excel_workbook.sheet_by_index(0)
    
    print(f'num rows - {excel_sheet.nrows}')
    print(f'num columns - {excel_sheet.ncols}')


    for i in range(excel_sheet.nrows - 1): 
        epoch = excel_sheet.cell_value(i+1, 0)
        list_epoch_plot.append(epoch/2140)

        d_loss_fake = excel_sheet.cell_value(i+1, 1)
        d_loss_real = excel_sheet.cell_value(i+1, 2)
        g_loss_mse_vgg = excel_sheet.cell_value(i+1, 4)
        g_loss_cxent = excel_sheet.cell_value(i+1, 5)

        minmax_d_wins = 2 **(-d_loss_real)
        list_minmax_d_wins.append(minmax_d_wins)

        minmax_g_wins = 2 ** (-g_loss_cxent)
        list_minmax_g_wins.append(minmax_g_wins)
        minmax = -d_loss_real + math.log2(1 -(2 ** (-g_loss_cxent)))
        list_minmax.append(minmax)


        list_minmax_goal.append(-2)

        list_d_loss_fake_plot.append(d_loss_fake)
        list_d_loss_real_plot.append(d_loss_real)
        list_g_loss_mse_vgg_plot.append(g_loss_mse_vgg)
        list_g_loss_cxent_plot.append(g_loss_cxent)

        list_d_combined_loss.append((d_loss_fake + d_loss_real)/2)
        list_g_combined_loss.append(g_loss_mse_vgg + 1e-3*g_loss_cxent)

    # Print plot
    # plt.plot(list_epoch_plot, list_d_loss_fake_plot, 'b', label='d fake')
    # plt.plot(list_epoch_plot, list_d_loss_real_plot, 'r', label='d real')
    # plt.plot(list_epoch_plot, list_g_loss_mse_vgg_plot, 'g', label='g mse vgg')
    # plt.plot(list_epoch_plot, list_g_loss_cxent_plot, 'b', label='g cxent')

    plt.plot(list_epoch_plot, list_minmax, 'y', label='MinMax Game')
    plt.plot(list_epoch_plot, list_minmax_d_wins, 'b', label='Discriminator prediction on real data')
    plt.plot(list_epoch_plot, list_minmax_g_wins, 'r', label='Discriminator prediction on generated data')
    plt.plot(list_epoch_plot, list_minmax_goal, 'k', label='2 x log(0.5)')

    # plt.plot(list_epoch_plot, list_d_combined_loss, 'b', label='Discriminator total loss')
    # plt.plot(list_epoch_plot, list_g_combined_loss, 'r', label='Generator total loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')
    
    plt.legend(loc='cubic loss')
    plt.show()

if __name__ == '__main__':
    plot_loss_excel()
