import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plots.data_info import DataInfo

b_idx = 0
k_idx = 1

def _load_data_and_plot(idx_figure, interval, signal_max, signal_min, files):
    for ind_index in interval:
        # file_name = "forceLog_" + f'{ind_index:04}' + ".dat"

        data = pd.read_csv(folder_name + files[ind_index])
        signal = data["tau [Nm]"]

        if max(signal) > signal_max:
            signal_max = max(signal)
        if min(signal) < signal_min:
            signal_min = min(signal)

        plt.figure(idx_figure)
        time = list(range(len(signal)))
        plt.plot(time, signal)
        plt.ylim([-0.5, 0.5])
        plt.xlim([0, 600])

    return signal_max, signal_min


def ga_plot_forces(n_individuals, n_generations, files, opt_type, name, signal_type='torque', save_fig=False):

    interval_1st = range(0, n_individuals[0])
    idx_first = 10
    # interval_last = range(n_individuals[-2] * (n_generations-1), n_individuals[-1] * n_generations)
    interval_last = range(np.sum(np.array(n_individuals[:n_generations-1])), np.sum(np.array(n_individuals[:n_generations])))
    idx_last = 20

    max_signal = 0.0
    min_signal = 0.0

    max_signal, min_signal = _load_data_and_plot(idx_figure=idx_first, interval=interval_1st,
                                                 signal_max=max_signal, signal_min=min_signal, files=files)
    max_signal, min_signal = _load_data_and_plot(idx_figure=idx_last, interval=interval_last,
                                                 signal_max=max_signal, signal_min=min_signal, files=files)

    # TODO: implement max and min scaling for same y-axis length
    # scaling_factor = 1.2
    # ax[0, 0].set(xlim=(time[0], time[-1]), ylim=(-100, 200))
    # ax.set(xlim=(time[0], time[-1]), ylim=(-100, 200))
    # ax1.set(xlim=(time[0], time[-1]), ylim=(min_f * scaling_factor, max_f * scaling_factor))
    # ax2.set(xlim=(time[0], time[-1]), ylim=(min_f * scaling_factor, max_f * scaling_factor))
    # fig1.tight_layout()
    # fig2.tight_layout()

    # plt.figure(10 + 1)
    # # ax[0, 1].set_ylabel("$F_z$ [N]")
    # plt.xlabel("Time [s]", fontsize='18')
    # plt.ylabel("$F_z$ [N]", fontsize='18')
    # aux = plt.gca().get_ylim()
    # plt.xticks([0, 10, 20, 30, 40], fontsize='15')
    # plt.yticks([-100, -50, 0, 50, 100, 150], fontsize='15')
    # plt.tight_layout()
    # plt.grid(True)

    if signal_type == 'torque':
        y_label = '$\\tau$ [Nm]'
    else:
        y_label = '$F$ [N]'

    plt.figure(idx_first)
    plt.xlabel("Time [s]", fontsize='18')
    plt.ylabel(y_label, fontsize='18')
    plt.tight_layout()
    plt.grid(True)

    plt.figure(idx_last)
    plt.xlabel("Time [s]", fontsize='18')
    plt.ylabel(y_label, fontsize='18')
    plt.tight_layout()
    plt.grid(True)

    if save_fig:
        plt.figure(idx_first)
        plt.savefig('./plots/' + signal_type + '_'+ name +'_'+ opt_type + '_first_gen.png')
        plt.figure(idx_last)
        plt.savefig('./plots/' + signal_type + '_'+ name +'_'+ opt_type +  '_last_gen.png')
    else:
        plt.show()


def _plot_pareto(metric_x, metric_y, s_first, s_last, idx_fig, xlabel='', ylabel=''):
    plt.figure(idx_fig)
    plt.plot(metric_x[s_first], metric_y[s_first], 'ro')
    plt.plot(metric_x[s_last], metric_y[s_last], 'k^')
    plt.xlabel(xlabel, fontsize='18')
    plt.ylabel(ylabel, fontsize='18')
    plt.legend(['1st Gen.', 'Last Gen.'])
    # plt.xscale('log')
    # plt.yscale('log')
    plt.ylim([0, 6])
    plt.xlim([0, .2])
    plt.grid()


def ga_plot_pareto(n_generations, n_individuals, metrics, opt_type, name, save_fig=False, used_error_ss=False):

    s_first = slice(0, n_individuals[0] + 1)
    s_last = slice(n_individuals[-2] * (n_generations-1) + 1, n_individuals[-1] * n_generations + 1)

    idx_1 = 0
    idx_2 = 1
    # idx_tv_ts = 2

    if used_error_ss:
        third_metric = '$e_{ss}$ [rad]'
    else:
        third_metric = '$t_s$ [s]'

    _plot_pareto(metric_x=metrics[:,0], metric_y=metrics[:,1], s_first=s_first, s_last=s_last, idx_fig=idx_1,
                 xlabel='$tau_{RMS}$ [Nm]', ylabel='$time_{max}$ [s]')
    # _plot_pareto(metric_x=tr, metric_y=ts, s_first=s_first, s_last=s_last, idx_fig=idx_2,
    #              xlabel='$t_r$ [s]', ylabel=third_metric)
    # _plot_pareto(metric_x=tv, metric_y=ts, s_first=s_first, s_last=s_last, idx_fig=idx_tv_ts,
    #              xlabel='$tv$ [N]', ylabel=third_metric)

    # plt.figure(idx_tr_tv)
    # plt.plot(tr[s_first], tv[s_first], 'ro')
    # plt.plot(tr[s_final], tv[s_final], 'k^')
    # # plt.xscale('log')
    # # plt.yscale('log')
    # plt.grid()
    #
    # plt.figure(idx_tr_tv)
    # plt.plot(tr[s_first], tv[s_first], 'ro')
    # plt.plot(tr[s_final], tv[s_final], 'k^')
    # # plt.xscale('log')
    # # plt.yscale('log')
    # plt.grid()

    if save_fig:
        plt.figure(idx_1)
        plt.savefig('./plots/pareto_' + name +'_'+ opt_type +'.png')
        # plt.figure(idx_2)
        # plt.savefig('./images/pareto_tr_ts.png')
        # plt.figure(idx_tv_ts)
        # plt.savefig('./images/pareto_tv_ts.png')
    else:
        plt.show()


# def OLD_ga_plot_pareto(gen_first=1, n_generations=10, n_individuals=20, save_fig=False):
#     file_name = "fitnessLog.dat"
#     # fitness_log = [x.split('\t') for x in open(folder_name+file_name).readlines()]
#     # fitness_log = [[float(number) for number in item] for item in fitness_log]
#     fitness_log = np.loadtxt(folder_name + file_name)
#
#     b, k, tv, tr = fitness_log.T
#
#     # b = [z[0] for z in fitness_log]
#     # k = [z[1] for z in fitness_log]
#     # kxy = [z[2] for z in fitness_log]
#     # metrics_x = [z[2:5] for z in fitness_log]
#     metrics_x = [tv, tr]
#     # metrics_z = [z[9:12] for z in fitness_log]
#     # m = [z[6] for z in fitness_log]
#     # b = [z[7] for z in fitness_log]
#     # k = [z[8] for z in fitness_log]
#
#     # fig, ax = plt.subplots(max_gen - starting_gen + 1, 3, sharex='row', sharey='row')
#     # fig, ax = plt.subplots(2, 3, sharex='row', sharey='row')
#
#     fig = [plt.figure(i, figsize=(6, 6)) for i in range(6)]
#
#     ax = []
#
#     for f in fig:
#         ax.append(f.add_axes([.1, .1, .8, .8]))
#
#     # ax = np.reshape(ax, (3, max_gen - starting_gen + 1))
#
#     for index in range(n_generations):
#
#         s_first = slice(0, n_individuals + 1)
#         # s_init = slice((gen_first+index)*n_individuals+1, (gen_last+index+1)*n_individuals+1)
#         s_final = slice(n_individuals * n_generations + 1, n_individuals * (n_generations + 1) + 1)
#
#         # fig, ax = plt.subplots()
#         # fig1, ax1 = plt.subplots()
#         # fig2, ax2 = plt.subplots()
#         plot_ess_tv = []
#         plot_ts_tv = []
#         plot_ts_ess = []
#         color_change = index/(n_generations - gen_first)
#
#         if index == 0:
#             for tvxy, trxy, essxy in metrics_x[s_first]:
#                 plot_ess_tv += ax[index].plot(essxy, tvxy, 'x', color=(0.1, 0.1, 0.1/2), markersize=10)
#                 ax[index].set_yscale('log')
#                 ax[index + 1].plot(trxy, tvxy, 'x', color=(0.1, 0.1, 0.1 / 2), markersize=10)
#                 ax[index + 1].set_yscale('log')
#                 ax[index + 2].plot(trxy, essxy, 'x', color=(0.1, 0.1, 0.1 / 2), markersize=10)
#             # for tvz, tsz, essz in metrics_z[s_first]:
#             #     plot_ts_tv += ax[index + 3].plot(essz, tvz, 'x', color=(0.1, 0.1, 0.1/2), markersize=10)
#             #     ax[index + 3].set_yscale('log')
#             #     ax[index + 1 + 3].plot(tsz, tvz, 'x', color=(0.1, 0.1, 0.1/2), markersize=10)
#             #     ax[index + 1 + 3].set_yscale('log')
#             #     ax[index + 2 + 3].plot(tsz, essz, 'x', color=(0.1, 0.1, 0.1/2), markersize=10)
#         # else:
#         #     for tvxy, trxy, essxy in metrics_x[s_init]:
#         #         plot_ess_tv += ax[index].plot(essxy, tvxy, 'x', color=(0.1, 0.1+color_change/2, 0.1+color_change/2),
#         #                                       markersize=10)
#         #         ax[index].set_yscale('log')
#         #         ax[index + 1].plot(trxy, tvxy, 'x', color=(0.1, 0.1 + color_change / 2, 0.1 + color_change / 2), markersize=10)
#         #         ax[index + 1].set_yscale('log')
#         #         ax[index + 2].plot(trxy, essxy, 'x', color=(0.1, 0.1 + color_change / 2, 0.1 + color_change / 2), markersize=10)
#
#         for tvxy, trxy, essxy in metrics_x[s_final]:
#             plot_ess_tv += ax[index].plot(essxy, tvxy, 'o', color=(0.9, 0.1, 0.1, 0.5), markersize=10)
#             ax[index].set_yscale('log')
#             ax[index + 1].plot(trxy, tvxy, 'o', color=(0.9, 0.1, 0.1, 0.5), markersize=10)
#             ax[index + 1].set_yscale('log')
#             ax[index + 2].plot(trxy, essxy, 'o', color=(0.9, 0.1, 0.1, 0.5), markersize=10)
#
#         # for tvz, tsz, essz in metrics_z[s_final]:
#         #     plot_ess_tv += ax[index + 3].plot(essz, tvz, 'o', color=(0.9, 0.1, 0.1, 0.5), markersize=10)
#         #     ax[index].set_yscale('log')
#         #     ax[index + 1 + 3].plot(tsz, tvz, 'o', color=(0.9, 0.1, 0.1, 0.5), markersize=10)
#         #     ax[index + 1 + 3].set_yscale('log')
#         #     ax[index + 2 + 3].plot(tsz, essz, 'o', color=(0.9, 0.1, 0.1, 0.5), markersize=10)
#
#
#         if index == 0:
#             ax[index].legend([plot_ess_tv[0], plot_ess_tv[-1]], ['1st Gen', '20th Gen'])
#         # else:
#         #     ax[index].legend([plot_ess_tv[0], plot_ess_tv[-1]], [str(starting_gen + index) + 'th Gen', '20th Gen'])
#
#
#         ax[index].grid()
#         ax[index + 1].grid()
#         ax[index + 2].grid()
#         ax[index + 3].grid()
#         ax[index + 1 + 3].grid()
#         ax[index + 2 + 3].grid()
#
#         # plt.show()
#
#         #if xy:
#         ax[index].set_ylabel("$TV_{xy}$", fontsize=18)
#         ax[index].set_xlabel("$Ess_{xy}$", fontsize=18 )
#         ax[index + 1].set_xlabel("$Ts_{xy}$", fontsize=18)
#         ax[index + 1].set_ylabel("$TV_{xy}$", fontsize=18)
#         ax[index + 2].set_xlabel("$Ts_{xy}$", fontsize=18)
#         ax[index + 2].set_ylabel("$Ess_{xy}$", fontsize=18)
#         # else:
#         ax[index + 3].set_ylabel("$TV_z$", fontsize=18)
#         ax[index + 3].set_xlabel("$Ess_z$", fontsize=18)
#         ax[index + 1 + 3].set_xlabel("$Ts_z$", fontsize=18)
#         ax[index + 1 + 3].set_ylabel("$TV_z$", fontsize=18)
#         ax[index + 2 + 3].set_xlabel("$Ts_z$", fontsize=18)
#         ax[index + 2 + 3].set_ylabel("$Ess_z$", fontsize=18)
#         # ax[1, index].legend([test2[0], test3[0]], ['1st Gen', str(starting_gen+index) + 'th Gen'])
#         # ax[1, index].set_title("Generation " + str(starting_gen + index))
#         ax[index].tick_params(axis='x', labelsize=15)
#         ax[index].tick_params(axis='y', labelsize=15)
#         ax[index + 1].tick_params(axis='x', labelsize=15)
#         ax[index + 1].tick_params(axis='y', labelsize=15)
#         ax[index + 2].tick_params(axis='x', labelsize=15)
#         ax[index + 2].tick_params(axis='y', labelsize=15)
#         ax[index + 3].tick_params(axis='x', labelsize=15)
#         ax[index + 3].tick_params(axis='y', labelsize=15)
#         ax[index + 4].tick_params(axis='x', labelsize=15)
#         ax[index + 4].tick_params(axis='y', labelsize=15)
#         ax[index + 5].tick_params(axis='x', labelsize=15)
#         ax[index + 5].tick_params(axis='y', labelsize=15)
#
#         # plt.figure(0)
#         # fig[0].tight_layout()
#         if save_fig:
#             fig[0].savefig('./images/pareto_xy_tv_ess.png', bbox_inches='tight')
#             fig[1].savefig('./images/pareto_xy_tv_ts.png', bbox_inches='tight')
#             fig[2].savefig('./images/pareto_xy_ts_ess.png', bbox_inches='tight')
#         # else:
#     plt.show()
#         # fig[3].savefig('./images/pareto_z_tv_ess.png', bbox_inches='tight')
#         # fig[4].savefig('./images/pareto_z_tv_ts.png', bbox_inches='tight')
#         # fig[5].savefig('./images/pareto_z_ts_ess.png', bbox_inches='tight')
#
#         # ax[2, index].legend([test4[0], test5[0]], ['1st Gen', str(starting_gen+index) + 'th Gen'])
#         # ax[2, index].set_title("Generation " + str(starting_gen + index))
#
#
#         # nao esquecer de plotar a primeira geracao

def ga_plot_mbk(gains, n_ind_per_gen, opt_type, name, n_gains=2, n_gen=20, save_fig=False):
    fig_, ax_ = plt.subplots(n_gains, 1)
    for gen, n_ind in zip(range(n_gen), n_ind_per_gen):
        starting_idx = int(np.sum(n_ind_per_gen[:gen]))

        mean_b = np.mean(gains[starting_idx:starting_idx+n_ind, b_idx])
        std_b = np.std(gains[starting_idx:starting_idx+n_ind, b_idx])
        mean_k = np.mean(gains[starting_idx:starting_idx+n_ind, k_idx])
        std_k = np.std(gains[starting_idx:starting_idx+n_ind, k_idx])

        ax_[0].plot(gen+1, mean_b, 'o', color="red")
        ax_[0].set_ylabel("B [Ns/mm]", fontsize='15')
        ax_[1].plot(gen+1, mean_k, 'o', color="orange")
        ax_[1].set_ylabel("K [N/mm]", fontsize='15')
        ax_[1].set_xlabel("Generations [-]", fontsize='15')
        ax_[0].set_ylim([0, 110])
        ax_[1].set_ylim([0, 150])

        ax_[0].errorbar(gen+1, mean_b, yerr=std_b, color="black", capsize=4)
        ax_[1].errorbar(gen+1, mean_k, yerr=std_k, color="black", capsize=4)
        ax_[0].set_xticks([2, 4, 6, 8, n_gen])
        ax_[1].set_xticks([2, 4, 6, 8, n_gen])
        ax_[0].grid(True)
        ax_[1].grid(True)
        # ax_[2].grid()
    if save_fig:
        plt.savefig('./plots/mbk_'+ name +'_'+ opt_type +'.png', bbox_inches='tight')
    else:
        plt.show()


def ga_plot_mbk_shaded(gains, n_ind_per_gen, opt_type, name, n_gains=2, n_gen=20, save_fig=False):
    mean_b = []
    std_b = []
    mean_k = []
    std_k = []
    for gen, n_ind in zip(range(n_gen), n_ind_per_gen):
        starting_idx = int(np.sum(n_ind_per_gen[:gen]))

        # mean_b = np.mean(gains[starting_idx:starting_idx+n_ind, b_idx])
        # std_b = np.std(gains[starting_idx:starting_idx+n_ind, b_idx])
        # mean_k = np.mean(gains[starting_idx:starting_idx+n_ind, k_idx])
        # std_k = np.std(gains[starting_idx:starting_idx+n_ind, k_idx])
        mean_b.append(np.mean(gains[starting_idx:starting_idx+n_ind, b_idx]))
        std_b.append(np.std(gains[starting_idx:starting_idx+n_ind, b_idx]))
        mean_k.append(np.mean(gains[starting_idx:starting_idx+n_ind, k_idx]))
        std_k.append(np.std(gains[starting_idx:starting_idx+n_ind, k_idx]))

    list_gen = np.arange(0, n_gen)+1

    mean_b = np.array(mean_b)
    std_b = np.array(std_b)
    mean_k = np.array(mean_k)
    std_k = np.array(std_k)



    # ax_[0].plot(list_gen, mean_b) #, color="red")
    # ax_[0].set_ylabel("B [Ns/m/rad]", fontsize='15')
    # ax_[0].fill_between(list_gen, mean_b-std_b, mean_b+std_b, alpha=0.3)
    ax_[0].plot(list_gen, generate_poly(list_gen, mean_b)) #, color="red")
    ax_[0].set_ylabel("B [Ns/m/rad]", fontsize='15')
    ax_[0].fill_between(list_gen, generate_poly(list_gen, mean_b)-generate_poly(list_gen, std_b), generate_poly(list_gen, mean_b) + generate_poly(list_gen, std_b), alpha=0.3)

    ax_[1].plot(list_gen, generate_poly(list_gen, mean_k))
    ax_[1].set_ylabel("K [N/m/rad]", fontsize='15')
    ax_[1].fill_between(list_gen, generate_poly(list_gen, mean_k)-generate_poly(list_gen, std_k), generate_poly(list_gen, mean_k) + generate_poly(list_gen, std_k), alpha=0.3)

    ax_[1].set_xlabel("Generations [-]", fontsize='15')
    ax_[0].set_ylim([0, 110])
    ax_[1].set_ylim([0, 50])

    # ax_[0].errorbar(gen+1, mean_b, yerr=std_b, color="black", capsize=4)
    # ax_[1].errorbar(gen+1, mean_k, yerr=std_k, color="black", capsize=4)
    # ax_[0] = plt.fill_between(gen+1, mean_b-std_b, mean_b+std_b)
    # ax_[1] = plt.fill_between(gen+1, mean_k-std_k, mean_k+std_k)
    ax_[0].set_xticks([2, 4, 6, 8, n_gen])
    ax_[1].set_xticks([2, 4, 6, 8, n_gen])
    ax_[0].grid(True)
    ax_[1].grid(True)
    # ax_[2].grid()

def generate_poly(x, y):
    poly = np.polyfit(x, y, 5)
    poly = np.poly1d(poly)(x)
    return poly


def get_number_generations(files_in_folder):
    return int(files_in_folder[-1][1:3])

def get_files_in_folder(folder_name):
    import os
    files = os.listdir(folder_name)
    files.sort()
    return files

def get_number_individuals(files):
    n_ind = []
    counter = 0
    gen = 0
    for ind in files:
        if "g" + str(gen).zfill(2) in ind:
            counter += 1
        else:
            n_ind.append(counter)
            counter = 1
            gen += 1
    return n_ind

def load_bk(files):
    _bk = []
    for ind in files:
        aux = ind.split("_")
        _b = float(aux[-1][1:-4])
        _k = float(aux[-2][1:])
        _bk.append([_b, _k])
    return np.array(_bk)

def load_metrics(files):
    metrics = []

    for ind in files:
        data = pd.read_csv(folder_name + ind)
        qvel = data_info.moving_average(np.diff(data[["q [rad]"]].values.reshape(-1,))/0.01)
        metric1 = data_info.rms(data["tau [Nm]"])
        metric2 = data_info.total_variation(qvel)
        metrics.append([metric1, metric2])

    return np.array(metrics)


if __name__ == '__main__':
    players = ['gustavo', 'velma', 'millhouse']
    opt_types = ["const", "unconst"]
    # name = 'gustavo'
    legend = []

    for opt_type in opt_types:
        fig_, ax_ = plt.subplots(2, 1)
        for name in players:
            folder_name = "./" + name + "_" + opt_type + "/"

            data_info = DataInfo()
            files_in_folder = get_files_in_folder(folder_name)
            # n_generations = get_number_generations(files_in_folder)
            n_generations = 9
            n_individuals_per_gen = get_number_individuals(files_in_folder)
            bk = load_bk(files_in_folder)
            metrics = load_metrics(files_in_folder)
            save_fig = True
            # ga_plot_forces(n_individuals=n_individuals_per_gen, n_generations=n_generations, save_fig=save_fig, files=files_in_folder, opt_type=opt_type, name=name)
            # ga_plot_pareto(n_individuals=n_individuals_per_gen, n_generations=n_generations, save_fig=save_fig, metrics=metrics, opt_type=opt_type, name=name)
            # ga_plot_mbk(save_fig=save_fig, gains=bk, n_ind_per_gen=n_individuals_per_gen, n_gen=n_generations, opt_type=opt_type, name=name)
            ga_plot_mbk_shaded(save_fig=save_fig, gains=bk, n_ind_per_gen=n_individuals_per_gen, n_gen=n_generations, opt_type=opt_type, name=name)
        if "unconst" == opt_type:
            ax_[0].set_title("Unconstrained")
        else:
            ax_[0].set_title("Constrained")
        ax_[1].legend(players)
        plt.savefig('./plots/mbk_'+opt_type+'.png', bbox_inches='tight')
        plt.clf()
