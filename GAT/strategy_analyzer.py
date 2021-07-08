import numpy as np
import json
import re
import sys
import tensorflow as tf
import google.protobuf.text_format as pbtf
from tensorflow.core.framework import graph_pb2

import numpy as np
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description='strategy Analyzer')
parser.add_argument('--input_path', type=str, default='', required=True,
                    help='Input file path. e.g, data/graph1')


TOP_K_OP = 100


def write_analysis_log(prefix, strategy_dict):
    with open(prefix+'/analysis.log', 'w') as logfile:
        with open(prefix+'/best_time.log', 'r') as f:
            txt_dict = json.load(f)
            best_reward = txt_dict['time']
            best_strategy = txt_dict['strategy']
            name_cost_dict = txt_dict['cost']
            group = txt_dict['group']
        print('Time:', best_reward)
        print('Group', group)
        logfile.write('Time:{}\n'.format(best_reward))
        logfile.write('Group:{}\n'.format(str(group)))

        sorted_tuple = list()
        for name, cost in name_cost_dict.items():
            sorted_tuple.append((int(cost[0]), name))
        sorted_tuple.sort(reverse=True)

        #strategy counter for all
        counter_dict = dict()
        for name, strategy in best_strategy.items():
            if counter_dict.get(str(strategy), None):
                counter_dict[str(strategy)] += 1
            else:
                counter_dict[str(strategy)] = 1

        for strategy, counter in counter_dict.items():
            ratio = float(counter) / len(best_strategy)
            print('Strategy:', strategy, ' Counter:', counter,' Ratio:', ratio)
            logfile.write('Strategy:{} Counter:{} Ratio:{}\n'.format(
                            strategy, counter, ratio))

        # Details of top K operation.
        if len(sorted_tuple) > TOP_K_OP:
            sorted_tuple = sorted_tuple[:TOP_K_OP]
        for item in sorted_tuple:
            name = item[1]
            cost = name_cost_dict[name]
            strategy = best_strategy[name]
            print('Name:', name, ' Strategy:', strategy, ' Cost:', cost)
            logfile.write('Name:{} Strategy:{} Cost:{}\n'.format(
                            name, strategy, cost))

        def _set_strategy_dict():
            null_gdef = graph_pb2.GraphDef()
            with open(prefix+'/null_graph.pbtxt', 'r')as f:
                txt = f.read()
            pbtf.Parse(txt, null_gdef)
            global_name_list = [nodedef.name for nodedef in null_gdef.node]
            for name, strategy in best_strategy.items():
                if strategy_dict.get(str(strategy), None) == None:
                    strategy_dict[str(strategy)] = list()
                name_list = strategy_dict.get(str(strategy), list())
                try:
                    input_node_idx = global_name_list.index(name)
                except Exception as e:
                    print(e)
                    continue
                input_nodedef = null_gdef.node[input_node_idx]
                size = 0
                for output_shape in input_nodedef.attr['_output_shapes'].list.shape:
                    local_size = 1
                    for dim in output_shape.dim:
                        local_size *= np.abs(dim.size)
                    size += local_size
                cost = name_cost_dict.get(name,[0])[0]
                name_list.append((name, size, cost))

        _set_strategy_dict()


def plot_strategy_analysis(prefix, strategy_dict):
    colors = ['green', 'blue', 'red', 'yellow', 'black']
    fig = plt.figure()
    ax = plt.subplot()
    for i, strategy_key in enumerate(sorted(strategy_dict)):
        strategy_item = strategy_dict[strategy_key]
        sizes, costs = list(), list()
        for name, size, cost in strategy_item:
            sizes.append(size)
            costs.append(cost)

        ax.scatter(sizes, costs, c=colors[i%len(colors)], label=strategy_key)
    plt.xlabel('size (Byte)')
    plt.ylabel('cost (ms)')
    plt.title(prefix)
    ax.legend()
    fig.savefig(prefix+'/analysis.png')


def write_strategy_analysis_log(prefix, strategy_dict):
    with open(prefix+'/strategy_analysis.log','w') as logfile:
        for _, strategy_key in enumerate(strategy_dict):
            logfile.write(strategy_key+'\n')
            strategy_item = strategy_dict[strategy_key]
            for name, size, cost in strategy_item:
                logfile.write('name:{}, size:{} Byte, cost: {}\n'.format(
                                name, size, cost))


def main():
    args = parser.parse_args()
    prefix = args.input_path

    strategy_dict = dict()

    # TODO: Integrate write_analysis_log() and write_strategy_analysis_log()
    write_analysis_log(prefix, strategy_dict)

    write_strategy_analysis_log(prefix, strategy_dict)

    plot_strategy_analysis(prefix, strategy_dict)




if __name__ == '__main__':
    main()
