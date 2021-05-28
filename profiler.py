import numpy as np
import tensorflow as tf
import re
import itertools

from sklearn.linear_model import HuberRegressor

from tensorflow.python.ops import collective_ops

PROFILE_NUM = 1 # 5

class NcclProfiler:
    def __init__(self, devices, target, seed=3399):
        self.target = target
        self.seed = seed
        self.devices = {}

        self._group_devices_by_task(devices)

    def _group_devices_by_task(self, devices):
        # Group devices by task.
        # --> {'0': ['/task:0/device:GPU:0', ..], '1': ['/task:1/device:GPU:0']}
        for dev in devices:
            # For Python 3.5, group() function is called
            # to extract task's index.
            task = re.search("task:(\d+)/", dev).group(1)
            if task in self.devices.keys():
                self.devices[task].append(dev)
            else:
                self.devices[task] = [dev]

        # Sort devices by GPU's index.
        for devs in self.devices.values():
            devs.sort()
        print('__init__ self.devices', self.devices)

    def profile(self):
        results = {}

        for task, devs in self.devices.items():
            profile_data = [x for _ in range(PROFILE_NUM) for x in self._profile(devs)]
            print('profile_data:', profile_data)
            results[','.join(devs)] = self._model(profile_data)

        return results

    def _model(self, data):
        model1 = HuberRegressor().fit([[x] for x, y in data if x <= 2**9],
                                      [y for x, y in data if x <= 2**9])
        model2 = HuberRegressor().fit([[x] for x, y in data if x >= 2**10],
                                      [y for x, y in data if x >= 2**10])

        return [model1.coef_[0].item(), model1.intercept_.item(),
                model2.coef_[0].item(), model2.intercept_.item()]

    def _profile(self, devices):
        seed = self.seed
        self.seed += 1
        num_workers = len(devices)

        result = []
        # Profile the transfer time of various tensor with size of 1KB to 1GB
        for size in (2**i for i in range(21)):
            nccl_ops = []
            tf.reset_default_graph()
            for dev in devices:
                with tf.device(dev):
                    x = tf.random.uniform((size, 128), dtype=tf.dtypes.float64)
                    nccl_op = collective_ops.all_reduce(x, num_workers,
                                                        seed, seed, 'Add', 'Id')
                    nccl_op_tensor = tf.identity(nccl_op)
                    nccl_ops.append(nccl_op_tensor)
            run_meta = tf.compat.v1.RunMetadata()
            run_opt = tf.compat.v1.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            with tf.Session(self.target) as sess:
                sess.run(nccl_ops, options=run_opt, run_metadata=run_meta)

            collective_ops_times = []
            for d in run_meta.step_stats.dev_stats:
                for node in d.node_stats:
                    if 'CollectiveReduce' in node.node_name:
                        print('trnasfer time (msecs):', node.all_end_rel_micros)
                        collective_ops_times.append(node.all_end_rel_micros)
            time = min(collective_ops_times)
            print('min transfer time (msecs):', time)
            result.append((size, time))

        return result


class Profiler:
    def __init__(self, graph_def, batchsize, target=None, sinks=["GradientDescent"]):
        self.graph_def = graph_def
        self.batchsize = batchsize
        self.names = { node.name for node in graph_def.node }
        self.sinks = sinks
        self.target = target
        self.profiled = set()
        self.cache = {} # TODO: persistence? LRU?

    def _profile(self, device, run_meta):
        if run_meta is None:
            tf.reset_default_graph()
            tf.import_graph_def(self.graph_def)
            graph = tf.get_default_graph()
            for op in graph.get_operations():
                op._set_device(device)
            init = graph.get_operation_by_name("import/init")

            sess = tf.Session(self.target)#, config=tf.ConfigProto(allow_soft_placement=False))
            sess.run(init)

            placeholders = (node.outputs[0] for node in graph.get_operations() if node.node_def.op == 'Placeholder')
            input_dict = { p: np.random.rand(self.batchsize, *p.shape.as_list()[1:]) for p in placeholders }

            run_meta = tf.compat.v1.RunMetadata()
            run_opt = tf.compat.v1.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)#, output_partition_graphs=True)
            opt = [graph.get_operation_by_name('import/' + x) for x in self.sinks]
            sess.run(opt, feed_dict=input_dict)
            sess.run(opt, options=run_opt, run_metadata=run_meta, feed_dict=input_dict)

        result = {}
        for dev in run_meta.step_stats.dev_stats:
            if 'Kernel' not in dev.device and 'stream' not in dev.device: # TODO: if no GPU data for this op, use the CPU data
                continue
            for node in dev.node_stats:
                name = node.node_name.split(':')[0]
                if name[:7] == 'import/':
                    name = name[7:]
                if name not in result:
                    result[name] = [float('inf'), 0]
                result[name][0] = min(result[name][0], node.all_start_micros)
                result[name][1] = max(result[name][1], node.all_start_micros + node.all_end_rel_micros)

        for name, [start, end] in result.items():
            self.cache[(name, device)] = end - start

        self.profiled.add(device)

    def profile(self, node_name, device, run_meta=None):
        if device not in self.profiled:
            self._profile(device, run_meta)
        return self.cache.get((node_name, device), 0)
