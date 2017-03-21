import os
import mxnet as mx

class QFuncInitializer(mx.initializer.Xavier):
    def __init__(self):
        super(QFuncInitializer, self).__init__(rnd_type='uniform', factor_type='in', magnitude=1.0)

    def _init_weight(self, name, arr):
        if name == 'qfunc_qval_weight':
            mx.random.uniform(-3e-3, 3e-3, out=arr)
        else:
            super(QFuncInitializer, self)._init_weight(name, arr)

    def _init_bias(self, name, arr):
        if name == 'qfunc_qval_bias':
            mx.random.uniform(-3e-3, 3e-3, out=arr)
        else:
            arr[:] = 0.0

class QFunc(object):
    """
    Base class for Q-Value Function.
    """

    def __init__(self, env_spec):

        self.env_spec = env_spec

    def get_qvals(self, obs, act):

        raise NotImplementedError


class ContinuousMLPQ(QFunc):
    """
    Continous Multi-Layer Perceptron Q-Value Network
    for determnistic policy training.
    """

    def __init__(
            self,
            env_spec,
            hidden_sizes=(32, 32),
            hidden_nonlinearity='relu',
            action_merge_layer=-2,
            output_nonlinearity=None,
            init=QFuncInitializer(),
    ):

        super(ContinuousMLPQ, self).__init__(env_spec)

        self.obs = mx.symbol.Variable("obs")
        self.act = mx.symbol.Variable("act")

        n_layers = len(hidden_sizes) + 1
        if n_layers > 1:
            action_merge_layer = \
                (action_merge_layer % n_layers + n_layers) % n_layers
        else:
            action_merge_layer = 1

        net = self.obs
        for idx, size in enumerate(hidden_sizes):
            if idx == action_merge_layer:
                net = mx.symbol.Concat(net, self.act, name="qunfc_concat")

            net = mx.symbol.FullyConnected(data=net, num_hidden=size, name='fc%d' % (idx+1))
            net = mx.symbol.Activation(data=net, act_type=hidden_nonlinearity, name='fc_%s%d' % (hidden_nonlinearity, idx+1))

        if action_merge_layer ==n_layers:
            net = mx.symbol.Concat(net, self.act, name="qunfc_concat")

        net = mx.symbol.FullyConnected(data=net, num_hidden=1, name='qfunc_qval')

        self.qval = net
        self.yval = mx.symbol.Variable("yval")

        self.init = init

    def get_output_symbol(self):

        return self.qval

    def get_loss_symbols(self):

        return {"qval": self.qval,
                "yval": self.yval}

    def define_loss(self, loss_exp):

        self.loss = mx.symbol.MakeLoss(loss_exp, name="qfunc_loss")
        self.loss = mx.symbol.Group([self.loss, mx.symbol.BlockGrad(self.qval)])

    def define_exe(self, ctx, updater, input_shapes=None, args=None,
                    grad_req=None):

        # define an executor, initializer and updater for batch version loss
        self.exe = self.loss.simple_bind(ctx=ctx, **input_shapes)
        self.arg_arrays = self.exe.arg_arrays
        self.grad_arrays = self.exe.grad_arrays
        self.arg_dict = self.exe.arg_dict
        
        for name, arr in self.arg_dict.items():
            if name not in input_shapes:
                self.init(mx.init.InitDesc(name), arr)
                # init(name, arr)
                
        self.updater = updater

    def update_params(self, obs, act, yval):

        self.arg_dict["obs"][:] = obs
        self.arg_dict["act"][:] = act
        self.arg_dict["yval"][:] = yval

        self.exe.forward(is_train=True)
        self.exe.backward()

        for i, pair in enumerate(zip(self.arg_arrays, self.grad_arrays)):
            weight, grad = pair
            self.updater(i, grad, weight)

    def save_params(self, dir_path='', name='QNet', itr=None, ctx=mx.cpu()):
        save_dict = {('arg:%s' % k): v.copyto(ctx) for k, v in self.arg_dict.items()}
        prefix = os.path.join(dir_path, name)
        if itr is not None:
            param_save_path = os.path.join('%s-%05d.params' % (prefix, itr))
        else:
            param_save_path = os.path.join('%s.params' % prefix)
        mx.nd.save(param_save_path, save_dict)

    def get_qvals(self, obs, act):

        self.exe.arg_dict["obs"][:] = obs
        self.exe.arg_dict["act"][:] = act
        self.exe.forward(is_train=False)

        return self.exe.outputs[1].asnumpy()


