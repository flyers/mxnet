import os
import mxnet as mx


class PolicyInitializer(mx.initializer.Xavier):
    def __init__(self):
        super(PolicyInitializer, self).__init__(rnd_type='uniform', factor_type='in', magnitude=1.0)

    def _init_weight(self, name, arr):
        if name == 'fc_out_weight':
            mx.random.uniform(-3e-3, 3e-3, out=arr)
        else:
            super(PolicyInitializer, self)._init_weight(name, arr)

    def _init_bias(self, name, arr):
        if name == 'fc_out_bias':
            mx.random.uniform(-3e-3, 3e-3, out=arr)
        else:
            arr[:] = 0.0

class Policy(object):
    """
    Base class of policy.
    """

    def __init__(self, env_spec):

        self.env_spec = env_spec

    def get_actions(self, obs):

        raise NotImplementedError

    @property
    def observation_space(self):

        return self.env_spec.observation_space

    @property
    def action_space(self):

        return self.env_spec.action_space


class DeterministicMLPPolicy(Policy):
    """
    Deterministic Multi-Layer Perceptron Policy used
    for deterministic policy training.
    """

    def __init__(
            self,
            env_spec,
            hidden_sizes=(32, 32),
            hidden_nonlinearity='relu',
            output_nonlinearity='tanh',
            init=PolicyInitializer(),
    ):

        super(DeterministicMLPPolicy, self).__init__(env_spec)

        self.obs = mx.symbol.Variable("obs")
        net = self.obs
        for idx, size in enumerate(hidden_sizes):
            net = mx.symbol.FullyConnected(data=net, num_hidden=size, name='fc%d' % (idx+1))
            net = mx.symbol.Activation(data=net, act_type=hidden_nonlinearity, name='fc_%s%d' % (hidden_nonlinearity, idx+1))

        net = mx.symbol.FullyConnected(data=net,
                                       num_hidden=self.env_spec.action_space.flat_dim,
                                       name='fc_out')
        net = mx.symbol.Activation(data=net, act_type=output_nonlinearity, name='act')
        self.act = net
        self.init = init

    def get_output_symbol(self):

        return self.act

    def get_loss_symbols(self):

        return {"obs": self.obs,
                "act": self.act}

    def define_loss(self, loss_exp):
        """
        Define loss of the policy. No need to do so here.
        """

        raise NotImplementedError

    def define_exe(self, ctx, updater, input_shapes=None, args=None,
                    grad_req=None):

        # define an executor, initializer and updater for batch version
        self.exe = self.act.simple_bind(ctx=ctx, **input_shapes)
        self.arg_arrays = self.exe.arg_arrays
        self.grad_arrays = self.exe.grad_arrays
        self.arg_dict = self.exe.arg_dict

        for name, arr in self.arg_dict.items():
            if name not in input_shapes:
                self.init(mx.init.InitDesc(name), arr)
                # init(name, arr)
                
        self.updater = updater

        # define an executor for sampled single observation
        # note the parameters are shared
        new_input_shapes = {"obs": (1, input_shapes["obs"][1])}
        self.exe_one = self.exe.reshape(**new_input_shapes)
        self.arg_dict_one = self.exe_one.arg_dict

    def update_params(self, grad_from_top):

        # policy accepts the gradient from the Value network
        self.exe.forward(is_train=True)
        self.exe.backward([grad_from_top])

        for i, pair in enumerate(zip(self.arg_arrays, self.grad_arrays)):
            weight, grad = pair
            self.updater(i, grad, weight)

    def get_actions(self, obs):

        # batch version
        self.arg_dict["obs"][:] = obs
        self.exe.forward(is_train=False)

        return self.exe.outputs[0].asnumpy()

    def get_action(self, obs):

        # single observation version
        self.arg_dict_one["obs"][:] = obs
        self.exe_one.forward(is_train=False)

        return self.exe_one.outputs[0].asnumpy()[0]

    def save_params(self, dir_path='', name='PolicyNet', itr=None, ctx=mx.cpu()):
        save_dict = {('arg:%s' % k): v.copyto(ctx) for k, v in self.arg_dict.items()}
        prefix = os.path.join(dir_path, name)
        if itr is not None:
            param_save_path = os.path.join('%s-%05d.params' % (prefix, itr))
        else:
            param_save_path = os.path.join('%s.params' % prefix)
        mx.nd.save(param_save_path, save_dict)






        