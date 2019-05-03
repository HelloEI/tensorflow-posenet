import numpy as np
import tensorflow as tf

DEFAULT_PADDING = 'SAME'


def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):

    def __init__(self, inputs, trainable=True):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        # Switch variable for dropout
        self.use_dropout = tf.placeholder_with_default(tf.constant(1.0),
                                                       shape=[],
                                                       name='use_dropout')
        self.setup()

    def setup(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        #data_dict = np.load(data_path).item()
        data_dict = np.load(data_path,encoding="latin1").item()    
        #201903 加入encoding="latin1"
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
            # 201903
            #with tf.variable_scope(op_name, reuse=tf.AUTO_REUSE):
                #for param_name, data in data_dict[op_name].iteritems():
                #201903 Python3, AttributeError: 'dict' object has no attribute 'iteritems'
                for param_name, data in data_dict[op_name].items():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            #if isinstance(fed_layer, basestring):
            # 201903 for Python 3
            if isinstance(fed_layer, str):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    #def make_var(self, name, shape):
    #    '''Creates a new TensorFlow variable.'''
    #    return tf.get_variable(name, shape, trainable=self.trainable)
    #2019 此函数就是在tensorflow格式下建立变量
    def make_var(self, name, shape, initializer=None):
        return tf.get_variable(name, shape, initializer=initializer, trainable=self.trainable)
    

    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             biased=True):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0

        #2019
        print (name, input.get_shape())

        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
        #no-201903 Variable conv1/weights already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope? Originally defined at:
        #with tf.variable_scope(name, reuse=True) as scope:
            # 201903 unsupported operand type(s) for /: 'Dimension' and 'int'
            var_temp = int(c_i)/group
            
            if (name=='conv1'):
                #2019 采取截断是正态初始化权重，这只是一种initializer方法，mean=0,stddev=0.01
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.015)
            elif (name == 'conv2'):
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.02)
            elif (name=='icp1_out1' or name=='icp2_out1' or name=='icp3_out1' or name=='icp4_out1' or name=='icp5_out1' or name=='icp6_out1' or name=='icp7_out1' or name=='icp8_out1' or name=='icp9_out1'):
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.04)
            elif (name=='icp1_out2' or name=='icp2_out2' or name=='icp3_out2' or name=='icp4_out2' or name=='icp5_out2' or name=='icp6_out2' or name=='icp7_out2' or name=='icp8_out2' or name=='icp9_out2'):
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.08)
            else:
                init_weights = tf.contrib.layers.xavier_initializer()
            #2019 这也只是定义initializer的方法，初始化为0
            init_biases = tf.constant_initializer(0.0)
            
            #kernel = self.make_var('weights', shape=[k_h, k_w, var_temp, c_o])
            #2019
            kernel = self.make_var('weights', shape=[k_h, k_w, var_temp, c_o], initializer=init_weights)
            #kernel = self.make_var('weights', shape=[k_h, k_w, c_i / group, c_o])
            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                output = convolve(input, kernel)
            else:
                # Split the input into groups and then convolve each of them independently
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                # Concatenate the groups
                output = tf.concat(3, output_groups)
            # Add the biases
            if biased:
                #biases = self.make_var('biases', [c_o])
                #2019
                biases = self.make_var('biases', [c_o], initializer=init_biases)
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        #2019
        print (name, input.get_shape())

        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        #2019
        print (name, input.get_shape())
        
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(values=inputs, axis=axis, name=name)

    @layer
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True):
        with tf.variable_scope(name) as scope:
        #no-201903 Variable cls1_fc1_pose/weights already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope? Originally defined at:
        #with tf.variable_scope(name, reuse=True) as scope:
            input_shape = input.get_shape()

            #2019
            print (name, input.get_shape())

            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, input_shape[-1].value)
            
            #2019 
            if (name=='cls1_fc_pose_xyz' or name=='cls2_fc_pose_xyz' or name=='cls3_fc_pose_xyz'):
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.5)
            else:    
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            #2019 这也只是定义initializer的方法，初始化为0
            init_biases = tf.constant_initializer(0.0)
            #2019 
            weights = self.make_var('weights', shape=[dim, num_out], initializer=init_weights)
            biases = self.make_var('biases', [num_out], initializer=init_biases)            
            #weights = self.make_var('weights', shape=[dim, num_out])
            #biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = map(lambda v: v.value, input.get_shape())
        if len(input_shape) > 2:
            # For certain models (like NiN), the singleton spatial dimensions
            # need to be explicitly squeezed, since they're not broadcast-able
            # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
            if input_shape[1] == 1 and input_shape[2] == 1:
                input = tf.squeeze(input, squeeze_dims=[1, 2])
            else:
                raise ValueError('Rank 2 tensor input expected for softmax!')
        return tf.nn.softmax(input, name)

    @layer
    def batch_normalization(self, input, name, scale_offset=True, relu=False):
        # NOTE: Currently, only inference is supported
        #no-201903 Variable conv1/weights already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope? Originally defined at:
        #with tf.variable_scope(name, reuse=True) as scope:
        with tf.variable_scope(name) as scope:
            shape = [input.get_shape()[-1]]
            if scale_offset:
                scale = self.make_var('scale', shape=shape)
                offset = self.make_var('offset', shape=shape)
            else:
                scale, offset = (None, None)
            output = tf.nn.batch_normalization(
                input,
                mean=self.make_var('mean', shape=shape),
                variance=self.make_var('variance', shape=shape),
                offset=offset,
                scale=scale,
                # TODO: This is the default Caffe batch norm eps
                # Get the actual eps from parameters
                variance_epsilon=1e-5,
                name=name)
            if relu:
                output = tf.nn.relu(output)
            return output

    @layer
    def dropout(self, input, keep_prob, name):
        keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
        return tf.nn.dropout(input, keep, name=name)
