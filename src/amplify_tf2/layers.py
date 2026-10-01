# TensorFlow 2.x compatible version of AMPlify layers
# This version uses tf.matmul instead of K.dot for compatibility

import tensorflow as tf
from keras import backend as K
from keras import initializers, regularizers, constraints, activations
from keras.layers import Layer


class Attention(Layer):
    """
    Context Attention layer - adapted for TensorFlow 2.x
    """

    def __init__(self,
                 activation='tanh',
                 initializer='glorot_uniform',
                 return_attention=False,
                 W_regularizer=None,
                 u_regularizer=None,
                 b_regularizer=None,
                 W_constraint=None,
                 u_constraint=None,
                 b_constraint=None,
                 bias=True,
                 **kwargs):
        
        self.activation = activations.get(activation)
        self.initializer = initializers.get(initializer)
        
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        
        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)
        
        self.bias = bias
        self.supports_masking = True
        self.return_attention = return_attention

        super().__init__(**kwargs)

    def build(self, input_shape):
        amount_features = input_shape[-1]
        attention_size = input_shape[-1]

        self.W = self.add_weight(shape=(amount_features, attention_size),
                                 initializer=self.initializer,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint,
                                 name='attention_W')
        self.b = None
        if self.bias:
            self.b = self.add_weight(shape=(attention_size,),
                                     initializer='zero',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint,
                                     name='attention_b')

        self.context = self.add_weight(shape=(attention_size,),
                                       initializer=self.initializer,
                                       regularizer=self.u_regularizer,
                                       constraint=self.u_constraint,
                                       name='attention_us')

        super().build(input_shape)

    def call(self, x, mask=None):        
        # U = tanh(H*W + b) (eq. 8)        
        ui = tf.matmul(x, self.W)              # (b, t, a)
        if self.b is not None:
            ui += self.b
        ui = self.activation(ui)               # (b, t, a)

        # Z = U * us (eq. 9)
        us = tf.expand_dims(self.context, axis=-1)  # (a, 1)
        ui_us = tf.matmul(ui, us)              # (b, t, a) * (a, 1) = (b, t, 1)
        ui_us = tf.squeeze(ui_us, axis=-1)     # (b, t, 1) -> (b, t)
        
        # alpha = softmax(Z) (eq. 9)
        alpha = self._masked_softmax(ui_us, mask) # (b, t)
        alpha = tf.expand_dims(alpha, axis=-1)     # (b, t, 1)
        
        if self.return_attention:
            return alpha
        else:
            # v = alpha_i * x_i (eq. 10)
            return tf.reduce_sum(x * alpha, axis=1)
    
    def _masked_softmax(self, logits, mask):
        b = tf.reduce_max(logits, axis=-1, keepdims=True)
        logits = logits - b

        exped = tf.exp(logits)

        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            exped *= mask

        partition = tf.reduce_sum(exped, axis=-1, keepdims=True)
        partition = tf.maximum(partition, tf.keras.backend.epsilon())

        return exped / partition

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return input_shape[:-1]
        else:
            return input_shape[:-2] + input_shape[-1:]

    def compute_mask(self, x, input_mask=None):
        return None

    def get_config(self):
        config = {
            'activation': self.activation,
            'initializer': self.initializer,
            'return_attention': self.return_attention,

            'W_regularizer': initializers.serialize(self.W_regularizer),
            'u_regularizer': initializers.serialize(self.u_regularizer),
            'b_regularizer': initializers.serialize(self.b_regularizer),

            'W_constraint': constraints.serialize(self.W_constraint),
            'u_constraint': constraints.serialize(self.u_constraint),
            'b_constraint': constraints.serialize(self.b_constraint),
            
            'bias': self.bias
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ScaledDotProductAttention(Layer):
    """Scaled dot-product attention - adapted for TF2"""

    def __init__(self,
                 return_attention=False,
                 history_only=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.return_attention = return_attention
        self.history_only = history_only

    def get_config(self):
        config = {
            'return_attention': self.return_attention,
            'history_only': self.history_only,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            query_shape, key_shape, value_shape = input_shape
        else:
            query_shape = key_shape = value_shape = input_shape
        output_shape = query_shape[:-1] + value_shape[-1:]
        if self.return_attention:
            attention_shape = query_shape[:2] + (key_shape[1],)
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            mask = mask[0]
        if self.return_attention:
            return [mask, None]
        return mask

    def call(self, inputs, mask=None, **kwargs):
        if isinstance(inputs, list):
            query, key, value = inputs
        else:
            query = key = value = inputs
        if isinstance(mask, list):
            mask = mask[1]
        feature_dim = tf.cast(tf.shape(query)[-1], tf.float32)
        e = tf.matmul(query, key, transpose_b=True) / tf.sqrt(feature_dim)
        e = tf.exp(e - tf.reduce_max(e, axis=-1, keepdims=True))
        if self.history_only:
            query_len = tf.shape(query)[1]
            key_len = tf.shape(key)[1]
            indices = tf.expand_dims(tf.range(0, key_len), axis=0)
            upper = tf.expand_dims(tf.range(0, query_len), axis=-1)
            e *= tf.expand_dims(tf.cast(indices <= upper, tf.float32), axis=0)
        if mask is not None:
            e *= tf.cast(tf.expand_dims(mask, axis=-2), tf.float32)
        a = e / (tf.reduce_sum(e, axis=-1, keepdims=True) + K.epsilon())
        v = tf.matmul(a, value)
        if self.return_attention:
            return [v, a]
        return v


class MultiHeadAttention(Layer):
    """
    Multi-head scaled dot-product attention - adapted for TF2
    Uses tf.matmul instead of K.dot
    """

    def __init__(self,
                 head_num,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 history_only=False,
                 return_multi_attention=False,
                 **kwargs):
        self.supports_masking = True
        self.head_num = head_num
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.history_only = history_only
        self.return_multi_attention = return_multi_attention

        self.Wq, self.Wk, self.Wv, self.Wo = None, None, None, None
        self.bq, self.bk, self.bv, self.bo = None, None, None, None
        super().__init__(**kwargs)

    def get_config(self):
        config = {
            'head_num': self.head_num,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'history_only': self.history_only,
            'return_multi_attention': self.return_multi_attention
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if self.return_multi_attention:
            if isinstance(input_shape, list):
                q, k, v = input_shape
                return (q[0], self.head_num, q[1], k[1])
            return (input_shape[0], self.head_num, input_shape[1], input_shape[1])
        else:
            if isinstance(input_shape, list):
                q, k, v = input_shape
                return q[:-1] + (v[-1],)
            return input_shape

    def compute_mask(self, inputs, input_mask=None):
        if isinstance(input_mask, list):
            return input_mask[0]
        return input_mask

    def build(self, input_shape):
        if isinstance(input_shape, list):
            q, k, v = input_shape
        else:
            q = k = v = input_shape
        feature_dim = int(v[-1])
        if feature_dim % self.head_num != 0:
            raise IndexError('Invalid head number %d with the given input dim %d' % (self.head_num, feature_dim))
        
        # Use tf.Variable instead of add_weight for better TF2 compatibility
        self.Wq = self.add_weight(
            shape=(int(q[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wq' % self.name,
        )
        if self.use_bias:
            self.bq = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bq' % self.name,
            )
        self.Wk = self.add_weight(
            shape=(int(k[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wk' % self.name,
        )
        if self.use_bias:
            self.bk = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bk' % self.name,
            )
        self.Wv = self.add_weight(
            shape=(int(v[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wv' % self.name,
        )
        if self.use_bias:
            self.bv = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bv' % self.name,
            )
        self.Wo = self.add_weight(
            shape=(feature_dim, feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wo' % self.name,
        )
        if self.use_bias:
            self.bo = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bo' % self.name,
            )
        super().build(input_shape)

    @staticmethod
    def _reshape_to_batches(x, head_num):
        input_shape = tf.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        head_dim = feature_dim // head_num
        x = tf.reshape(x, (batch_size, seq_len, head_num, head_dim))
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, (batch_size * head_num, seq_len, head_dim))

    @staticmethod
    def _reshape_from_batches(x, head_num):
        input_shape = tf.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        x = tf.reshape(x, (batch_size // head_num, head_num, seq_len, feature_dim))
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, (batch_size // head_num, seq_len, feature_dim * head_num))
    
    @staticmethod
    def _reshape_attention_from_batches(x, head_num):
        input_shape = tf.shape(x)
        batch_size, seq_len = input_shape[0], input_shape[1]
        return tf.reshape(x, (batch_size // head_num, head_num, seq_len, seq_len))

    @staticmethod
    def _reshape_mask(mask, head_num):
        if mask is None:
            return mask
        seq_len = tf.shape(mask)[1]
        mask = tf.expand_dims(mask, axis=1)
        mask = tf.tile(mask, [1, head_num, 1])
        return tf.reshape(mask, (-1, seq_len))

    def call(self, inputs, mask=None):
        if isinstance(inputs, list):
            q, k, v = inputs
        else:
            q = k = v = inputs
        if isinstance(mask, list):
            q_mask, k_mask, v_mask = mask
        else:
            q_mask = k_mask = v_mask = mask
        q = tf.matmul(q, self.Wq)
        k = tf.matmul(k, self.Wk)
        v = tf.matmul(v, self.Wv)
        if self.use_bias:
            q += self.bq
            k += self.bk
            v += self.bv
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)            
        y, a = ScaledDotProductAttention(
            return_attention=True,
            history_only=self.history_only,
            name='%s-Attention' % self.name,
        )(
            inputs=[
                self._reshape_to_batches(q, self.head_num),
                self._reshape_to_batches(k, self.head_num),
                self._reshape_to_batches(v, self.head_num),
            ],
            mask=[
                self._reshape_mask(q_mask, self.head_num),
                self._reshape_mask(k_mask, self.head_num),
                self._reshape_mask(v_mask, self.head_num),
            ],
        )
        
        y = self._reshape_from_batches(y, self.head_num)
        a = self._reshape_attention_from_batches(a, self.head_num)
        y = tf.matmul(y, self.Wo)
        if self.use_bias:
            y += self.bo
        if self.activation is not None:
            y = self.activation(y)
        if self.return_multi_attention:
            return a
        return y
