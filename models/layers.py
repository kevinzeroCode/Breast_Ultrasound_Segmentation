import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.filters = filters
        self.query = tf.keras.layers.Conv2D(filters // 8, kernel_size=1, strides=1, padding='same')
        self.key = tf.keras.layers.Conv2D(filters // 8, kernel_size=1, strides=1, padding='same')
        self.value = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=1, padding='same')
        self.softmax = tf.keras.layers.Softmax(axis=-1)
        
    def call(self, inputs):
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)
        
        attention_map = tf.matmul(q, k, transpose_b=True)
        attention_map = self.softmax(attention_map)
        
        attention_out = tf.matmul(attention_map, v)
        return attention_out

    def get_config(self):
        config = super(SelfAttention, self).get_config()
        config.update({"filters": self.filters})
        return config