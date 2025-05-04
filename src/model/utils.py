import tensorflow as tf

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg_1 = tf.math.rsqrt(step)
        arg_2 = step * (self.warmup_steps ** -1.5)
        lr_rate = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg_1, arg_2)
        return lr_rate
    
class MaskedMetrics:
    def __init__(self):
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

    def masked_loss(self, label, pred):
        mask = label != 0
        loss = self.loss_object(label, pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
        return loss

    def masked_accuracy(self, label, pred):
        pred = tf.argmax(pred, axis=2)
        label = tf.cast(label, pred.dtype)
        match = label == pred
        mask = label != 0
        match = match & mask
        match = tf.cast(match, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(match) / tf.reduce_sum(mask)