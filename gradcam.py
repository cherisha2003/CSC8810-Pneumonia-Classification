import tensorflow as tf, numpy as np, cv2

def gradcam(model, img, layer_name="conv5_block16_concat"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img)
        loss = preds[:,0]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))

    heatmap = tf.reduce_mean(pooled * conv_out[0], axis=-1)
    heatmap = np.maximum(heatmap,0)
    heatmap /= np.max(heatmap)+1e-6
    return heatmap