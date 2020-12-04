import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys
import pandas as pd
import pickle
plt.rcParams['image.cmap'] = 'gray'

mask = np.load("common_mask.npy").astype('float32')
mask = np.array([mask])
mean_X = 0
std_X = 40
minX = -50.90334856299455
maxX = 60.29698656028302
time_length = 20

with open("train_test_split2.json", 'rb') as f:
    splits = pickle.load(f)
    train_files = splits["train_files"]
    train_labels = splits["train_labels"]
    test_files = splits["test_files"]
    test_labels = splits["test_labels"]

train_files = train_files
train_labels = train_labels
print(np.unique(train_labels, return_counts=True))
def load_file(filename):
    if not os.path.exists("NYU_dataset_fMRI/{}_func_preproc.nii.gz".format(filename)):
        return None
    return nib.load("NYU_dataset_fMRI/{}_func_preproc.nii.gz".format(filename)).get_fdata()

def preprocess(img):
    img = (img - mean_X)/std_X
    img = 2 * img / (maxX - minX)
    img = img * mask
    return img

def sample_time_sequence(img):
    random_time = np.random.choice(176 - time_length + 1)
    img = img[:, :, :, :, random_time:random_time+time_length]
    return img

def get_train_minibatch():
    while True:
        random_idx = np.random.choice(len(train_files))
        img = load_file(train_files[random_idx])
        if img is not None:
            break
    img = preprocess(img)
    label = train_labels[random_idx]
    return img, label

def get_test_minibatch():
    while True:
        random_idx = np.random.choice(len(test_files))
        img = load_file(test_files[random_idx])
        if img is not None:
            break
    img = preprocess(img)
    label = test_labels[random_idx]
    return img, label

# get_train_minibatch()
    
'''
X_train = (X_train - np.mean(X_train))/(np.std(img_unscaled))
img_unscaled = original_img[:, :, :, ::10]
img = (img_unscaled - np.mean(img_unscaled))/(np.std(img_unscaled))
img = 2 * img / (np.max(img) - np.min(img))
img = (img - np.min(img)) - 1
img = mask * img
'''

final_hidden_dims = 64
hidden_dims = 64
encoder_state_dims = 64
decoder_state_dims = 64
latentX, latentY, latentZ = [10, 10, 10]

def image_encoder(X, training=True, dropout_rate=0, debug=False):
    if debug:
        print("\nEncoder\n")
    init=tf.random_normal_initializer(0, 0.02)
    print(X)
    with tf.variable_scope("image_ae/encoder", reuse=tf.AUTO_REUSE):
        z1 = tf.layers.conv3d(inputs=X, filters=hidden_dims, kernel_size=(3, 4, 3), strides=2, kernel_initializer=init)
        z1_relu = tf.nn.leaky_relu(z1)
        if debug:
            print(z1_relu)
        z2 = tf.layers.conv3d(inputs=z1_relu, filters=hidden_dims, kernel_size=(3, 4, 3), strides=2, kernel_initializer=init)
        z2 = tf.layers.batch_normalization(z2, training=training)
        z2_relu = tf.nn.leaky_relu(z2)
        if debug:
            print(z2_relu)
        z2_relu = tf.nn.dropout(z2_relu, rate=dropout_rate)

        z3 = tf.layers.conv3d(inputs=z2_relu, filters=hidden_dims, kernel_size=(3, 4, 3), strides=1,  padding='valid', kernel_initializer=init)
        z3 = tf.layers.batch_normalization(z3, training=training)
        z3_relu = tf.nn.leaky_relu(z3)
        if debug:
            print(z3_relu)
        z3_relu = tf.nn.dropout(z3_relu, rate=dropout_rate)

        z5 = tf.layers.conv3d(inputs=z3_relu, filters=final_hidden_dims, kernel_size=(3, 4, 3), strides=1, padding='valid', kernel_initializer=init)
        z5_relu = tf.nn.tanh(z5)
        if debug:
            print(z5_relu)

    return z5_relu

def image_decoder(z, training=True, debug=False):
    if debug:
        print("\nDecoder\n")
    init=tf.random_normal_initializer(0, 0.02)
    with tf.variable_scope("image_ae/decoder", reuse=tf.AUTO_REUSE):
        z1 = tf.layers.Conv3DTranspose(filters=final_hidden_dims, kernel_size=(3, 4, 3), padding='valid', kernel_initializer=init)(z)
        z1_relu = tf.nn.leaky_relu(z1)
        if debug:
            print(z1_relu)
        z3 = tf.layers.Conv3DTranspose(filters=hidden_dims, kernel_size=(3, 4, 3), strides=1, padding='valid', kernel_initializer=init)(z1_relu)
        z3 = tf.layers.batch_normalization(z3, training=training)
        z3_relu = tf.nn.leaky_relu(z3)
        if debug:
            print(z3_relu)
        z3_relu = tf.nn.dropout(z3_relu, rate=dropout_rate)

        z4 = tf.layers.Conv3DTranspose(filters=hidden_dims, kernel_size=(3, 4, 3), strides=2, kernel_initializer=init)(z3_relu)
        z4 = tf.layers.batch_normalization(z4, training=training)
        z4_relu = tf.nn.leaky_relu(z4)
        if debug:
            print(z4_relu)
        z4_relu = tf.nn.dropout(z4_relu, rate=dropout_rate)

        z5 = tf.layers.Conv3DTranspose(filters=hidden_dims, kernel_size=(3, 4, 3), strides=2, padding='valid', kernel_initializer=init)(z4_relu)
        z5 = tf.layers.batch_normalization(z5, training=training)
        z5_relu = tf.nn.leaky_relu(z5)
        if debug:
            print(z5_relu)
        z6 = tf.layers.Conv3DTranspose(filters=1, kernel_size=(3, 4, 3), strides=1, padding='valid', kernel_initializer=init)(z5_relu)
        z6_relu = tf.nn.tanh(z6)
        if debug:
            print(z6_relu)
        
    return z6_relu


class Conv3DLSTMEncoder():
    def __init__(self, state_size, output_size):
        self.state_size = state_size
        self.output_size = output_size
        
    def call(self, inputs, states):
        init=tf.random_normal_initializer(0, 0.02)
        inputs = tf.concat([inputs, states[0]], axis=-1)
        forget = tf.layers.Conv3D(1, kernel_size=(2, 2, 2), padding='same', name='forget_gate', 
                              activation=tf.nn.sigmoid, kernel_initializer=init)(inputs)
        
        forget = tf.layers.Conv3D(encoder_state_dims, kernel_size=(2, 2, 2), padding='same', name='forget_gate', 
                              activation=tf.nn.sigmoid, kernel_initializer=init)(forget)
        
        I = tf.layers.Conv3D(1, kernel_size=(2, 2, 2), padding='same', name='input_gate', 
                              activation=tf.nn.sigmoid, kernel_initializer=init)(inputs)
        
        I = tf.layers.Conv3D(encoder_state_dims, kernel_size=(2, 2, 2), padding='same', name='input_gate', 
                              activation=tf.nn.sigmoid, kernel_initializer=init)(I)
        C = tf.layers.Conv3D(1, kernel_size=(2, 2, 2), padding='same', name='cell_temp', 
                              activation=tf.nn.tanh, kernel_initializer=init)(inputs)
        
        C = tf.layers.Conv3D(encoder_state_dims, kernel_size=(2, 2, 2), padding='same', name='cell_temp', 
                              activation=tf.nn.tanh, kernel_initializer=init)(C)
        
        next_state = tf.nn.tanh(tf.add(tf.multiply(forget, states[0]), tf.multiply(I, C)))
        return next_state, [next_state]
    
class Conv3DLSTMDecoder():
    def __init__(self, state_size, output_size, encoder_states, initial_input):
        self.state_size = state_size
        self.output_size = output_size
        self.encoder_states_time_c_first = tf.transpose(encoder_states, [0, 1, 5, 2, 3, 4])
        self.encoder_states_time_c_last = tf.transpose(encoder_states, [0, 2, 3, 4, 1, 5])
        self.final_state = encoder_states[:, -1, :, :, :]
        self.input = initial_input
        self.attention_weights = tf.Variable(tf.truncated_normal([latentX, latentY, latentZ], stddev=0.1))
                
    def get_initial_state(inputs):
        return inputs        
    
    def get_attention_features(self, prev_state):
        at = self.encoder_states_time_c_first * tf.transpose(prev_state, [0, 4, 1, 2, 3]) * self.attention_weights
        print("Attention", at)
        at = tf.reduce_sum(at, axis=[-1, -2, -3])
        print("Attention", at)
        at = tf.nn.softmax(at, axis=1)
        print("Softmax", at)
        C = tf.reduce_sum(at * self.encoder_states_time_c_last, axis=-2)
        print("C", C)
        return C
        
    def call(self, inputs, states):
        init=tf.random_normal_initializer(0, 0.02)
        # attention_state = self.get_attention_features(states[0])
        inputs = tf.concat([self.input, states[0]], axis=-1)        
        print("IP: ", inputs)
        forget = tf.layers.Conv3DTranspose(1, kernel_size=(2, 2, 2), padding='same', name='forget_gate', 
                              activation=tf.nn.sigmoid, kernel_initializer=init)(inputs)
        
        forget = tf.layers.Conv3DTranspose(decoder_state_dims, kernel_size=(2, 2, 2), padding='same', name='forget_gate', 
                              activation=tf.nn.sigmoid, kernel_initializer=init)(forget)
        
        I = tf.layers.Conv3DTranspose(1, kernel_size=(2, 2, 2), padding='same', name='input_gate', 
                              activation=tf.nn.sigmoid, kernel_initializer=init)(inputs)
        I = tf.layers.Conv3DTranspose(decoder_state_dims, kernel_size=(2, 2, 2), padding='same', name='input_gate', 
                              activation=tf.nn.sigmoid, kernel_initializer=init)(I)
        
        C = tf.layers.Conv3DTranspose(1, kernel_size=(2, 2, 2), padding='same', name='cell_temp', 
                              activation=tf.nn.tanh, kernel_initializer=init)(inputs)
        
        C = tf.layers.Conv3DTranspose(decoder_state_dims, kernel_size=(2, 2, 2), padding='same', name='cell_temp', 
                              activation=tf.nn.tanh, kernel_initializer=init)(C)
        
        next_state = tf.nn.tanh(tf.add(tf.multiply(forget, states[0]), tf.multiply(I, C)))
        # print(next_state)
        outputs = tf.layers.Conv3DTranspose(1, kernel_size=(2, 2, 2), padding='same', name='cell_out', 
                                   activation=tf.nn.tanh, kernel_initializer=init)(inputs)
        
        outputs = tf.layers.Conv3DTranspose(final_hidden_dims, kernel_size=(2, 2, 2), padding='same', name='cell_out', 
                                   activation=tf.nn.tanh, kernel_initializer=init)(outputs)
        self.input = outputs
        return outputs, [next_state]
        
    
def generate_labels(features):    
    with tf.variable_scope("prediction", reuse=tf.AUTO_REUSE):
        z1 = tf.keras.layers.MaxPool3D([2, 2, 2])(features)
        print(z1)
        flatten = tf.layers.flatten(z1)
        print(flatten)
        pred = tf.layers.dense(flatten, 1)
        print(pred)
        return pred
        
tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=[1, 61, 73, 61, time_length])
Y = tf.placeholder(tf.float32, shape=[1, 1])
dropout_rate = 0.1
# mask_flag = tf.placeholder(tf.float32, shape = [1, 61, 73, 61, 1])
mask_flag = tf.constant(mask)
image_opts = []
image_losses = []
X_rearranged = tf.expand_dims(tf.transpose(tf.squeeze(X, 0), [3, 0, 1, 2]), -1)
print(X_rearranged)
z = image_encoder(X_rearranged, training=True, debug=True)
print(z)
rx = image_decoder(z, training=True, debug=True)
print(rx)
rx = rx * mask_flag
print(rx)
reconstructed_image_pre = tf.expand_dims(tf.transpose(tf.squeeze(rx, -1), [1, 2, 3, 0]), axis=0)
print(reconstructed_image_pre)
image_losses = tf.reduce_mean(tf.square(rx - X_rearranged))
print(image_losses)

encoded_sequence = tf.expand_dims(z, axis=0)

with tf.variable_scope("seq2seq/encoder", tf.AUTO_REUSE):
    
    enc_state_size = tf.TensorShape([latentX, latentY, latentZ, encoder_state_dims])
    enc_output_size = tf.TensorShape([latentX, latentY, latentZ, encoder_state_dims])
    conv_3d_lstm_enc = Conv3DLSTMEncoder(enc_state_size, enc_output_size)
    enc_states, enc_last_state = tf.keras.layers.RNN(conv_3d_lstm_enc, 
                                                      return_sequences=True, 
                                                      return_state=True)(encoded_sequence)

print(enc_states, enc_last_state)

predicted_label = generate_labels(enc_last_state)

prediction_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predicted_label, labels=Y))
print(prediction_loss)

with tf.variable_scope("seq2seq/decoder", tf.AUTO_REUSE):
    dec_state_size = tf.TensorShape([latentX, latentY, latentZ, decoder_state_dims])
    dec_output_size = tf.TensorShape([latentX, latentY, latentZ, final_hidden_dims])
    input_sequence = tf.ones([tf.shape(enc_states)[0], time_length - 1, 1])
    conv_3d_lstm_dec = Conv3DLSTMDecoder(state_size=dec_state_size, 
                                         output_size=dec_output_size, 
                                         encoder_states=enc_states,
                                         initial_input=encoded_sequence[:, 0] )
    dec_outputs = tf.keras.layers.RNN(conv_3d_lstm_dec, return_sequences=True)(input_sequence, initial_state=enc_last_state)


print("Before Concat: ", dec_outputs)
dec_outputs = tf.concat([tf.expand_dims(encoded_sequence[:, 0], axis=1), dec_outputs], axis=1)    
print("After Concat: ", dec_outputs)

dec_outputs_rearranged = tf.squeeze(dec_outputs, 0)
print(dec_outputs_rearranged)
rx = image_decoder(dec_outputs_rearranged, training=True, debug=False)
print(rx)
reconstructed_full_images = rx * mask_flag
reconstructed_full_images = tf.expand_dims(tf.transpose(tf.squeeze(reconstructed_full_images, -1), [1, 2, 3, 0]), 0)
print("Full Image: ", reconstructed_full_images)
print(encoded_sequence, encoded_sequence[:, 1:])
sequence_losses = tf.reduce_mean(tf.square(encoded_sequence - dec_outputs))

end_to_end_loss = tf.reduce_mean(tf.square(reconstructed_full_images - X)) + sequence_losses + image_losses

prediction_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="image_ae/encoder") + \
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="seq2seq/encoder") + \
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="prediction")
                        
prediction_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope="image_ae/encoder") + \
                        tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope="seq2seq/encoder") + \
                        tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope="prediction")

image_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="image_ae")

seq2seq_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="seq2seq")

image_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="image_ae")

seq2seq_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="seq2seq")

with tf.control_dependencies(image_update_ops):
    image_opts = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(image_losses,                                                                         var_list=image_variables)

with tf.control_dependencies(seq2seq_update_ops):
    seq_opts = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(sequence_losses, var_list=seq2seq_variables)

with tf.control_dependencies(prediction_update_ops):
    pred_opts = tf.train.AdamOptimizer(learning_rate=5e-4).minimize(prediction_loss, var_list=prediction_variables)

end_to_end_opt = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(end_to_end_loss)

def train_seq2seq(sess, image):
    sl, _ = sess.run([sequence_losses, seq_opts], feed_dict={X: image})
    return sl

def train_image_ae(sess, image):
    il, _ = sess.run([image_losses, image_opts], feed_dict={X: image})
    return il

def train_end_to_end(sess, image):
    el, _ = sess.run([image_losses, end_to_end_opt], feed_dict={X: image})
    return el

def get_end_to_end_loss(sess, image):
    return sess.run(end_to_end_loss, feed_dict={X: image})

def train_prediction(sess, image, label):
    logit, pl, _ = sess.run([tf.nn.sigmoid(predicted_label), prediction_loss, pred_opts],
                            {X: image, Y: [[label]]})
    label_hat = np.squeeze(np.round(logit))
    return pl
    
def get_pred_acc(sess, full_image, label):
    pred = 0
    for _ in range(5):
        image = sample_time_sequence(full_image)
        label_ = sess.run(tf.nn.sigmoid(predicted_label), feed_dict={X: image})
        pred += label_
    pred = np.round(pred / 5)
    print(pred, label, pred==label)
    if pred == label:
        return 1
    else:
        return 0

def get_image_losses(sess, image):
    losses = sess.run(image_losses, feed_dict={X: image})
    return np.mean(losses)

def get_reconstructed_fmri(sess, image, mode="ae"):
    if mode == "ae":
        recon = sess.run(reconstructed_image_pre, feed_dict={X: image})
    else:
        recon = sess.run(reconstructed_full_images, feed_dict={X: image})
    return recon

def get_sequence_losses(sess, image):
    losses = sess.run(sequence_losses, feed_dict={X: image})
    return np.mean(losses)


def get_losses(sess, image):
    s_l, i_l = sess.run([sequence_losses, image_losses], 
                        feed_dict={X: image})
    return np.mean(s_l), np.mean(i_l)
    
def get_latent(sess, image):
    return sess.run(tf.keras.layers.MaxPool3D([2, 2, 2])(enc_last_state), feed_dict={X: image})

def get_test_loss(sess, mode):
    avg_ae_loss = []
    avg_seq_loss = []
    avg_pred_accuracy = []
    for _ in range(10):
        test_image, test_labels = get_test_minibatch()
        for __ in range(5):
            test_img = sample_time_sequence(test_image)
            if 'ae' in mode:
                avg_ae_loss.append(get_image_losses(sess, test_img))
            if 'seq' in mode:
                avg_seq_loss.append(get_sequence_losses(sess, test_img))
        if 'e2e' in mode:
            avg_pred_accuracy.append(get_pred_acc(sess, test_image, test_labels))
    return np.mean(avg_ae_loss), \
           0 if len(avg_seq_loss) == 0 else np.mean(avg_seq_loss), \
           0 if len(avg_pred_accuracy) == 0 else np.mean(avg_pred_accuracy)

print("IMG: ", image_variables)
print("Seq: ", seq2seq_variables)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

print("\nNumber of trainable parameters: ", np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

if sys.argv[1] == 'viz':
    saver.restore(sess, sys.argv[2])
    train_image, label = get_test_minibatch()
    img = sample_time_sequence(train_image)
    np.save("reconstructed/original.npy", img)
    recon = get_reconstructed_fmri(sess, img, "ae")
    recon2 = get_reconstructed_fmri(sess, img, "seq")
    print("AE Loss: ", np.mean(np.square(img - recon)))
    print("SEQ Loss: ", np.mean(np.square(img - recon2)))
    np.save("reconstructed/recon_ae.npy", recon)
    np.save("reconstructed/recon_seq.npy", recon2)

elif sys.argv[1] == 'predict':
    saver.restore(sess, sys.argv[2])
    accs = []
    for file, label in zip(test_files, test_labels):
        original_img = load_file(file)
        
        # print(np.shape(original_img))
        if original_img is None:
            continue
        else:
            original_img = preprocess(original_img)
        acc = get_pred_acc(sess, original_img, label)
        accs.append(acc)
        print(acc, label)
    print(np.mean(acc))
    
    
elif sys.argv[1] == 'latent':
    saver.restore(sess, sys.argv[2])
    latent_embs = []
    ground_truth_labels = []
    print(train_labels)
    group_0 = np.nonzero(np.array(train_labels) == 0)[0]
    group_1 = np.nonzero(np.array(train_labels) == 1)[0]
    np.random.shuffle(group_0)
    np.random.shuffle(group_1)
    train_idx = np.concatenate([group_0[:40], group_1[:55]])
    test_idx = np.concatenate([group_0[40:], group_1[55:]])
    print(len(train_idx), len(test_idx))
    for idx in train_idx:
        file = train_files[idx]
        label = train_labels[idx]
        original_img = load_file(file)
        if original_img is None:
            continue
        original_img = preprocess(original_img)
        for _ in range(10):
            img = sample_time_sequence(original_img)
            encodings = get_latent(sess, img)
            # print(np.shape(encodings))
            latent_embs.append(encodings)
            ground_truth_labels.append(label)
        print(file)
    np.save("supervised/X_train.npy", latent_embs)
    np.save("supervised/Y_train.npy", ground_truth_labels)
    
    latent_embs = []
    ground_truth_labels = []
    
    #for file, label in zip(train_files[test_idx], train_labels[test_idx]):
    for idx in test_idx:
        file = train_files[idx]
        label = train_labels[idx]
        
        original_img = load_file(file)
        if original_img is None:
            continue
        original_img = preprocess(original_img)
        for _ in range(10):
            img = sample_time_sequence(original_img)
            encodings = get_latent(sess, img)
            latent_embs.append(encodings)
            ground_truth_labels.append(label)
        print(file)
    np.save("supervised/X_test.npy", latent_embs)
    np.save("supervised/Y_test.npy", ground_truth_labels)

elif sys.argv[1] == 'train':
    if len(sys.argv) < 3:
        sess.run(tf.global_variables_initializer())
    else:
        saver.restore(sess, sys.argv[2])
    training_mode = "e2e"
    loss_history = []
    best_model = np.inf
    import time
    train_image, label = get_train_minibatch()
    for _ in range(100000):
        img = sample_time_sequence(train_image)
        start_time = time.time()
        if training_mode == 'e2e':
            # ae_loss = train_image_ae(sess, img)
            # seq_loss = train_seq2seq(sess, img)
            train_image_ae(sess, img)
            train_seq2seq(sess, img)
            loss = train_prediction(sess, img, label)
            
        elif training_mode == 'ae':
            loss = train_image_ae(sess, img)
        elif training_mode == 'seq':
            loss = train_seq2seq(sess, img)
        loss_history.append(loss)
        # print(ae_loss, seq_loss, pred_loss)
        if _ % 2 == 0:
            print("Mode: {}, Iteration: {}, Mean Loss: {}".format(training_mode, _, np.mean(loss_history)))
            loss_history = []
            train_image, label = get_train_minibatch()
            saver.save(sess, "models/model_att5_Full/model.ckpt")
        if _ % 25 == 0:
            if training_mode == 'e2e':
                ae_test_loss, seq_test_loss, pred_test_acc = get_test_loss(sess, mode=['ae', 'seq', 'e2e'])
                print("Test Image Loss: {}, Test Sequence Loss: {}, Test Pred Accuracy: {}".format(ae_test_loss, seq_test_loss, pred_test_acc))
                if best_model > -pred_test_acc:
                    best_model = -pred_test_acc
                    saver.save(sess, "models/model_att5_best_pred/model.ckpt")
                    print("SAVED")
            if training_mode == 'seq':
                ae_test_loss, seq_test_loss, pred_test_acc = get_test_loss(sess, mode=['ae', 'seq'])
                print("Test Image Loss: {}, Test Sequence Loss: {}".format(ae_test_loss, seq_test_loss))
                if best_model > seq_test_loss:
                    best_model = seq_test_loss
                    saver.save(sess, "models/model_att5_best_seq/model.ckpt")
                    print("SAVED")
            if training_mode == 'ae':
                ae_test_loss, seq_test_loss, pred_test_acc = get_test_loss(sess, mode=['ae'])
                print("Test Image Loss: {}".format(ae_test_loss))
                if best_model > ae_test_loss:
                    best_model = ae_test_loss
                    saver.save(sess, "models/model_att5_best_ae/model.ckpt")
                    print("SAVED")
            