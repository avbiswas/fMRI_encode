import numpy as np
import tensorflow as tf

def getMiniBatch(dataset, label_dataset, batch_size=8):
    random_indices = np.random.choice(len(dataset), [batch_size], replace=False)
    return dataset[random_indices], label_dataset[random_indices]

X_train = np.load("supervised/X_train.npy")
Y_train = np.load("supervised/Y_train.npy")
X_train = np.squeeze(X_train, axis=1)
Y_train = np.expand_dims(Y_train, axis=-1)
X_test = np.load("supervised/X_test.npy")
Y_test = np.load("supervised/Y_test.npy")
X_test = np.squeeze(X_test, axis=1)
Y_test = np.expand_dims(Y_test, axis=-1)

tf.reset_default_graph()
X_input = tf.placeholder(tf.float32, [None, 5, 5, 5, 64])
label = tf.placeholder(tf.float32, shape=[None, 1], name='label')
keep_prob = 0.8
istraining = True

init=tf.truncated_normal_initializer(stddev=0.05)

conv1 = tf.layers.conv3d(inputs=X_input, filters=32, kernel_size=(3,3,3),
                         activation=tf.nn.relu,kernel_initializer=init)
conv1 = tf.layers.batch_normalization(conv1, training=istraining)
conv1 = tf.layers.dropout(conv1, rate=keep_prob)
print("conv1:", np.shape(conv1))

flatten = tf.layers.flatten(conv1)
print("flatten",np.shape(flatten))

dense1 = tf.layers.dense(flatten,16,activation=tf.nn.relu,kernel_initializer=init)
dense1 = tf.layers.batch_normalization(dense1, training=istraining)
dense1 = tf.layers.dropout(dense1, rate=keep_prob)

predict = tf.layers.dense(dense1, 1,kernel_initializer=init)
print("predict",np.shape(predict))

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=label, logits=predict, name=None
))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

acc = tf.reduce_mean(tf.cast(tf.equal(label, tf.round(tf.nn.sigmoid(predict))), tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

loss_sum = 0
epochs=1000
testing_interval = 2
logging_interval = 100
batch_size = 64
train_acc = 0
best_acc_yet = 0
for i in range(epochs):
    minibatchX, minibatchY = getMiniBatch(X_train, Y_train, batch_size)
    accuracy, l,_ = sess.run(fetches = [acc,loss,optimizer],
                             feed_dict = {X_input : minibatchX, label: minibatchY})
    loss_sum += l
    train_acc += accuracy
    if (i + 1) % testing_interval == 0:
        t_acc, l_test =  sess.run(fetches = [acc, loss], feed_dict = {X_input : X_test,
                                                                      label: Y_test,
                                                                     })
        if (i + 1) % logging_interval == 0:
            loss_sum = 0
            train_acc = 0
        if t_acc > best_acc_yet:
            best_acc_yet = t_acc
print(best_acc_yet)
        

