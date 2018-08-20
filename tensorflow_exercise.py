### 3 Input XOR Gate ###

import numpy as np
import tensorflow as tf

tf.reset_default_graph()

#X = tf.placeholder(tf.float32, [4, 3])
#y = tf.placeholder(tf.int64, [4,1])

hidden_size = 5

W1 = tf.get_variable("W1",shape = [3,5])
b1 = tf.get_variable("b1",shape = [5])
W2 = tf.get_variable("W2",shape = [5,1])
b2 = tf.get_variable("b2",shape = [1])

X_train = np.array([ [0,0,0],
                     [0,0,1],
                     [0,1,0],
                     [0,1,1],
                     [1,0,0],
                     [1,0,1],
                     [1,1,0],
                     [1,1,1] ])

y_train = np.array([[0,1,1,0,1,0,0,1]]).T

X_train = tf.cast(X_train, tf.float32)
y_train = tf.cast(y_train, tf.float32)

def forward (X): #forward function

    out1 = tf.matmul(X,W1) + b1
    hidden_out = tf.nn.sigmoid(out1)
    out2 = tf.matmul(hidden_out,W2) + b2
    score = tf.nn.sigmoid(out2)
    
    return score

score = forward(X_train) 
#cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_train, logits=score))
error = tf.square(y_train - score)
updates = tf.train.GradientDescentOptimizer(0.2).minimize(error)

X＿test = np.array([[1,0,0],
                    [1,0,1],
                    [1,1,0],
                    [1,1,1] ])  

X_test = tf.cast(X_test, tf.float32)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(5000):
        sess.run(updates)
    print ("After Training score:")
    print (score.eval())
    print ("After Training error:")
    print (error.eval())
    score = forward(X_test)
    print ("After Training y_test:")
    print (np.round(score.eval()))
    