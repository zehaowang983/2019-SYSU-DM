

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Try to find value for W and b to compute y_data = x_data * W + b  

# Model parameters
W = tf.Variable(tf.zeros([2, 1], tf.float32), name="weights")
b = tf.Variable(tf.zeros([1], tf.float32), name="biases")

# Model input and output
x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

# hypothesis
linear_regression_model = tf.add(tf.matmul(x, W), b)
# cost/loss function
loss = tf.reduce_mean(tf.square(linear_regression_model - y)) / 100

# optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0002)
train = optimizer.minimize(loss)

training_filename = "dataForTraining.txt"
testing_filename = "dataForTesting.txt"
training_dataset = np.loadtxt("dataForTraining.txt")
testing_dataset = np.loadtxt("dataForTesting.txt")
x_train = np.array(training_dataset[:,:2])
y_train = np.array(training_dataset[:,2:3])
x_test = np.array(testing_dataset[:,:2])
y_test = np.array(testing_dataset[:,2:3])
print(x_train.shape)
print(y_train.shape)

save_step_loss = {"step":[],"train_loss":[],"test_loss":[]}# 保存step和loss用于可视化操作

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)  # reset values to wrong
    steps = 1500000
    for i in range(steps):
        sess.run(train, {x: x_train, y: y_train})
        if i % 100000 == 0:
            # evaluate training accuracy
            print("iteration times: %s" % i)
            curr_W, curr_b, curr_train_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
            print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_train_loss))
            # Accuracy computation
            curr_test_loss = sess.run(loss,{x:x_test,y:y_test})
            print("Test loss: %s\n" % curr_test_loss)
            save_step_loss["step"].append(i)
            save_step_loss["train_loss"].append(curr_train_loss)
            save_step_loss["test_loss"].append(curr_test_loss)

#画图损失函数变化曲线
plt.plot(save_step_loss["step"],save_step_loss["train_loss"],label='Training Loss')
plt.xlabel('Iteration times')
plt.ylabel('Loss (in billion RMB)')
plt.plot(save_step_loss["step"],save_step_loss["test_loss"],label='Testing Loss')
plt.legend()
plt.show()
#画图损失函数变化曲线
plt.plot(save_step_loss["step"][1:],save_step_loss["train_loss"][1:],label='Training Loss')
plt.xlabel('Iteration times')
plt.ylabel('Loss (in billion RMB)')
plt.plot(save_step_loss["step"][1:],save_step_loss["test_loss"][1:],label='Testing Loss')
plt.legend()
plt.show()
```

    WARNING:tensorflow:From /anaconda3/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    (50, 2)
    (50, 1)
    iteration times: 0
    W: [[0.16818722]
     [0.00662086]] b: [0.00160448] loss: 1825.3218
    Test loss: 2320.236
    
    iteration times: 100000
    W: [[  7.2115164]
     [-67.090904 ]] b: [2.6265802] loss: 5.7926745
    Test loss: 2.7971475
    
    iteration times: 200000
    W: [[  7.4772115]
     [-72.71891  ]] b: [4.4434733] loss: 3.2938375
    Test loss: 2.222327
    
    iteration times: 300000
    W: [[  7.4788413]
     [-73.069244 ]] b: [6.1595473] loss: 3.132317
    Test loss: 2.2343056
    
    iteration times: 400000
    W: [[  7.4633064]
     [-73.069244 ]] b: [7.8284774] loss: 2.992777
    Test loss: 2.170964
    
    iteration times: 500000
    W: [[  7.448169]
     [-73.069244]] b: [9.454624] loss: 2.8599422
    Test loss: 2.1114948
    
    iteration times: 600000
    W: [[  7.433078]
     [-73.069244]] b: [11.0758705] loss: 2.7305748
    Test loss: 2.054429
    
    iteration times: 700000
    W: [[  7.4188004]
     [-73.069244 ]] b: [12.609732] loss: 2.6110058
    Test loss: 2.002491
    
    iteration times: 800000
    W: [[  7.4045963]
     [-73.069244 ]] b: [14.135611] loss: 2.4947855
    Test loss: 1.952779
    
    iteration times: 900000
    W: [[  7.390605]
     [-73.069244]] b: [15.638727] loss: 2.3829556
    Test loss: 1.9057382
    
    iteration times: 1000000
    W: [[  7.3766255]
     [-73.069244 ]] b: [17.140522] loss: 2.273856
    Test loss: 1.8606379
    
    iteration times: 1100000
    W: [[  7.363965]
     [-73.069244]] b: [18.50062] loss: 2.1773226
    Test loss: 1.8214375
    
    iteration times: 1200000
    W: [[  7.3515368]
     [-73.069244 ]] b: [19.835764] loss: 2.0846574
    Test loss: 1.7844735
    
    iteration times: 1300000
    W: [[  7.3391085]
     [-73.069244 ]] b: [21.170908] loss: 1.9940772
    Test loss: 1.7490122
    
    iteration times: 1400000
    W: [[  7.3266807]
     [-73.069244 ]] b: [22.506052] loss: 1.9055735
    Test loss: 1.7150581
    



![png](output_0_1.png)



![png](output_0_2.png)

