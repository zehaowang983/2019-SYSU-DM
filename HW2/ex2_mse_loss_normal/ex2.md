

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing   

# Try to find value for W and b to compute y_data = x_data * W + b  

# Define dimensions
d = 2    # Size of the parameter space
N = 50 # Number of data sample

# Model parameters
W = tf.Variable(tf.zeros([d, 1], tf.float32), name="weights")
b = tf.Variable(tf.zeros([1], tf.float32), name="biases")

# Model input and output
x = tf.placeholder(tf.float32, shape=[None, d])
y = tf.placeholder(tf.float32, shape=[None, 1])

# hypothesis
linear_regression_model = tf.add(tf.matmul(x, W), b)
# cost/loss function
loss = tf.reduce_mean(tf.square(linear_regression_model - y))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0002)
train = optimizer.minimize(loss)

training_filename = "dataForTraining.txt"
testing_filename = "dataForTesting.txt"
training_dataset = np.loadtxt("dataForTraining.txt")
testing_dataset = np.loadtxt("dataForTesting.txt")

dataset = np.vstack((training_dataset,testing_dataset))
min_max_scaler = preprocessing.MinMaxScaler()  
dataset = min_max_scaler.fit_transform(dataset)

# x_train = np.array(training_dataset[:,:2])
# y_train = np.array(training_dataset[:,2:3])
# x_test = np.array(testing_dataset[:,:2])
# y_test = np.array(testing_dataset[:,2:3])
x_train = np.array(dataset[:50,:2])
y_train = np.array(dataset[:50,2:3])
x_test = np.array(dataset[50:,:2])
y_test = np.array(dataset[50:,2:3])
print(x_train.shape)
print(y_train.shape)

save_step_loss = {"step":[],"train_loss":[],"test_loss":[]}# 保存step和loss用于可视化操作

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)  # reset values to wrong
    steps = 1500001
    for i in range(steps):
        sess.run(train, {x: x_train, y: y_train})
        if i % 100000 == 0:
            # evaluate training accuracy
            print("iteration times: %s" % i)
            curr_W, curr_b, curr_train_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
            print("W: %s \nb: %s \nTrain Loss: %s" % (curr_W, curr_b, curr_train_loss))
            # Accuracy computation
            curr_test_loss = sess.run(loss,{x:x_test,y:y_test})
            print("Test Loss: %s\n" % curr_test_loss)
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

    (50, 2)
    (50, 1)
    iteration times: 0
    W: [[6.8845053e-05]
     [4.4863267e-05]] 
    b: [0.00015073] 
    Train Loss: 0.19124699
    Test Loss: 0.24409673
    
    iteration times: 100000
    W: [[ 0.5242363 ]
     [-0.71658206]] 
    b: [0.439802] 
    Train Loss: 0.0048598954
    Test Loss: 0.0024313312
    
    iteration times: 200000
    W: [[ 0.6980384]
     [-0.9363729]] 
    b: [0.45488858] 
    Train Loss: 0.00048084414
    Test Loss: 0.00015238086
    
    iteration times: 300000
    W: [[ 0.7551583]
     [-1.0028392]] 
    b: [0.45740935] 
    Train Loss: 5.2551386e-05
    Test Loss: 0.00010532235
    
    iteration times: 400000
    W: [[ 0.77312946]
     [-1.022904  ]] 
    b: [0.45790595] 
    Train Loss: 1.0540489e-05
    Test Loss: 0.00015177278
    
    iteration times: 500000
    W: [[ 0.7782558]
     [-1.0288999]] 
    b: [0.45815554] 
    Train Loss: 6.200664e-06
    Test Loss: 0.00017157636
    
    iteration times: 600000
    W: [[ 0.7782558]
     [-1.0288999]] 
    b: [0.45815554] 
    Train Loss: 6.200664e-06
    Test Loss: 0.00017157636
    
    iteration times: 700000
    W: [[ 0.7782558]
     [-1.0288999]] 
    b: [0.45815554] 
    Train Loss: 6.200664e-06
    Test Loss: 0.00017157636
    
    iteration times: 800000
    W: [[ 0.7782558]
     [-1.0288999]] 
    b: [0.45815554] 
    Train Loss: 6.200664e-06
    Test Loss: 0.00017157636
    
    iteration times: 900000
    W: [[ 0.7782558]
     [-1.0288999]] 
    b: [0.45815554] 
    Train Loss: 6.200664e-06
    Test Loss: 0.00017157636
    
    iteration times: 1000000
    W: [[ 0.7782558]
     [-1.0288999]] 
    b: [0.45815554] 
    Train Loss: 6.200664e-06
    Test Loss: 0.00017157636
    
    iteration times: 1100000
    W: [[ 0.7782558]
     [-1.0288999]] 
    b: [0.45815554] 
    Train Loss: 6.200664e-06
    Test Loss: 0.00017157636
    
    iteration times: 1200000
    W: [[ 0.7782558]
     [-1.0288999]] 
    b: [0.45815554] 
    Train Loss: 6.200664e-06
    Test Loss: 0.00017157636
    
    iteration times: 1300000
    W: [[ 0.7782558]
     [-1.0288999]] 
    b: [0.45815554] 
    Train Loss: 6.200664e-06
    Test Loss: 0.00017157636
    
    iteration times: 1400000
    W: [[ 0.7782558]
     [-1.0288999]] 
    b: [0.45815554] 
    Train Loss: 6.200664e-06
    Test Loss: 0.00017157636
    
    iteration times: 1500000
    W: [[ 0.7782558]
     [-1.0288999]] 
    b: [0.45815554] 
    Train Loss: 6.200664e-06
    Test Loss: 0.00017157636
    



![png](output_0_1.png)



![png](output_0_2.png)



```python
#画图损失函数变化曲线
plt.plot(save_step_loss["step"],save_step_loss["train_loss"],label='Training Loss')
plt.plot(save_step_loss["step"],save_step_loss["test_loss"],label='Testing Loss')
plt.xlabel('Iteration times')
plt.ylabel('Loss')
plt.legend()
plt.show()
#画图损失函数变化曲线
plt.plot(save_step_loss["step"][1:],save_step_loss["train_loss"][1:],label='Training Loss')
plt.plot(save_step_loss["step"][1:],save_step_loss["test_loss"][1:],label='Testing Loss')
plt.xlabel('Iteration times')
plt.ylabel('Loss')
plt.legend()
plt.show()
#画图损失函数变化曲线
plt.plot(save_step_loss["step"][3:],save_step_loss["train_loss"][3:],label='Training Loss')
plt.plot(save_step_loss["step"][3:],save_step_loss["test_loss"][3:],label='Testing Loss')
plt.xlabel('Iteration times')
plt.ylabel('Loss')
plt.legend()
plt.show()
#画图损失函数变化曲线
plt.plot(save_step_loss["step"][5:],save_step_loss["train_loss"][5:],label='Training Loss')
plt.plot(save_step_loss["step"][5:],save_step_loss["test_loss"][5:],label='Testing Loss')
plt.xlabel('Iteration times')
plt.ylabel('Loss')
plt.legend()
plt.show()

print('Train Loss:\n',save_step_loss["train_loss"])
print('')
print('Test Loss:\n',save_step_loss["test_loss"])
```


![png](output_1_0.png)



![png](output_1_1.png)



![png](output_1_2.png)



![png](output_1_3.png)


    Train Loss:
     [0.19124699, 0.0048598954, 0.00048084414, 5.2551386e-05, 1.0540489e-05, 6.200664e-06, 6.200664e-06, 6.200664e-06, 6.200664e-06, 6.200664e-06, 6.200664e-06, 6.200664e-06, 6.200664e-06, 6.200664e-06, 6.200664e-06, 6.200664e-06]
    
    Test Loss:
     [0.24409673, 0.0024313312, 0.00015238086, 0.00010532235, 0.00015177278, 0.00017157636, 0.00017157636, 0.00017157636, 0.00017157636, 0.00017157636, 0.00017157636, 0.00017157636, 0.00017157636, 0.00017157636, 0.00017157636, 0.00017157636]

