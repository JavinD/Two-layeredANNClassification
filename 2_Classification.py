import tensorflow as tf
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

def get_data():
    x_data = pd.read_csv('Classification.csv', usecols=['volatile acidity', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'])
    y_data = pd.read_csv('Classification.csv', usecols=['quality'])
    return x_data, y_data

dataset, y_data = get_data()

# Deriving Features
dataset['free sulfur dioxide'] = dataset['free sulfur dioxide'].replace(['High', 'Medium', 'Low', 'Unknown'], [3, 2, 1, 0])
dataset['density'] = dataset['density'].replace(['Very High', 'High', 'Medium', 'Low'], [0, 3, 2, 1])
dataset['pH'] = dataset['pH'].replace(['Very Basic', 'Normal', 'Very Acidic', 'Very Accidic', 'Unknown'], [3, 2, 1, 1, 0])

y_data = y_data.replace(['Great', 'Good', 'Fine', 'Decent', 'Fair'], [4, 3, 2, 1, 0])

one_encoder = OneHotEncoder(sparse=False)
y_data = one_encoder.fit_transform(y_data)

# print (y_data)
# print (dataset)

# Normalizing Data and PCA

def apply_pca(dataset):
    scaler = MinMaxScaler()
    pca = PCA(n_components=4)

    data_normal = scaler.fit_transform(dataset)
    data_pca = pca.fit_transform(data_normal)

    principaldf = pd.DataFrame(data = data_pca, columns= ['PC 1', 'PC 2', 'PC 3', 'PC 4'])
    return principaldf

principalComponents = apply_pca(dataset)
# Turn into list
PClist = principalComponents.values.tolist()

x_train, x_test, y_train, y_test = train_test_split(dataset, y_data, test_size=0.3, random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

layer = {
    'input' : 8,
    'hidden' : 7,
    'output' : 5
}

weight = {
    'to_hidden' : tf.Variable(tf.random_normal([ layer['input'], layer['hidden'] ])),
    'to_output' : tf.Variable(tf.random_normal([ layer['hidden'], layer['output'] ]))
}

bias = {
    'to_hidden' : tf.Variable(tf.random_normal([ layer['hidden'] ])),
    'to_output' : tf.Variable(tf.random_normal([ layer['output'] ]))
}

x = tf.placeholder(tf.float32, [ None, layer['input'] ])

target = tf.placeholder(tf.float32, [ None, layer['output'] ])

lr = .5 #learning_rate
epoch = 5000

def predict():
    #u = x.w + b
    u = tf.matmul(x, weight['to_hidden'] + bias['to_hidden'])
    y = tf.nn.sigmoid(u)

    o = tf.matmul(y, weight['to_output'] + bias['to_output'])
    z = tf.nn.sigmoid(o)

    return z

y = predict()
.5*(target-y)**2
#MSE .5*(target-y)**2
loss = tf.reduce_mean(.5*(target-y)**2)
train = tf.train.GradientDescentOptimizer(lr).minimize(loss)
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(epoch):
        sess.run(train, feed_dict={x:x_train, target: y_train})

        if i%100 == 0:
            error = tf.reduce_mean(.5*(target-y)**2)
            error = sess.run(error, feed_dict={x:x_test, target:y_test})
            print(f"Epoch {i}, Error = {error}")

        if i==500:
            print(f'ITERATION:', i, 'Loss:', end=" ")
            print(sess.run(loss, feed_dict={x:x_test, target:y_test}))
        saver.save(sess,'model/my-model.ckpt')
        tempError = error

        if i>500 & i%500 == 0:
          if error<tempError:
            tempError = error
        saver.save(sess,'model/my-model.ckpt')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(epoch):
        sess.run(train, feed_dict={x:x_train, target: y_train})

        if i%5000 == 0: 
            true_predict = tf.equal(tf.argmax(y, axis=1), tf.argmax(target, axis=1))
            
            acc = tf.reduce_mean(tf.cast(true_predict, tf.float32))

            acc = sess.run(acc, feed_dict={x:x_test, target:y_test})

            print(f"Accuracy = {acc*100}%")