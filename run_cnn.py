from network import Network
from layers import Relu, Linear, Conv2D, AvgPool2D, Reshape
from utils import LOG_INFO, onehot_encoding
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net, data_iterator
from load_data import load_mnist_4d

import json
import sys
import numpy as np
import matplotlib.pyplot as plt


# visualize the output
def vis_square(data):
    """
    Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)

    Here I referred this website: http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb
    """

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data)
    plt.axis('off')


train_data, test_data, train_label, test_label = load_mnist_4d('data')

if len(sys.argv) >= 2:
    filename = sys.argv[1]
else :
    filename = "test"


# Your model defintion here
# You should explore different model architecture
model = Network()
model.add(Conv2D('conv1', 1, 4, 3, 1, 1))
model.add(Relu('relu1'))
model.add(AvgPool2D('pool1', 2, 0))  # output shape: N x 4 x 14 x 14
model.add(Conv2D('conv2', 4, 4, 3, 1, 1))
model.add(Relu('relu2'))
model.add(AvgPool2D('pool2', 2, 0))  # output shape: N x 4 x 7 x 7
model.add(Reshape('flatten', (-1, 196)))
model.add(Linear('fc3', 196, 10, 0.1))

loss = SoftmaxCrossEntropyLoss(name='loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 0.01,
    'weight_decay': 0,
    'momentum': 0.9,
    'batch_size': 100,
    'max_epoch': 10,
    'disp_freq': 5,
    'test_epoch': 50
}

loss_plot_list = []
acc_plot_list = []
iter_plot_list = []
acc_test_plot_list = []

for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'], loss_plot_list, acc_plot_list, iter_plot_list)

    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        test_net(model, loss, test_data, test_label, config['batch_size'], acc_test_plot_list)


with open("acc_" + filename + ".txt", 'w') as json_file:
    json_file.write(json.dumps(acc_test_plot_list, indent=4))
with open("loss_" + filename + ".txt", 'w') as json_file:
    json_file.write(json.dumps(loss_plot_list, indent=4))



#
# for i in range(10):
#     for input, label in data_iterator(train_data, train_label, 10000):
#         number = label == i
#         label = label[number][:100]
#         input = input[number][:100]
#         target = onehot_encoding(label, 10)
#         output = model.layer_list[4].forward(input)
#         vis_square(output[:,0,:,:])
#         plt.savefig('%d.png'%i)
#         break
