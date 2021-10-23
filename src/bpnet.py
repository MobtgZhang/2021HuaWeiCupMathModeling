import theano
import theano.tensor as T
import numpy as np


class Linear:
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim

        W = np.random.uniform(-np.sqrt(1.0 / in_dim), np.sqrt(1.0 / in_dim), (in_dim, out_dim))
        b = np.random.uniform(-np.sqrt(1.0 / out_dim), np.sqrt(1.0 / out_dim), (out_dim))

        self.weight = theano.shared(value=W.astype(theano.config.floatX), name="weight")
        self.bais = theano.shared(value=b.astype(theano.config.floatX), name="bais")
        self.gradient = {}

    def forwad(self, x):
        W_plus_b = T.dot(x, self.weight) + self.bais
        return W_plus_b

    def update_grad(self, loss_func):
        gW, gb = T.grad(loss_func, [self.weight, self.bais])
        self.gradient["grad_weight"] = gW
        self.gradient["grad_bais"] = gb

    def size(self):
        return (self.in_dim, self.out_dim)
class BpNet:
    def __init__(self, in_dim, hid_dim, out_dim, normalize_x, normalize_y):
        self.normalize_x = normalize_x
        self.normalize_y = normalize_y
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.inlayer = Linear(in_dim, hid_dim)
        self.outlayer = Linear(hid_dim, out_dim)
        self._build_model()

    def _build_model(self):
        # defination of some input and output values
        input = T.matrix(name="input")
        target = T.matrix(name="target")
        learning_rate = T.scalar(name="learning_rate")
        lambd = T.scalar(name="lambda")
        # build the calculating graph
        hidden = self.inlayer.forwad(input)
        relu_hid = T.nnet.relu(hidden)
        predict = self.outlayer.forwad(relu_hid)
        # defination of the loss function
        loss_func = T.mean(T.square(target - predict))
        # delta = target - predict
        # batch = delta.shape[0]
        # length = delta.shape[1]
        # loss_func = T.sum(T.sum(T.square(target-predict),axis=0)/length)/batch
        # get loss
        self.loss = theano.function(inputs=[target, predict], outputs=loss_func)
        # forward
        self.forward = theano.function(inputs=[input], outputs=predict)
        # backward
        self.inlayer.update_grad(loss_func)
        self.outlayer.update_grad(loss_func)
        # training process
        updates_list = []
        W_inlayer = self.inlayer.weight
        b_inlayer = self.inlayer.bais
        gW_inlayer = self.inlayer.gradient["grad_weight"]
        gb_inpayer = self.inlayer.gradient['grad_bais']
        updates_list.append((W_inlayer, W_inlayer - learning_rate * (gW_inlayer + lambd * W_inlayer)))
        updates_list.append((b_inlayer, b_inlayer - learning_rate * gb_inpayer))

        W_outlayer = self.outlayer.weight
        b_outlayer = self.outlayer.bais
        gW_outlayer = self.outlayer.gradient["grad_weight"]
        gb_outlayer = self.outlayer.gradient['grad_bais']
        updates_list.append((W_outlayer, W_outlayer - learning_rate * (gW_outlayer + lambd * W_outlayer)))
        updates_list.append((b_outlayer, b_outlayer - learning_rate * gb_outlayer))

        self.train = theano.function(inputs=[input, target, learning_rate, lambd],
                                     outputs=loss_func,
                                     updates=updates_list)
