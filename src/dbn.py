import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np

# RBM model
class RBM(object):
    def __init__(self,input=None,n_visible=784,n_hidden=500,\
                 W=None,hbias=None,vbias=None,numpy_rng=None,
                 theano_rng=None):
        self.n_visible=n_visible
        self.n_hidden=n_hidden
        #生成随机数
        if numpy_rng is None:
            numpy_rng=np.random.RandomState(1234)

        #if theano_rng is None:
        #    theano_rng=RandomStreams(numpy_rng.randint(2**30))
        if W is None:
            initial_W=np.asarray(numpy_rng.uniform(
                    low=-4*np.sqrt(6./(n_hidden+n_visible)),
                    high=4*np.sqrt(6./(n_hidden+n_visible)),
                    size=(n_visible,n_hidden)),
                    dtype=theano.config.floatX)
            W=theano.shared(value=initial_W,name='W',borrow=True)
        if hbias is None:
            hbias=theano.shared(value=np.zeros(n_hidden,
                                                  dtype=theano.config.floatX),
                                name='hbias',borrow=True)
        if vbias is None:
            vbias=theano.shared(value=np.zeros(n_visible,
                                                  dtype=theano.config.floatX),
                                name='vbias',borrow=True)
        self.input=input
        if not input:
            self.input=T.matrix('input')

        self.W=W
        self.hbias=hbias
        self.vbias=vbias
        self.theano_rng=theano_rng
        self.params=[self.W,self.hbias,self.vbias]
    def free_energy(self,v_sample):
        wx_b=T.dot(v_sample,self.W)+self.hbias
        vbias_term=T.dot(v_sample,self.vbias)
        hbias_term=T.sum(T.log(1+T.exp(wx_b)),axis=1)
        return -vbias_term-hbias_term
    def propup(self,vis):
        pre_sigmoid_activation=T.dot(vis,self.W)+self.hbias
        return [pre_sigmoid_activation,T.nnet.sigmoid(pre_sigmoid_activation)]
    def sample_h_given_v(self,v0_sample):
        pre_sigmoid_h1,h1_mean=self.propup(v0_sample)
        h1_sample=self.theano_rng.binomial(size=h1_mean.shape,
                                           n=1,p=h1_mean,
                                           dtype=theano.config.floatX)
        return [pre_sigmoid_h1,h1_mean,h1_sample]
    def propdown(self,hid):
        pre_sigmoid_activation=T.dot(hid,self.W.T)+self.vbias
        return [pre_sigmoid_activation,T.nnet.sigmoid(pre_sigmoid_activation)]
    def sample_v_given_h(self,h0_sample):
        pre_sigmoid_v1,v1_mean=self.propdown(h0_sample)
        v1_sample=self.theano_rng.binomial(size=v1_mean.shape,n=1,p=v1_mean,
                                           dtype=theano.config.floatX)
        return [pre_sigmoid_v1,v1_mean,v1_sample]
    def gibbs_hvh(self,h0_sample):
        pre_sigmoid_v1,v1_mean,v1_sample=self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1,h1_mean,h1_sample=self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1,v1_mean,v1_sample,
                pre_sigmoid_h1,h1_mean,h1_sample]
    def gibbs_vhv(self,v0_sample):
        pre_sigmoid_h1,h1_mean,h1_sample=self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1,v1_mean,v1_sample=self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1,h1_mean,h1_sample,
                pre_sigmoid_v1,v1_mean,v1_sample]

    def get_cost_updates(self,lr=0.1,persistent=None,k=1):
        pre_sigmoid_ph,ph_mean,ph_sample=self.sample_h_given_v(self.input)
        if persistent is None:
            chain_start=ph_sample
        else:
            chain_start=persistent
        [pre_sigmoid_nvs,nv_means,nv_samples,
         pre_sigmoid_nhs,nh_means,nh_samples],updates=\
            theano.scan(self.gibbs_hvh,
                    outputs_info=[None,None,None,None,None,chain_start],
                    n_steps=k)
        chain_end=nv_samples[-1]

        cost=T.mean(self.free_energy(self.input))-T.mean(self.free_energy(chain_end))
        gparams=T.grad(cost,self.params,consider_constant=[chain_end])
        for gparam,param in zip(gparams,self.params):
            updates[param]=param-gparam*T.cast(lr,dtype=theano.config.floatX)
        if persistent:
            updates[persistent]=nh_samples[-1]
            monitoring_cost=self.get_pseudo_likehood_cost(updates)
        else:
            monitoring_cost=self.get_reconstruction_cost(updates,pre_sigmoid_nvs[-1])
        return monitoring_cost,updates
    def get_pseudo_likehood_cost(self,updates):
        bit_i_idx=theano.shared(value=0,name='bit_i_idx')
        xi=T.round(self.input)
        fe_xi=self.free_energy(xi)
        xi_flip=T.set_subtensor(xi[:,bit_i_idx],1-xi[:,bit_i_idx])
        fe_xi_flip=self.free_energy(xi_flip)
        cost=-T.mean(self.n_visible*T.log(T.nnet.sigmoid(fe_xi_flip-fe_xi)))
        updates[bit_i_idx]=(bit_i_idx+1)%self.n_visible
        return cost
    def get_reconstruction_cost(self,updates,pre_sigmoid_nv):
        cross_entropy=T.mean(T.sum(self.input*T.log(T.nnet.sigmoid(pre_sigmoid_nv))+
                                   (1-self.input)*T.log(1-T.nnet.sigmoid(pre_sigmoid_nv)),axis=1))
        return cross_entropy
    def reconstruct(self,v):
        x = T.matrix("input")
        h = T.nnet.sigmoid(T.dot(x,self.W) + self.hbias)
        reconstructed_v = T.nnet.sigmoid(T.dot(h,self.W.T) + self.vbias)
        reconstruct_func = theano.function(inputs = [x],outputs = [reconstructed_v])
        return reconstruct_func(v)

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        self.input = input
        if W is None:
            W_values = np.asarray(rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]
class DBNMlp:
    def __init__(self,numpy_rng,theano_rng=None,n_ins=784,n_outs=10,hidden_layers_sizes=[500,500],
                 normalize_x=False,normalize_y=False):
        self.normalize_x = normalize_x
        self.normalize_y = normalize_y
        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0
        if not theano_rng:
            theano_rng=RandomStreams(numpy_rng.randint(2**30))
        self.x = T.matrix('x')
        self.y = T.matrix('y')

        for k in range(self.n_layers):
            if k== 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[k-1]
            if k==0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[k-1].output

            sigmoid_layer=HiddenLayer(rng=numpy_rng,input=layer_input,n_in=input_size,
                                      n_out=hidden_layers_sizes[k],activation=T.nnet.sigmoid)
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)

            rbm_layer = RBM(input=layer_input,n_visible=input_size,n_hidden=hidden_layers_sizes[k],
                          W=sigmoid_layer.W,hbias=sigmoid_layer.b,numpy_rng=numpy_rng,theano_rng=theano_rng)
            self.rbm_layers.append(rbm_layer)
        # 最后一层神经网络
        self.last_layer = HiddenLayer(rng=numpy_rng,input = self.sigmoid_layers[-1].output,n_in = hidden_layers_sizes[-1],n_out = n_outs,activation = None)
        self.params.extend(self.last_layer.params)
        # 代价损失函数
        self.cost_func = T.mean(T.square(self.last_layer.output - self.y))
    def pretraining_functions(self,k = 1):
        learning_rate=T.scalar('lr') #学习率

        pretrain_fns=[]
        for rbm in self.rbm_layers:  #依次训练每个RBM
            #获得代价值和更新列表
            #使用CD-k(这里persisitent=None)，训练每个RBM
            cost,updates=rbm.get_cost_updates(learning_rate,persistent=None,k=k)

            #定义thenao函数,需要将learning_rate转换为tensor类型
            fn=theano.function(inputs=[self.x,theano.Param(learning_rate,default=0.1)],outputs=cost,updates=updates)
            #将'fn'增加到list列表中
            pretrain_fns.append(fn)
        return pretrain_fns
    def build_finetune_function(self,learning_rate):
        #计算梯度下降率
        gparams = T.grad(self.cost_func,self.params)
        #生成更新列表
        updates=[]
        for param,gparam in zip(self.params,gparams):
            updates.append((param,param-gparam*learning_rate))
        #定义训练函数
        train_fn=theano.function(inputs = [self.x,self.y],outputs = self.cost_func,updates=updates)
        return train_fn


