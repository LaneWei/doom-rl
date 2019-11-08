import threading
import tensorflow as tf
from tensorflow.layers import Conv2D, Dense, Flatten, Dropout
from time import sleep

from config import Config
from utils import process_batch


class A3CNetwork:
    def __init__(self, session=None, graph=tf.Graph()):
        self.state_shape = (Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, Config.IMAGE_CHANNELS)
        self.n_actions = Config.N_ACTIONS
        self.batch_size = Config.BATCH_SIZE
        self.decay_rate = Config.DECAY_RATE
        self.gamma = Config.GAMMA
        self.gamma_n = self.gamma ** Config.N_STEP_RETURN
        self.queue_length = Config.QUEUE_LEN
        
        self.s_input = None
        self.a_input = None
        self.r_input = None
        self.pi = None
        self.v = None
        self.loss = None
        self.train_op = None
        self.advantage = None

        self._session = session
        self.graph = graph
        with self.graph.as_default():
            self.lr = tf.Variable(Config.LEARNING_RATE, dtype=tf.float32, trainable=False, name='lr')
            self.update_lr = self.lr.assign(self.lr * self.decay_rate)
            self.steps = tf.Variable(0, dtype=tf.int32, trainable=False, name='GlobalSteps')
            self._build()
            self.saver = tf.train.Saver()
            self.init_op = tf.global_variables_initializer()

        self.save_steps = 1
        self.queue = [[], [], [], [], []]
        self.queue_size = 0
        self.lock = threading.Lock()

    def _build(self):
        self.s_input = tf.placeholder(tf.float32, [None] + list(self.state_shape), name='States')
        self.a_input = tf.placeholder(tf.float32, [None, self.n_actions], name='Actions')
        self.r_input = tf.placeholder(tf.float32, [None], name='Rewards')

        conv1 = Conv2D(32, 8, strides=(6, 6),
                       activation=tf.nn.relu,
                       kernel_initializer=tf.initializers.glorot_normal(),
                       bias_initializer=tf.initializers.glorot_normal(),
                       name='conv1')(self.s_input)
        conv2 = Conv2D(48, 6, strides=(3, 3),
                       activation=tf.nn.relu,
                       kernel_initializer=tf.initializers.glorot_normal(),
                       bias_initializer=tf.initializers.glorot_normal(),
                       name='conv2')(conv1)
        f = Flatten()(conv2)
        dense1 = Dense(512,
                       activation=tf.nn.relu,
                       kernel_initializer=tf.initializers.glorot_normal(),
                       bias_initializer=tf.initializers.glorot_normal(),
                       name='dense1')(f)
        dense_pi = Dense(128,
                         activation=tf.nn.relu,
                         kernel_initializer=tf.initializers.glorot_normal(),
                         bias_initializer=tf.initializers.glorot_normal(),
                         name='dense_pi')(dense1)
        self.pi = Dense(self.n_actions,
                        activation=tf.nn.softmax,
                        kernel_initializer=tf.initializers.glorot_normal(),
                        bias_initializer=tf.initializers.glorot_normal(),
                        name='Pi')(dense_pi)
        self.pi = tf.clip_by_value(self.pi, 1e-8, 1.)
        dense_v = Dense(32,
                        activation=tf.nn.relu,
                        kernel_initializer=tf.initializers.glorot_normal(),
                        bias_initializer=tf.initializers.glorot_normal(),
                        name='dense_v')(dense1)
        self.v = Dense(1,
                       activation=None,
                       kernel_initializer=tf.initializers.glorot_normal(),
                       name='V')(dense_v)
        self.v = tf.squeeze(self.v, axis=1)

        log_policy = tf.log(tf.reduce_sum(self.pi * self.a_input, axis=1) + 1e-10)
        self.advantage = self.r_input - self.v

        loss_pi = -tf.multiply(log_policy, tf.stop_gradient(self.advantage))
        loss_v = 0.5 * tf.square(self.advantage)
        entropy_pi = - 0.05 * tf.reduce_sum(self.pi * tf.log(self.pi), axis=1)
        self.loss = tf.reduce_mean(loss_pi + loss_v - entropy_pi)
        opt = tf.train.AdamOptimizer(self.lr)
        self.train_op = opt.minimize(self.loss)
    
    def add_sample(self, state, action, reward, next_state, mask):
        if self.queue_size >= self.queue_length:
            return False
        with self.lock:
            self.queue[0].append(state)
            self.queue[1].append(action)
            self.queue[2].append(reward)
            self.queue[3].append(next_state)
            self.queue[4].append(mask)
            self.queue_size += 1
        return True
    
    def get_sample(self):
        if self.queue_size < self.batch_size:
            sleep(0)
            return
        with self.lock:
            if self.queue_size < self.batch_size:
                return
            batch = [x[:self.batch_size] for x in self.queue]
            self.queue = [x[self.batch_size:] for x in self.queue]
            self.queue_size -= self.batch_size
        return batch
    
    def optimize(self):
        sample = self.get_sample()
        if sample is None:
            return
        
        states, actions, rewards, states_, masks = sample
        states = process_batch(states)
        states_ = process_batch(states_)
        v_ = self.session.run(self.v, {self.s_input: states_})
        target = rewards + self.gamma_n * v_ * masks
        _, a, loss = self.session.run([self.train_op, self.advantage, self.loss], {self.s_input: states,
                                                                                   self.a_input: actions,
                                                                                   self.r_input: target})
        return loss
            
    def predict(self, s):
        s = process_batch(s)
        v, pi, = self.session.run([self.v, self.pi], {self.s_input: s})
        return v, pi
    
    def lr_decay(self):
        self.session.run(self.update_lr)
        return self.learning_rate

    def reset_lr(self):
        self.session.run(self.lr.initializer)

    def save_variables(self, path):
        save_path = self.saver.save(self.session, path, self.save_steps)
        self.save_steps += 1
        print("Saving weights to: ", save_path)
    
    def retore(self, ckpt_dir):
        ckpt = tf.train.latest_checkpoint(ckpt_dir)
        if ckpt:
            print("Restoring session from %s..." % ckpt)
            self.saver.restore(self.session, ckpt)
        else:
            print("Unable to restore session from dir %s" % ckpt_dir)

    def init_variables(self):
        self.session.run(self.init_op)
    
    def finalize_graph(self):
        self.graph.finalize()
        
    @property
    def session(self):
        if self._session is None:
            self._session = tf.Session(graph=self.graph)
        return self._session
    
    @property
    def learning_rate(self):
        return self.session.run(self.lr)

