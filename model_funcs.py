# Contains model functions for building InceptionTime model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback

# Generator functions for producing segments of ECG
def train_generator(X, y, wind, bs, prob_type):
    if prob_type=='uniform':
        # Calculate probabilities of each ECG being selected based on count of conditions
        # Only needs to happen once
        probs = []
        cond_counts = np.count_nonzero(y, axis=0) # 1D array with the total number of each condition in y
        ecg_counts = cond_counts * y  # 2D array, the above for each row/ecg where condition exists
        for i in range(ecg_counts.shape[0]):
            row = list(ecg_counts[i, :])
            while 0 in row:  
                row.remove(0)
            value = np.mean(row)   # Can change this to see effect of min, mean, max etc
            value = 1/value
            probs.append(value)

        probs = probs/sum(probs)
        
    elif prob_type=='same':
        probs  = None
    else:
        print('ERROR: prob_type value invalid')
    
    dist_tracker = [0]*y.shape[1]
    
    inputs = []
    targets = []
    bc = 0  # Batch count
    max_start_idx = X.shape[1] - wind

    t_idxs = np.random.randint(0, max_start_idx, size=bs)
    # Select ecgs based on probabilities to ensure uniform share of conditions
    ecg_idxs = np.random.choice(range(X.shape[0]), size=bs, p=probs)
    
    while True:
        # Get the current indexes
        t_idx = t_idxs[bc]
        ecg_idx = ecg_idxs[bc]
        # Get the segment and label
        segment = X[ecg_idx, t_idx:t_idx+wind, :]
        label = y[ecg_idx, :]
        
        # Append to list
        inputs.append(segment)
        targets.append(label)
        
        bc += 1
        if bc >= bs:
            # End of batch, output and reset
            retX = np.array(inputs, dtype='float32')
            rety = np.array(targets, dtype='float32')
            yield (retX, rety)
            # Generator will resume here after yield
            inputs = []
            targets = []
            bc = 0  # Batch count
            max_start_idx = X.shape[1] - wind
            t_idxs = np.random.randint(0, max_start_idx, size=bs)
            ecg_idxs = np.random.randint(0, X.shape[0], size=bs)


# Callback functions
class CosineAnnealer:
    
    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps
        self.n = 0
        
    def step(self):
        self.n += 1
        cos = np.cos(np.pi * (self.n / self.steps)) + 1
        return self.end + (self.start - self.end) / 2. * cos

    
class OneCycleScheduler(Callback):
    """ `Callback` that schedules the learning rate on a 1cycle policy as per Leslie Smith's paper(https://arxiv.org/pdf/1803.09820.pdf).
    If the model supports a momentum parameter, it will also be adapted by the schedule.
    The implementation adopts additional improvements as per the fastai library: https://docs.fast.ai/callbacks.one_cycle.html, where
    only two phases are used and the adaptation is done using cosine annealing.
    In phase 1 the LR increases from `lr_max / div_factor` to `lr_max` and momentum decreases from `mom_max` to `mom_min`.
    In the second phase the LR decreases from `lr_max` to `lr_max / (div_factor * 1e4)` and momemtum from `mom_max` to `mom_min`.
    By default the phases are not of equal length, with the phase 1 percentage controlled by the parameter `phase_1_pct`.
    """

    def __init__(self, lr_max, steps, wd=1e-2, mom_min=0.85, mom_max=0.95, phase_1_pct=0.3, div_factor=25.):
        super(OneCycleScheduler, self).__init__()
        lr_min = lr_max / div_factor
        final_lr = lr_max / (div_factor * 1e4)
        phase_1_steps = steps * phase_1_pct
        phase_2_steps = steps - phase_1_steps
        self.wd = wd
        
        self.phase_1_steps = phase_1_steps
        self.phase_2_steps = phase_2_steps
        self.phase = 0
        self.step = 0
        
        self.phases = [[CosineAnnealer(lr_min, lr_max, phase_1_steps), CosineAnnealer(mom_max, mom_min, phase_1_steps)], 
                 [CosineAnnealer(lr_max, final_lr, phase_2_steps), CosineAnnealer(mom_min, mom_max, phase_2_steps)]]
        
        self.lrs = []
        self.moms = []

    def on_train_begin(self, logs=None):
        self.phase = 0
        self.step = 0

        self.set_lr(self.lr_schedule().start)
        self.set_momentum(self.mom_schedule().start)
        
    def on_train_batch_begin(self, batch, logs=None):
        self.lrs.append(self.get_lr())
        self.moms.append(self.get_momentum())

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.step >= self.phase_1_steps:
            self.phase = 1
            
        self.set_lr(self.lr_schedule().step())
        self.set_momentum(self.mom_schedule().step())
        
    def get_lr(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.lr)
        except AttributeError:
            return None
        
    def get_momentum(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.momentum)
        except AttributeError:
            return None
        
    def set_lr(self, lr):
        try:
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
            # tf.keras.backend.set_value(self.model.optimizer.weight_decay, self.wd)
        except AttributeError:
            pass # ignore
        
    def set_momentum(self, mom):
        try:
            tf.keras.backend.set_value(self.model.optimizer.momentum, mom)
        except AttributeError:
            pass # ignore

    def lr_schedule(self):
        return self.phases[self.phase][0]
    
    def mom_schedule(self):
        return self.phases[self.phase][1]
    
    def plot(self):
        ax = plt.figure()
        plt.plot(self.lrs)
        plt.plot(self.mom)























