import torch
import torch.multiprocessing as mp
import tempfile
import gym

import numpy as np

class MultiEnv(object):
    def __init__(self, env, n_workers, seed=0):
        self.env = env
        self.seed = seed
        self.n_workers = n_workers
        self.pipes = [mp.Pipe() for rank in range(self.n_workers)]
        self.workers = [
            mp.Process(
                target=self.work, 
                args=(rank, self.pipes[rank][1])) for rank in range(self.n_workers)]
        [w.start() for w in self.workers]
        self.dones = {rank:False for rank in range(self.n_workers)}

    def reset(self, rank=None, **kwargs):
        if rank is not None:
            parent_end, _ = self.pipes[rank]
            self.send_msg(('reset', {}), rank)
            o = parent_end.recv()
            return o

        self.broadcast_msg(('reset', kwargs))
        return np.vstack([parent_end.recv() for parent_end, _ in self.pipes])

    def step(self, actions):
        assert len(actions) == self.n_workers
        [self.send_msg(
            ('step', {'action':actions[rank]}), 
            rank) for rank in range(self.n_workers)]
        results = []
        for rank in range(self.n_workers):
            parent_end, _ = self.pipes[rank]
            o, r, d, i = parent_end.recv()
            results.append((o, 
                            np.array(r, dtype=np.float), 
                            np.array(d, dtype=np.float), 
                            i))
        return [np.vstack(block) for block in np.array(results).T]

    def close(self, **kwargs):
        self.broadcast_msg(('close', kwargs))
        [w.join() for w in self.workers]

    def _past_limit(self, **kwargs):
        self.broadcast_msg(('_past_limit', kwargs))
        return np.vstack([parent_end.recv() for parent_end, _ in self.pipes])
    
    def work(self, rank, worker_end):
        env = self.env
        env.seed(self.seed + rank)
        while True:
            cmd, kwargs = worker_end.recv()
            if cmd == 'reset':
                worker_end.send(env.reset(**kwargs))
            elif cmd == 'step':
                worker_end.send(env.step(**kwargs))
            elif cmd == '_past_limit':
                worker_end.send(env._elapsed_steps >= env._max_episode_steps)
            else:
                # including close command 
                env.close(**kwargs) ; del env ; worker_end.close()
                break

    def send_msg(self, msg, rank):
        parent_end, _ = self.pipes[rank]
        parent_end.send(msg)

    def broadcast_msg(self, msg):    
        [parent_end.send(msg) for parent_end, _ in self.pipes]
