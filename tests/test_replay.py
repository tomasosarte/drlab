import unittest

import numpy as np
import torch as th

from drlab.replay import ReplayBuffer, TransitionBatch


class TransitionBatchTest(unittest.TestCase):
    def test_to_returns_batch_on_requested_device(self):
        batch = TransitionBatch(
            states=th.zeros(2, 3),
            actions=th.zeros(2, 1, dtype=th.long),
            rewards=th.ones(2, 1),
            dones=th.zeros(2, 1, dtype=th.bool),
            next_states=th.ones(2, 3),
            returns=th.full((2, 1), 2.0),
        )

        moved = batch.to("cpu")

        self.assertIsInstance(moved, TransitionBatch)
        self.assertEqual(moved.states.device.type, "cpu")
        self.assertTrue(th.equal(moved.actions, batch.actions))

    def test_to_converts_numpy_fields_to_tensors(self):
        batch = TransitionBatch(
            states=np.zeros((2, 3), dtype=np.float32),
            actions=np.zeros((2, 1), dtype=np.int64),
            rewards=np.ones((2, 1), dtype=np.float32),
            dones=np.zeros((2, 1), dtype=np.bool_),
            next_states=np.ones((2, 3), dtype=np.float32),
            returns=np.full((2, 1), 2.0, dtype=np.float32),
        )

        moved = batch.to("cpu")

        self.assertEqual(moved.states.dtype, th.float32)
        self.assertEqual(moved.actions.dtype, th.int64)
        self.assertEqual(moved.dones.dtype, th.bool)
        self.assertTrue(th.equal(moved.returns, th.full((2, 1), 2.0)))

    def test_cat_concatenates_every_field(self):
        first = TransitionBatch(
            states=th.zeros(1, 2),
            actions=th.zeros(1, 1, dtype=th.long),
            rewards=th.ones(1, 1),
            dones=th.zeros(1, 1, dtype=th.bool),
            next_states=th.ones(1, 2),
            returns=th.ones(1, 1),
        )
        second = TransitionBatch(
            states=th.ones(2, 2),
            actions=th.ones(2, 1, dtype=th.long),
            rewards=th.full((2, 1), 2.0),
            dones=th.ones(2, 1, dtype=th.bool),
            next_states=th.full((2, 2), 3.0),
            returns=th.full((2, 1), 4.0),
        )

        combined = first.cat(second)

        self.assertEqual(combined.states.shape, (3, 2))
        self.assertEqual(combined.actions.shape, (3, 1))
        self.assertTrue(th.equal(combined.rewards.squeeze(-1), th.tensor([1.0, 2.0, 2.0])))


class ReplayBufferTest(unittest.TestCase):
    def test_add_get_all_and_sample_return_transition_batches(self):
        buffer = ReplayBuffer(capacity=4, obs_shape=(2,), device="cpu")
        states = np.asarray([[0, 1], [2, 3], [4, 5]], dtype=np.float32)
        next_states = states + 1

        buffer.add(
            states=states,
            actions=np.asarray([0, 1, 0]),
            rewards=np.asarray([1.0, 2.0, 3.0]),
            dones=np.asarray([False, False, True]),
            next_states=next_states,
            returns=np.asarray([6.0, 5.0, 3.0]),
        )

        all_data = buffer.get_all()
        sample = buffer.sample(2)

        self.assertEqual(len(buffer), 3)
        self.assertIsInstance(all_data, TransitionBatch)
        self.assertEqual(all_data.states.shape, (3, 2))
        self.assertEqual(all_data.actions.dtype, th.int64)
        self.assertEqual(all_data.dones.dtype, th.bool)
        self.assertIsInstance(sample, TransitionBatch)
        self.assertEqual(sample.states.shape, (2, 2))

    def test_add_wraps_when_batch_crosses_capacity(self):
        buffer = ReplayBuffer(capacity=3, obs_shape=(1,), device="cpu")

        buffer.add(
            states=np.asarray([[0], [1]], dtype=np.float32),
            actions=np.asarray([0, 1]),
            rewards=np.asarray([0.0, 1.0]),
            dones=np.asarray([False, False]),
            next_states=np.asarray([[1], [2]], dtype=np.float32),
            returns=np.asarray([1.0, 1.0]),
        )
        buffer.add(
            states=np.asarray([[2], [3]], dtype=np.float32),
            actions=np.asarray([0, 1]),
            rewards=np.asarray([2.0, 3.0]),
            dones=np.asarray([False, True]),
            next_states=np.asarray([[3], [4]], dtype=np.float32),
            returns=np.asarray([5.0, 3.0]),
        )

        all_data = buffer.get_all()

        self.assertEqual(len(buffer), 3)
        self.assertEqual(all_data.states.shape, (3, 1))
        self.assertCountEqual(all_data.states.squeeze(-1).tolist(), [1.0, 2.0, 3.0])

    def test_sample_empty_buffer_raises(self):
        buffer = ReplayBuffer(capacity=3, obs_shape=(2,), device="cpu")

        with self.assertRaisesRegex(ValueError, "empty buffer"):
            buffer.sample(1)
