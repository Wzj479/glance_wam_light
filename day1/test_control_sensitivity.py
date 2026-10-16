"""Protocol checks: delays must flush the full sequence without horizon bias."""
import unittest
import numpy as np
from control_sensitivity_cpu import schedule


class TimingProtocolTests(unittest.TestCase):
    def setUp(self):
        self.actions=np.zeros((12,7))
        self.actions[:,0]=np.arange(12)/20
        self.actions[:,6]=[-1]*4+[1]*4+[-1]*4

    def test_whole_delay_preserves_every_action_and_common_horizon(self):
        baseline=schedule(self.actions,'baseline',0)
        shifted=schedule(self.actions,'all_delay',4)
        self.assertEqual(baseline.shape,shifted.shape)
        np.testing.assert_array_equal(shifted[4:16],self.actions)
        np.testing.assert_array_equal(shifted[:4,:6],np.zeros((4,6)))
        np.testing.assert_array_equal(shifted[:4,6],[-1]*4)

    def test_gripper_delay_does_not_delay_arm_commands(self):
        delayed=schedule(self.actions,'gripper_delay',2)
        np.testing.assert_array_equal(delayed[:12,:6],self.actions[:,:6])
        np.testing.assert_array_equal(delayed[2:14,6],self.actions[:,6])
        self.assertEqual(np.flatnonzero(np.diff(delayed[:,6]))[0]+1,6)

    def test_hold_repeats_command_without_changing_physics_step_count(self):
        held=schedule(self.actions,'hold',4)
        self.assertEqual(len(held),len(self.actions)+24)
        np.testing.assert_array_equal(held[:4],np.repeat(self.actions[:1],4,axis=0))
        np.testing.assert_array_equal(held[4:8],np.repeat(self.actions[4:5],4,axis=0))
        np.testing.assert_array_equal(held[12:,:6],np.zeros((24,6)))


if __name__=='__main__':unittest.main()
