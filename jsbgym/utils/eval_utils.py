from weakref import ref
import numpy as np
from enum import IntEnum
from jsbgym.trim.trim_point import TrimPoint

class State(IntEnum):
    ROLL = 0
    PITCH = 1
    AIRSPEED = 2

class RefSequence():
    def __init__(self, num_refs: int=5, min_step_bound: int=300, max_step_bound: int=500,
                 roll_bound: float=45.0, pitch_bound: float=15.0, airspeed_bound: float=10.0):
        self.num_refs = num_refs
        self.min_step_bound = min_step_bound
        self.max_step_bound = max_step_bound
        self.pitch_bound = pitch_bound
        self.roll_bound = roll_bound
        self.airspeed_bound = airspeed_bound
        self.ref_steps = None
        self.ref_cnts = None
        self.roll_ref = 0.0
        self.pitch_ref = 0.0 
        self.airspeed_ref = TrimPoint().Va_kph


    def sample_steps(self, offset: int=0) -> None:
        self.ref_steps = np.ones((self.num_refs, 3), dtype=np.int32)
        self.ref_cnts = np.zeros(3, dtype=np.int8)  # one counter for each state ref : roll, pitch, airspeed

        # Generate the remaining print steps
        for fcs in range(3):
            for i in range(1, self.num_refs):
                last_step = self.ref_steps[i-1, fcs]
                next_step = np.random.randint(last_step + self.min_step_bound, 
                                              min(last_step + self.max_step_bound, 2000), 
                                              dtype=np.int16)
                self.ref_steps[i, fcs] = next_step

        self.ref_steps += offset  # Convert the steps to integers
        print(self.ref_steps)  # Print the steps


    def sample_refs(self, step: int, env_num: int=0):
        for state in State:
            if self.ref_cnts[state] < self.num_refs:
                if step % self.ref_steps[self.ref_cnts[state], state] == 0:
                    if state == State.ROLL:
                        self.roll_ref = np.deg2rad(np.random.uniform(-self.roll_bound, self.roll_bound))
                        print(f"env_num {env_num}, roll ref change @ step {step}: {self.roll_ref}")
                    elif state == State.PITCH:
                        self.pitch_ref = np.deg2rad(np.random.uniform(-self.pitch_bound, self.pitch_bound))
                        print(f"env_num {env_num}, pitch ref change @ step {step}: {self.pitch_ref}")
                    elif state == State.AIRSPEED:
                        self.airspeed_ref = np.random.uniform(TrimPoint().Va_kph - self.airspeed_bound, 
                                                              TrimPoint().Va_kph + self.airspeed_bound)
                    self.ref_cnts[state] += 1

        return self.roll_ref, self.pitch_ref, self.airspeed_ref


def main():
    np.random.seed(1)
    np.set_printoptions(suppress = True)
    ref_seq = RefSequence()
    ref_seq.sample_steps()
    total_steps = 200_000
    steps_per_episode = 2000
    n_episodes = total_steps // steps_per_episode
    step_seq_arr: np.ndarray = np.zeros((n_episodes, ref_seq.num_refs, 3), dtype=np.int32)
    ref_seq_arr: np.ndarray = np.zeros((total_steps, 3), dtype=np.float32)

    step_seq_arr[0] = ref_seq.ref_steps

    # save the reference steps and values to a file
    for i in range(total_steps):
        ref_seq.sample_refs(i)
        ref_seq_arr[i] = np.array([ref_seq.roll_ref, ref_seq.pitch_ref, ref_seq.airspeed_ref])
        if i % 2000 == 0:
            ref_seq.sample_steps(i)
            print(i // 2000)
            step_seq_arr[i // 2000] = ref_seq.ref_steps # save the reference steps

    # print(step_seq_arr)
    # print(ref_seq_arr)

    np.save("step_seq_arr.npy", step_seq_arr)
    np.save("ref_seq_arr.npy", ref_seq_arr)

    # read the reference steps and values from a file
    step_seq_arr = np.load("step_seq_arr.npy")
    ref_seq_arr = np.load("ref_seq_arr.npy")
    print(step_seq_arr)
    print(ref_seq_arr)

if __name__ == "__main__":
    main()