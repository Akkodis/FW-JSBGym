import numpy as np
from jsbgym.utils.eval_utils import State

SUCCESS_THR = 100

def compute_metrics(obs, refs, steps):
    states = obs[:, :5]
    errors = obs[:, 5:7]
    steps = steps[:2]
    steps = np.reshape(steps, (-1, steps.shape[-1]))
    for state_id, steps_per_state in zip(State, steps.T):
        if state_id == State.ROLL:
            print(steps_per_state)
            for i, step in enumerate(steps_per_state[:-1]):
                ref_begin = step
                ref_end = steps_per_state[i+1]
                error_per_ref = errors[ref_begin:ref_end, state_id]
                print(np.rad2deg(error_per_ref))
                success_cnt = 0
                for j, error in enumerate(error_per_ref):
                    if abs(error) < np.deg2rad(15): # if error is less than 5 degrees
                        success_cnt += 1
                    else: # if error is not less than 5 degrees, reset success_cnt
                        success_cnt = 0
                    if success_cnt >= SUCCESS_THR:
                        print(f"roll success at {i}")
                        break
        if state_id == State.PITCH:
            print(steps_per_state)

def main():
    obs = np.load("e_obs.npy")
    act = np.load("e_actions.npy")
    refs = np.load("ref_seq_arr.npy")
    steps = np.load("step_seq_arr.npy")
    compute_metrics(obs, refs, steps)

if __name__ == "__main__":
    main()