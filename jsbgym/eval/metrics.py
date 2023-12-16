from email import errors
import numpy as np
from jsbgym.utils.eval_utils import State

SUCCESS_THR = 200 # probably gotta increase this since I change refs around every 500

def compute_success(obs, refs, steps):
    states = obs[:, :5]
    errors = obs[:, 6:8]
    steps = steps[:2]
    ss_errors = [[],[]]
    successes = [[],[]]
    steps = np.reshape(steps, (-1, steps.shape[-1]))
    for state_id, steps_per_state in zip(State, steps.T):
        if state_id == State.ROLL:
            print(steps_per_state)
            for i, step in enumerate(steps_per_state[:-1]):
                ref_begin = step
                ref_end = steps_per_state[i+1]
                error_per_ref = errors[ref_begin:ref_end, state_id]
                success_cnt = 0
                for j, error in enumerate(np.flipud(error_per_ref)):
                    ref_len = len(error_per_ref)
                    if abs(error) < np.deg2rad(5): # if error is less than 5 degrees
                        success_cnt += 1
                    else: # if error is not less than 5 degrees, reset success_cnt
                        success_cnt = 0
                        print(f"roll failed at {i}th ref, step {ref_len - j}")
                        successes[state_id].append(False)
                        break
                    if success_cnt >= SUCCESS_THR:
                        print(f"roll success at {i}th ref, step {ref_len - j}")
                        successes[state_id].append(True)
                        break
        if state_id == State.PITCH:
            print(steps_per_state)
            for i, step in enumerate(steps_per_state[:-1]):
                ref_begin = step
                ref_end = steps_per_state[i+1]
                error_per_ref = errors[ref_begin:ref_end, state_id]
                success_cnt = 0
                for j, error in enumerate(np.flipud(error_per_ref)):
                    if abs(error) < np.deg2rad(5):
                        success_cnt += 1
                    else:
                        success_cnt = 0
                        print(f"pitch failed at {i}th ref, step {ref_len - j}")
                        successes[state_id].append(False)
                        break
                    if success_cnt >= SUCCESS_THR:
                        print(f"pitch success at {i}th ref, step {ref_len - j}")
                        successes[state_id].append(True)
                        break
    return successes


def compute_ss_error(obs, refs, steps):
    states = obs[:, :5]
    errors = obs[:, 6:8]
    steps = steps[:2]
    ss_errors = [[],[]]
    settling_times = [[],[]]
    steps = np.reshape(steps, (-1, steps.shape[-1]))
    for state_id, steps_per_state in zip(State, steps.T):
        if state_id == State.ROLL:
            print("ROLL")
            for i, bound_step in enumerate(steps_per_state[:-1]):
                ref_begin = bound_step
                ref_end = steps_per_state[i+1]
                error_per_ref = errors[ref_begin:ref_end, state_id]
                print(f"ref_begin: {ref_begin}, ref_end: {ref_end}")
                for step, error in enumerate(np.flipud(error_per_ref)):
                    ref_len = len(error_per_ref)
                    if abs(error) > np.deg2rad(5):
                        print(f"{i}th ref, reached ss at step: {ref_len - step}")
                        settling_times[state_id].append(ref_len - step)
                        ss_error = np.mean(error_per_ref[-step:])
                        ss_errors[state_id].append(ss_error)
                        break
                    if (np.abs(error_per_ref) <= np.deg2rad(5)).all(): # if error always stays in bounds
                        print(f"{i}th ref, reached ss at step: {0}")
                        settling_times[state_id].append(0)
                        ss_error = np.mean(error_per_ref)
                        ss_errors[state_id].append(ss_error)
                        break
        if state_id == State.PITCH:
            print("PITCH")
            for i, bound_step in enumerate(steps_per_state[:-1]):
                ref_begin = bound_step
                ref_end = steps_per_state[i+1]
                error_per_ref = errors[ref_begin:ref_end, state_id]
                print(f"ref_begin: {ref_begin}, ref_end: {ref_end}")
                for step, error in enumerate(np.flipud(error_per_ref)):
                    ref_len = len(error_per_ref)
                    if abs(error) > np.deg2rad(5):
                        print(f"{i}th ref, reached ss at step: {ref_len - step}")
                        settling_times[state_id].append(ref_len - step)
                        ss_error = np.mean(error_per_ref[-step:])
                        ss_errors[state_id].append(ss_error)
                        break
                    if (np.abs(error_per_ref) <= np.deg2rad(5)).all(): # if error always stays in bounds
                        print(f"{i}th ref, reached ss at step: {0}")
                        settling_times[state_id].append(0)
                        ss_error = np.mean(error_per_ref)
                        ss_errors[state_id].append(ss_error)
                        break
    return ss_errors, settling_times



# TODO: now that I have my error bounds and success I can compute the steady state error by taking the mean of the last 200 steps
# then I can compute the rise time, settling time, and overshoot

def main():
    np.set_printoptions(suppress=True)
    refs = np.load("ref_seq_arr.npy")
    steps = np.load("step_seq_arr.npy")
    
    print("********** PPO METRICS **********")
    ppo_obs = np.load("e_ppo_obs.npy")
    ppo_act = np.load("e_ppo_actions.npy")
    print(ppo_obs.shape)
    compute_success(ppo_obs, refs, steps)
    compute_ss_error(ppo_obs, refs, steps)

    print("********** PID METRICS **********")
    pid_obs = np.load("e_pid_obs.npy")
    pid_act = np.load("e_pid_actions.npy")
    print(pid_obs.shape)
    compute_success(pid_obs, refs, steps)
    compute_ss_error(pid_obs, refs, steps)

if __name__ == "__main__":
    main()