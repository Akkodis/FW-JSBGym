import numpy as np
from jsbgym.utils.eval_utils import State

SUCCESS_THR = 200 # probably gotta increase this since I change refs around every 500

def compute_metrics(obs, refs, steps):
    states = obs[:, :5]
    errors = obs[:, 6:8]
    steps = steps[:2]
    steps = np.reshape(steps, (-1, steps.shape[-1]))
    for state_id, steps_per_state in zip(State, steps.T):
        if state_id == State.ROLL:
            print(steps_per_state)
            for i, step in enumerate(steps_per_state[:-1]):
                ref_begin = step
                ref_end = steps_per_state[i+1]
                print(ref_begin, ref_end)
                error_per_ref = errors[ref_begin:ref_end, state_id]
                success_cnt = 0
                for j, error in enumerate(np.flipud(error_per_ref)):
                    if abs(error) < np.deg2rad(5): # if error is less than 5 degrees
                        success_cnt += 1
                    else: # if error is not less than 5 degrees, reset success_cnt
                        success_cnt = 0
                        print(f"roll failed at {i}th ref, step {j}")
                        break
                    if success_cnt >= SUCCESS_THR:
                        print(f"roll success at {i}th ref, step {j}")
                        break
        if state_id == State.PITCH:
            print(steps_per_state)
            for i, step in enumerate(steps_per_state[:-1]):
                ref_begin = step
                ref_end = steps_per_state[i+1]
                error_per_ref = errors[ref_begin:ref_end, state_id]
                # print(error_per_ref)
                success_cnt = 0
                for j, error in enumerate(np.flipud(error_per_ref)):
                    if abs(error) < np.deg2rad(5):
                        success_cnt += 1
                    else:
                        success_cnt = 0
                        print(f"pitch failed at {i}th ref, step {j}")
                        break
                    if success_cnt >= SUCCESS_THR:
                        print(f"pitch success at {i}th ref, step {j}")
                        break

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
    compute_metrics(ppo_obs, refs, steps)

    print("********** PID METRICS **********")
    pid_obs = np.load("e_pid_obs.npy")
    pid_act = np.load("e_pid_actions.npy")
    print(pid_obs.shape)
    compute_metrics(pid_obs, refs, steps)

if __name__ == "__main__":
    main()