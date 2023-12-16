import numpy as np
from jsbgym.utils.eval_utils import State, StateNoVa

SUCCESS_THR = 200 # probably gotta increase this since I change refs around every 500

def compute_success(errors):
    print("COMPUTING SUCCESS")
    successes = [[],[]]
    for state_id, errors_per_state in zip(StateNoVa, errors):
        if state_id == StateNoVa.ROLL:
            print("  ROLL")
        elif state_id == StateNoVa.PITCH:
            print("  PITCH")
        for ref, errors_per_st_per_ref in enumerate(errors_per_state):
            success_streak = 0
            for step, error in enumerate(np.flipud(errors_per_st_per_ref)):
                ref_len = len(errors_per_st_per_ref)
                if abs(error) < np.deg2rad(5):
                    success_streak += 1
                else:
                    success_streak = 0
                    successes[state_id].append(False)
                    print(f"    ref {ref} failed at step {ref_len - step}")
                    break
                if success_streak >= SUCCESS_THR:
                    successes[state_id].append(True)
                    print(f"    ref {ref} success at step {ref_len - step}")
                    break
    return successes


def compute_steady_state(errors):
    print("COMPUTING SS ERRORS AND SETTLING TIMES")
    ss_errors = [[],[]]
    settling_times = [[],[]]
    for state_id, errors_per_state in zip(StateNoVa, errors):
        if state_id == StateNoVa.ROLL:
            print("  ROLL")
        elif state_id == StateNoVa.PITCH:
            print("  PITCH")
        for ref, errors_per_st_per_ref in enumerate(errors_per_state):
            for step, error in enumerate(np.flipud(errors_per_st_per_ref)):
                ref_len = len(errors_per_st_per_ref)
                if abs(error) > np.deg2rad(5):
                    settling_times[state_id].append(ref_len - step)
                    ss_error = np.mean(errors_per_st_per_ref[-step:])
                    ss_errors[state_id].append(ss_error)
                    print(f"    ref {ref} reached ss at step {ref_len - step}, ss error: {ss_error}")
                    break
                if (np.abs(errors_per_st_per_ref) <= np.deg2rad(5)).all():
                    settling_times[state_id].append(0)
                    ss_error = np.mean(errors_per_st_per_ref)
                    ss_errors[state_id].append(ss_error)
                    print(f"    ref {ref} reached ss at step {0}, ss error: {ss_error}")
                    break
    return ss_errors, settling_times


# function to rearrange the errors into a 3D list where each sublist is the errors for a ref
# output error "shape": (num_states=2, num_refs, ref_length)
def split_errors(obs, steps): 
    errors = obs[:, 6:8]
    steps = steps[:2]
    steps = np.reshape(steps, (-1, steps.shape[-1]))
    errors_per_ref = [[], []]
    for state_id, steps_per_state in zip(StateNoVa, steps.T):
        for i, ref_bound_step in enumerate(steps_per_state[:-1]): # don't take last ref on purpose to avoid out of bounds error
            # one_ref = []
            ref_begin = ref_bound_step
            ref_end = steps_per_state[i+1]
            error_per_ref = errors[ref_begin:ref_end, state_id]
            # one_ref.append(error_per_ref)
            errors_per_ref[state_id].append(error_per_ref)
    return errors_per_ref # can't return as numpy array because each ref has different length (inhomogeneous)

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
    splitted_errors = split_errors(ppo_obs, steps)
    compute_success(splitted_errors)
    compute_steady_state(splitted_errors)

    print("********** PID METRICS **********")
    pid_obs = np.load("e_pid_obs.npy")
    pid_act = np.load("e_pid_actions.npy")
    print(pid_obs.shape)
    splitted_errors = split_errors(pid_obs, steps)
    compute_success(splitted_errors)
    compute_steady_state(splitted_errors)

if __name__ == "__main__":
    main()