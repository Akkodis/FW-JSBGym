import numpy as np
from jsbgym.utils.eval_utils import State, StateNoVa

SUCCESS_THR = 100 # probably gotta increase this since I change refs around every 500

def compute_success(errors):
    # print("COMPUTING SUCCESS")
    successes = [[],[]]
    for state_id, errors_per_state in zip(StateNoVa, errors):
        # if state_id == StateNoVa.ROLL:
        #     print("  ROLL")
        # elif state_id == StateNoVa.PITCH:
        #     print("  PITCH")
        # for ref, errors_per_st_per_ref in enumerate(errors_per_state):
        #     success_streak = 0
        #     for step, error in enumerate(np.flipud(errors_per_st_per_ref)):
        #         ref_len = len(errors_per_st_per_ref)
        #         if abs(error) < np.deg2rad(5):
        #             success_streak += 1
        #         else:
        #             success_streak = 0
        #             successes[state_id].append(False)
        #             # print(f"    ref {ref} failed at step {ref_len - step}")
        #             break
        #         if success_streak >= SUCCESS_THR:
        #             successes[state_id].append(True)
        #             # print(f"    ref {ref} success at step {ref_len - step}")
        #             break
        for ref, errors_per_st_per_ref in enumerate(errors_per_state):
            success_streak = 0
            for step, error in enumerate(errors_per_st_per_ref):
                ref_len = len(errors_per_st_per_ref)
                if abs(error) < np.deg2rad(5):
                    success_streak += 1
                if success_streak >= SUCCESS_THR:
                    successes[state_id].append(True)
                    print(f"    ref {ref} success at step {step}")
                    success_streak = 0
                    break
                if step == ref_len - 1 and success_streak < SUCCESS_THR:
                    successes[state_id].append(False)
                    print(f"    ref {ref} failed at step {step}")
                    success_streak = 0
                    break

    successes = np.array(successes)
    successes_metric = np.nanmean(successes, axis=1)
    return successes, successes_metric


def compute_steady_state(errors):
    # print("COMPUTING SS ERRORS AND SETTLING TIMES")
    ss_errors = [[],[]]
    settling_times = [[],[]]
    for state_id, errors_per_state in zip(StateNoVa, errors):
        # if state_id == StateNoVa.ROLL:
        #     print("  ROLL")
        # elif state_id == StateNoVa.PITCH:
        #     print("  PITCH")
        for ref, errors_per_st_per_ref in enumerate(errors_per_state):
            for step, error in enumerate(np.flipud(errors_per_st_per_ref)):
                ref_len = len(errors_per_st_per_ref)
                if abs(error) > np.deg2rad(5): # traversing backwards so this is the first time error is out of ss bounds
                    settling_times[state_id].append(ref_len - step) # settling time is the step at which it first goes out of bounds
                    ss_error = np.mean(errors_per_st_per_ref[-step:]) # ss error is the mean of the last n steps before going out of bounds
                    ss_errors[state_id].append(ss_error)
                    # print(f"    ref {ref} reached ss at step {ref_len - step}, ss error: {ss_error}")
                    break
                if (np.abs(errors_per_st_per_ref) <= np.deg2rad(5)).all(): # if it never goes out of bounds
                    settling_times[state_id].append(0) # settling time is 0
                    ss_error = np.mean(errors_per_st_per_ref) # ss error is the mean of the whole ref traj
                    ss_errors[state_id].append(ss_error)
                    # print(f"    ref {ref} reached ss at step {0}, ss error: {ss_error}")
                    break
    ss_errors = np.array(ss_errors)
    settling_times = np.array(settling_times) * 0.01 # convert to seconds
    ss_errors_metric = np.mean(np.abs(ss_errors), axis=1)
    settling_times_metric = np.mean(settling_times, axis=1)
    return ss_errors, settling_times, ss_errors_metric, settling_times_metric


def compute_rise_time(errors, ss_errors):
    # print("COMPUTING RISE TIMES")
    rise_times = [[],[]]
    for state_id, errors_per_state in zip(StateNoVa, errors):
        # if state_id == StateNoVa.ROLL:
        #     print("  ROLL")
        # elif state_id == StateNoVa.PITCH:
        #     print("  PITCH")
        for ref, errors_per_st_per_ref in enumerate(errors_per_state):
            initial_error = errors_per_st_per_ref[0]
            rise_end = 0.0
            rise_start = 0.0
            ref_len = len(errors_per_st_per_ref)
            for step, error in enumerate(np.flipud(errors_per_st_per_ref)):
                error_to_ss = np.abs(error - ss_errors[state_id][ref])
                if step > 0:
                    prev_error = np.abs(errors_per_st_per_ref[-step] - ss_errors[state_id][ref])
                    low_lim = np.abs(0.1 * initial_error)
                    high_lim = np.abs(0.9 * initial_error)
                    if error_to_ss >= low_lim and prev_error < low_lim:
                        rise_end = ref_len - step
                    if error_to_ss >= high_lim and prev_error < high_lim:
                        rise_start = ref_len - step
            rise_time = np.abs(rise_end - rise_start) * 0.01 # sometimes due to turb the rise end is before the rise start + convert to seconds
            rise_times[state_id].append(rise_time)
            # print(f"    ref {ref} rise time: {rise_time}")
    rise_times = np.array(rise_times)
    rise_times_metric = np.nanmean(rise_times, axis=1)
    return rise_times, rise_times_metric


# maybe I should find the find local max/min of the error and then compute the overshoot from there
# because roll / pitch are coupled and when there's a big command in roll, it shows up as a big error in pitch
# which is here considered as overshoot (metric code from Bohn's repo)
def compute_overshoot(errors):
    # print("COMPUTING OVERSHOOT")
    overshoots = [[],[]]
    for state_id, errors_per_state in zip(StateNoVa, errors):
        # if state_id == StateNoVa.ROLL:
        #     print("  ROLL")
        # elif state_id == StateNoVa.PITCH:
        #     print("  PITCH")
        for ref, errors_per_st_per_ref in enumerate(errors_per_state):
            initial_error = errors_per_st_per_ref[0]
            op = getattr(np, "min" if initial_error > 0 else "max")
            max_opposite_error = op(errors_per_st_per_ref, axis=0)
            if np.sign(max_opposite_error) == np.sign(initial_error):
                overshoot = np.nan
                # print(f"    ref {ref} overshoot: {overshoot}")
            else:
                overshoot = np.abs(np.abs(max_opposite_error / initial_error)-1) * 100
                # print(f"    ref {ref} overshoot: {overshoot}")
            overshoots[state_id].append(overshoot)
    overshoots = np.array(overshoots)
    overshoots_metric = np.nanmean(overshoots, axis=1)
    return overshoots, overshoots_metric


def compute_angular_variation(obs):
    ang_rate = obs[:, 3:6]
    angular_variance = np.var(ang_rate, axis=0)
    # print(f"    angular variance: {angular_variance}")
    return angular_variance


def compute_control_variation(actions):
    control_variance = np.var(actions, axis=0)
    # print(f"    control variance: {control_variance}")
    return control_variance


# function to rearrange the errors into a 3D list where each sublist is the errors for a ref
# output error "shape": (num_states=2, num_refs, ref_length)
def split_errors(obs, steps): 
    errors = obs[:, 6:8]
    states = obs[:, 0:2]
    steps = steps[:2]
    steps = np.reshape(steps, (-1, steps.shape[-1]))
    errors_per_ref = [[], []]
    obss_per_ref = [[], []]
    for state_id, steps_per_state in zip(StateNoVa, steps.T):
        for i, ref_bound_step in enumerate(steps_per_state[:-1]): # don't take last ref on purpose to avoid out of bounds error
            ref_begin = ref_bound_step
            ref_end = steps_per_state[i+1]
            error_per_ref = errors[ref_begin:ref_end, state_id]
            errors_per_ref[state_id].append(error_per_ref)
            obs_per_ref = states[ref_begin:ref_end, state_id]
            obss_per_ref[state_id].append(obs_per_ref)
    return errors_per_ref, obss_per_ref # can't return as numpy array because each ref has different length (inhomogeneous)


def compute_all_metrics(obs, act, steps):
    splitted_errors, splitted_obs = split_errors(obs, steps)
    successes_arr, successes_metric = compute_success(splitted_errors)
    ss_errors_arr, settling_times_arr, ss_errors_metric, settling_times_metric = compute_steady_state(splitted_errors)
    rise_times, rise_times_metric = compute_rise_time(splitted_errors, ss_errors_arr)
    overshoots_arr, overshoots_metric = compute_overshoot(splitted_errors)
    angular_var = compute_angular_variation(obs)
    control_var = compute_control_variation(act)
    roll_mse = np.mean(np.square(obs[:, 6]))
    pitch_mse = np.mean(np.square(obs[:, 7]))
    ret_dict = {"successes": successes_metric,
                "roll_mse": roll_mse,
                "pitch_mse": pitch_mse,
                "ss_errors": ss_errors_metric,
                "settling_times": settling_times_metric,
                "rise_times": rise_times_metric,
                "overshoots": overshoots_metric,
                "angular_var": angular_var,
                "control_var": control_var}
    # print(f"    successes: {successes_metric}\n"
    #       f"    ss errors: {ss_errors_metric}\n"
    #       f"    settling times: {settling_times_metric}\n"
    #       f"    rise times: {rise_times_metric}\n"
    #       f"    overshoots: {overshoots_metric}\n"
    #       f"    angular var: {angular_var}\n"
    #       f"    control var: {control_var}\n")
    return ret_dict



def main():
    np.set_printoptions(suppress=True)
    refs = np.load("ref_seq_arr.npy")
    steps = np.load("step_seq_arr.npy")
    
    print("********** PPO METRICS **********")
    ppo_obs = np.load("e_ppo_obs.npy")
    ppo_act = np.load("e_ppo_actions.npy")
    compute_all_metrics(ppo_obs, ppo_act, steps)
    # splitted_errors, splitted_obs = split_errors(ppo_obs, steps)
    # successes_arr, successes_metric = compute_success(splitted_errors)
    # ss_errors_arr, settling_times_arr, ss_errors_metric, settling_times_metric = compute_steady_state(splitted_errors)
    # rise_times, rise_times_metric = compute_rise_time(splitted_errors, ss_errors_arr)
    # overshoots_arr, overshoots_metric = compute_overshoot(splitted_errors)
    # angular_var = compute_angular_variation(ppo_obs)
    # control_var = compute_control_variation(ppo_act)
    # print(f"    successes: {successes_metric}\n"
    #       f"    ss errors: {ss_errors_metric}\n"
    #       f"    settling times: {settling_times_metric}\n"
    #       f"    rise times: {rise_times_metric}\n"
    #       f"    overshoots: {overshoots_metric}\n"
    #       f"    angular var: {angular_var}\n"
    #       f"    control var: {control_var}\n")

    print("********** PID METRICS **********")
    pid_obs = np.load("e_pid_obs.npy")
    pid_act = np.load("e_pid_actions.npy")
    compute_all_metrics(pid_obs, pid_act, steps)
    # splitted_errors, splitted_obs = split_errors(pid_obs, steps)
    # successes_arr, successes_metric = compute_success(splitted_errors)
    # ss_errors_arr, settling_times_arr, ss_errors_metric, settling_times_metric = compute_steady_state(splitted_errors)
    # rise_times, rise_times_metric = compute_rise_time(splitted_errors, ss_errors_arr)
    # overshoots_arr, overshoots_metric = compute_overshoot(splitted_errors)
    # angular_var = compute_angular_variation(pid_obs)
    # control_var = compute_control_variation(pid_act)
    # print(f"    successes: {successes_metric}\n"
    #       f"    ss errors: {ss_errors_metric}\n"
    #       f"    settling times: {settling_times_metric}\n"
    #       f"    rise times: {rise_times_metric}\n"
    #       f"    overshoots: {overshoots_metric}\n"
    #       f"    angular var: {angular_var}\n"
    #       f"    control var: {control_var}\n")


if __name__ == "__main__":
    main()