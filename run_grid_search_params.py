import numpy as np
import pandas as pd
import os, sys, config
from collections import namedtuple
import matplotlib.pyplot as plt
import tqdm

# Helper functions
def compute_backlog_threshold(vector, B):
    """Find the threshold index for decisions based on the backlog."""
    try:
        return vector.index('e')  # Find the first 'e' index
    except ValueError:
        return B  # Default to full buffer

def power_method(P, num_states, tol=1e-10, max_iter=100000):
    """Solve for steady-state probabilities using the power method."""
    pi = np.ones(num_states) / num_states  # Uniform initial distribution
    for _ in range(max_iter):
        next_pi = pi @ P
        if np.linalg.norm(next_pi - pi, ord=1) < tol:
            return next_pi
        pi = next_pi
    raise ValueError("Power method did not converge")

def solve_markov_chain(params, tau):
    """Solve the SMDP using a Markov Chain approach."""
    B, mu_a, mu_b, lambda_rate, p_a, p_b = params

    num_states = 2 * B + 1
    states = [(0, "empty")] + [(i, "E") for i in range(1, B + 1)] + [(i, "N") for i in range(1, B + 1)]
    state_index = {state: idx for idx, state in enumerate(states)}

    Q = np.zeros((num_states, num_states))
    rewards = np.zeros(num_states)

    # Transition rate matrix and rewards setup
    for state in states:
        idx = state_index[state]
        if state == (0, "empty"):
            if tau == 0:
                Q[idx, state_index[(1, "E")]] = lambda_rate
            else:
                Q[idx, state_index[(1, "N")]] = lambda_rate
        elif state[1] == "E":
            i = state[0]
            if i < B:
                Q[idx, state_index[(i + 1, "E")]] = lambda_rate
            if i - 1 == 0:
                Q[idx, state_index[(0, "empty")]] = mu_a
            else:
                if i - 1 < tau:
                    Q[idx, state_index[(i - 1, "N")]] = mu_a
                else:
                    Q[idx, state_index[(i - 1, "E")]] = mu_a
            rewards[idx] += (1 - p_a) * mu_a
        elif state[1] == "N":
            i = state[0]
            if i < B:
                Q[idx, state_index[(i + 1, "N")]] = lambda_rate
            if i - 1 == 0:
                Q[idx, state_index[(0, "empty")]] = mu_b
            else:
                if i - 1 < tau:
                    Q[idx, state_index[(i - 1, "N")]] = mu_b
                else:
                    Q[idx, state_index[(i - 1, "E")]] = mu_b
            rewards[idx] += (1 - p_b) * mu_b

    for i in range(num_states):
        Q[i, i] = -np.sum(Q[i])

    max_rate = max(-Q.diagonal())
    P = Q / max_rate + np.eye(num_states)
    steady_state = power_method(P, num_states)

    P_full_buffer = steady_state[state_index[(B, "E")]] + steady_state[state_index[(B, "N")]]
    P_class_E = sum(steady_state[state_index[(i, "E")]] for i in range(1, B + 1))
    P_class_N = sum(steady_state[state_index[(i, "N")]] for i in range(1, B + 1))
    throughput = mu_a * P_class_E * (1 - p_a) + mu_b * P_class_N * (1 - p_b)
    misclassification_rate = mu_a * P_class_E * p_a + mu_b * P_class_N * p_b
    drop_rate = lambda_rate * P_full_buffer
    assert 0 <= drop_rate <= lambda_rate, "Drop rate should be between 0 and lambda_rate"
    assert 0 <= misclassification_rate <= lambda_rate, "Misclassification rate should be between 0 and lambda_rate"
    assert 0 <= throughput <= lambda_rate, "Throughput should be between 0 and lambda_rate"
    assert 0 <= P_full_buffer <= 1, "P_full_buffer should be between 0 and 1"
    assert abs(throughput+misclassification_rate+drop_rate - lambda_rate) < 10e-8, f"Flow in should equal flow out: ({throughput}+{misclassification_rate}+{drop_rate} - {lambda_rate}) =  ({throughput+misclassification_rate+drop_rate - lambda_rate})"

    decision_vector = ['e' if i >= tau else 'n' for i in range(B)]
    return P_full_buffer, P_class_E, P_class_N, throughput, misclassification_rate, drop_rate, decision_vector, compute_backlog_threshold(decision_vector, B)



def multi_operator_value_iteration(params, gamma):
    """
    Implements Algorithm 1 with post-convergence integrity checks for mathematical equations.

    Args:
        params (tuple): Contains B (int), mu_a (float), mu_b (float), lambda_rate (float), p_a (float), p_b (float).
        gamma (float): Discount factor.

    Returns: Final value function, decision vector, and threshold.
    """
    import numpy as np

    B, mu_a, mu_b, lambda_rate, p_a, p_b = params
    max_iterations = 100000
    tolerance = 1e-10
    check_tolerance = 1e-8  # Tolerance for integrity checks

    # Initialize value functions
    U = np.random.rand(B)
    U_a_A = np.random.rand(B + 1)  # Arrival state values for "a"
    U_b_A = np.random.rand(B + 1)  # Arrival state values for "b"
    U_a_D = np.random.rand(B)      # Departure state values for "a"
    U_b_D = np.random.rand(B)      # Departure state values for "b"

    # Rewards
    c_a = (1 - p_a) * mu_a / (gamma + mu_a)
    c_b = (1 - p_b) * mu_b / (gamma + mu_b)

    # Precompute factors
    delta_a = 1 / (mu_a + lambda_rate + gamma)
    delta_b = 1 / (mu_b + lambda_rate + gamma)
    delta_bar = 1 / (lambda_rate + gamma)

    for iteration in range(max_iterations):
        U_prev = U.copy()

        # Temporary arrays for updated values
        U_a_A_new = np.ones(B + 1)
        U_b_A_new = np.ones(B + 1)
        U_a_D_new = np.ones(B)
        U_b_D_new = np.ones(B)
        U_new = np.ones(B)

        # Update arrival values
        for n in range(B):
            U_a_A_new[n] = mu_a * delta_a * U[n] + lambda_rate * delta_a * U_a_A[n + 1]
            U_b_A_new[n] = mu_b * delta_b * U[n] + lambda_rate * delta_b * U_b_A[n + 1]

        U_a_A_new[0] = U_a_A_new[0] + c_a
        U_b_A_new[0] = U_b_A_new[0] + c_b


        # Boundary condition for arrival states
        U_a_A_new[B] = mu_a * delta_a * U[B - 1] + lambda_rate * delta_a * U_a_A[B]
        U_b_A_new[B] = mu_b * delta_b * U[B - 1] + lambda_rate * delta_b * U_b_A[B]

        # Update departure values
        for n in range(1, B):
            U_a_D_new[n] = mu_a * delta_a * U[n - 1] + lambda_rate * delta_a * U_a_A[n] + c_a
            U_b_D_new[n] = mu_b * delta_b * U[n - 1] + lambda_rate * delta_b * U_b_A[n] + c_b

        # Update overall value function
        for n in range(1, B):
            U_new[n] = max(U_a_D[n], U_b_D[n])

        # Boundary condition for empty buffer
        U_new[0] = lambda_rate * delta_bar * max(U_a_A[0], U_b_A[0])
        U_b_D_new[0] = U_new[0]
        U_a_D_new[0] = U_new[0]

        # Check convergence
        if np.max(np.abs(U_new - U_prev)) < tolerance and iteration > 1000:
            break

        # Update old arrays with new values
        U_a_A = U_a_A_new.copy()
        U_b_A = U_b_A_new.copy()
        U_a_D = U_a_D_new.copy()
        U_b_D = U_b_D_new.copy()
        U = U_new.copy()

    # Post-convergence integrity checks
    #print("iteration: ", iteration)
    for n in range(B):
        if 1 <= n < B - 1:
            assert abs(U_a_A[n] - (mu_a * delta_a * U[n] + lambda_rate * delta_a * U_a_A[n + 1])) < check_tolerance, \
                f"Equation (1) not satisfied at n={n}"
        if 1 <= n < B:
            assert abs(U_a_D[n] - (mu_a * delta_a * U[n - 1] + lambda_rate * delta_a * U_a_A[n] + c_a)) < check_tolerance, \
                f"Equation (2) for 'a' not satisfied at n={n}"
            assert abs(U_b_D[n] - (mu_b * delta_b * U[n - 1] + lambda_rate * delta_b * U_b_A[n] + c_b)) < check_tolerance, \
                f"Equation (2) for 'b' not satisfied at n={n}"
        assert abs(U[n] - max(U_a_D[n], U_b_D[n])) < check_tolerance, \
            f"Equation (3) not satisfied at n={n}"

    # Check for V0 conditions
    assert abs(U_a_D[0] - U_b_D[0]) < check_tolerance, "V_a^D(0) != V_b^D(0)"
    assert abs(U[0] - U_a_D[0]) < check_tolerance, "V(0) != V_a^D(0)"
    assert abs(U[0] - lambda_rate * delta_bar * max(U_a_A[0], U_b_A[0])) < check_tolerance, \
        "V(0) != max(lambda_bar * delta_bar * V_a^A(0), lambda_bar * delta_bar * V_b^A(0))"

    # Generate decision vector
    decision_vector = ['e' if U_a_D[n] >= U_b_D[n] else 'n' for n in range(B)]
    decision_vector[0] = 'e' if U_a_A[0] >= U_b_A[0] else 'n'

    # Print all vectors
    #print("Final Value Function U:", U)
    #print("U_a_A (Arrival values for 'a'):", U_a_A)
    #print("U_b_A (Arrival values for 'b'):", U_b_A)
    #print("U_a_D (Departure values for 'a'):", U_a_D)
    #print("U_b_D (Departure values for 'b'):", U_b_D)
    #print("Decision Vector:", decision_vector)

    return U, decision_vector, compute_backlog_threshold(decision_vector, B)


def is_threshold_policy(policy):
    """
    Check if a policy is of threshold type.

    Args:
        policy (list): A policy represented as a list of 'n' and 'e'.

    Returns:
        bool: True if the policy is threshold-based, False otherwise.
        int: The threshold index if the policy is valid, else -1.
    """
    try:
        threshold = policy.index('e')
        # Ensure all values after the threshold are 'e'
        if all(p == 'e' for p in policy[threshold:]) and all(p == 'n' for p in policy[:threshold]):
            return True, threshold
    except ValueError:
        return True, len(policy)
    return False, -1

def run_markov_chain(params):
  # Solve with Markov Chain
  best_throughput = -float('inf')
  best_results = None

  for tau in range(params.B + 1):
    results = solve_markov_chain(params, tau)
    _, _, _, throughput, _, _, policy_mc, threshold_mc = results

    if throughput > best_throughput:
      best_throughput = throughput
      best_results = (tau, throughput, policy_mc, threshold_mc)

  best_tau, best_throughput, best_policy_mc, best_threshold_mc = best_results

  return best_tau, best_throughput, best_policy_mc, best_threshold_mc


def parametrizing_service_rate(df_edge, df_cloud, class_name, overhead):

	df_ee_classified_edge = df_edge#[df_edge.conf_branch_1 >= threshold]
	df_end_classified_edge = df_edge#[df_edge.conf_branch_1 < threshold]
	df_end_classified_cloud = df_cloud#[df_cloud.conf_branch_1 < threshold]

	avg_ee_edge_inf_time = 1000*df_ee_classified_edge.delta_inf_time_branch_1.mean()
	df_end_cloud_inf_time = 1000*df_end_classified_edge.delta_inf_time_branch_1 + df_end_classified_cloud.delta_inf_time_branch_2
	avg_end_cloud_inf_time = df_end_cloud_inf_time.mean() + overhead


	mu_a, mu_b = float(1)/float(avg_ee_edge_inf_time), float(1)/float(avg_end_cloud_inf_time)

	service_rate_results = {"mu_a": [mu_a], "mu_b": [mu_b], "ee_edge_inf_time": [avg_ee_edge_inf_time],
	"end_cloud_inf_time": [avg_end_cloud_inf_time],
	"class_name": [class_name], "overhead": [overhead]}

	return service_rate_results

def parametrizing_packet_loss(df_edge, df_cloud, class_name, overhead, factor):

  df_ee_classified_edge = df_edge#[df_edge.conf_branch_1 >= threshold]

  df_end_classified_cloud = df_cloud#[df_cloud.conf_branch_1 < threshold]

  ee_acc = float(df_ee_classified_edge.correct_branch_1.sum())/float(df_ee_classified_edge.shape[0])

  end_acc = float(df_end_classified_cloud.correct_branch_2.sum())/float(df_end_classified_cloud.shape[0])

  end_acc = min(end_acc+factor, 1)

  packet_loss_a, packet_loss_b = 1 - ee_acc, 1 - end_acc

  packet_loss_results = {"packet_loss_a": [packet_loss_a], "packet_loss_b": [packet_loss_b],
                         "ee_acc": [ee_acc], "end_acc": [end_acc], "class_name": [class_name], "overhead": [overhead],
                         "factor": [factor]}

  return packet_loss_results

def extract_SMDP_params(df, B, lambda_rate):
  Params = namedtuple("Params", ["B", "mu_a", "mu_b", "lambda_rate", "p_a", "p_b"])

  #print(df.mu_a, df.mu_b, df.packet_loss_a, df.packet_loss_b)
  params = Params(B=B, mu_a=df.mu_a.item(), mu_b=df.mu_b.item(),
                  lambda_rate=lambda_rate, p_a=df.packet_loss_a.item(), p_b=df.packet_loss_b.item()-0.01)
  return params


def save_QAdaEE_results(mc_results, vi_results, class_name, overhead, lambda_rate, factor, params, B, results_path):

  mc_tau, mc_throughput, mc_policy, mc_threshold = mc_results
  vi_U, vi_policy, vi_threshold = vi_results

  results_dict = {}

  for i in range(1, B + 1):
    results_dict.update({"U_%s"%(i): [vi_U[i-1]], "vi_policy_%s"%(i): [vi_policy[i-1]],
                         "mc_policy_%s"%(i): [mc_policy[i-1]]})


  results_dict.update({"throughput": [mc_throughput], "tau": [mc_tau], "class_name": [class_name],
                       "overhead": [overhead], "lambda_rate": [lambda_rate],
                       "mc_threshold": [mc_threshold], "vi_threshold": [vi_threshold],
                       "mu_a": [params.mu_a], "mu_b": [params.mu_b], "p_a": [params.p_a], "p_b": [params.p_b],
                       "factor": [factor]})

  df_results = pd.DataFrame(np.array(list(results_dict.values())).T, columns=list(results_dict.keys()))

  df_results.to_csv(results_path, mode='a', header=not os.path.exists(results_path))



def check_good_policy(matrix):

    if not matrix or not matrix[0]:
        return True  # Retorna True para matriz vazia ou colunas vazias

    # Transpor a matriz para acessar as colunas como listas
    transposed = list(zip(*matrix))

    # Verificar cada coluna
    for col in transposed:
        if not all(col[i] >= col[i + 1] for i in range(len(col) - 1)):
            return False

    return True

save_params_path = os.path.join(config.DIR_PATH, "mobilenet", "inference_data",
	"qAdaEE_params_mobilenet_1_branches_crescent_id_3_cifar10.csv")


df_params = pd.read_csv(save_params_path)

class_name_list = df_params.class_name.unique()

buffer_size_list = [1, 5, 10, 15, 20]

lambda_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

gamma = 0.05

overhead_list, factor_list = np.round(np.arange(0, 100, 0.1), 2), np.round(np.arange(0.01, 0.2, 0.01), 2)



for B in tqdm.tqdm(buffer_size_list):
  results_path = os.path.join(config.DIR_PATH, "mobilenet", "inference_data", 
  	"qAdaEE_results_mobilenet_1_branches_3_id_crescent_cifar10_buffer_%s_full.csv"%(B))

  for lambda_rate in tqdm.tqdm(lambda_rate_list):

    for factor in tqdm.tqdm(factor_list):

      threshold_mc_overhead_list = []

      for overhead in overhead_list:

        threshold_mc_classes_list = []

        for class_name in class_name_list:
          #print("Buffer size: %s, Lambda Rate: %s, overhead: %s, class_name: %s"%(B, lambda_rate, overhead, class_name))
          df_params_filtered = df_params[(df_params.class_name == class_name) & (df_params.overhead == overhead) & (df_params.factor == factor)]

          #print(factor, overhead)
          params = extract_SMDP_params(df_params_filtered, B, lambda_rate)
          #print(overhead, params)

          mc_results = run_markov_chain(params)

          tau_mc, throughput, policy_mc, threshold_mc = mc_results
          threshold_mc_classes_list.append(threshold_mc)

          vi_results = multi_operator_value_iteration(params, gamma)

        threshold_mc_overhead_list.append(threshold_mc_classes_list)
        good_policy = check_good_policy(threshold_mc_overhead_list)

        if(good_policy):
          #print("GOOD POLICY")
          #print(params)
          save_QAdaEE_results(mc_results, vi_results, class_name, overhead, lambda_rate, factor, params, B, results_path)
