# %%
import sys
import os
import math
import numpy as np

states = { "s": 0, "E": 1, "5": 2, "I" : 3, "e": 4}
id2state = {0: "s", 1: "E", 2: "5", 3: "I", 4: "e"}

state_transition_prob = np.array([[0.0, 1.0, 0.0, 0.0, 0.0], 
                                  [0.0, 0.9, 0.1, 0.0, 0.0], 
                                  [0.0, 0.0, 0.0, 1.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.9, 0.1],
                                  [0.0, 0.0, 0.0, 0.0, 0.0]]) 
emission_nuc_codes = {'A': 0, 
                      'C': 1, 
                      'G': 2, 
                      'T': 3}

emission_probs = np.array([[0.00, 0.00, 0.00, 0.00], 
                           [0.25, 0.25, 0.25, 0.25],
                           [0.05, 0.00, 0.95, 0.00],
                           [0.40, 0.10, 0.10, 0.40],
                           [0.00, 0.00, 0.00, 0.00]]) 

query_sequence = "CTTCATGTGAAAGCAGACGTAAGTCA"


# %%
def get_log_prob_for_state_path (state_path, query_sequence):
    res = math.log(0.25)
    for i in range(1, len(state_path)):
        res += math.log(state_transition_prob[ states[state_path[i-1]] ][ states[state_path[i]] ]*emission_probs[ states[state_path[i]] ][ emission_nuc_codes[query_sequence[i]] ])
    return res

# %%
# CTTCATGTGAAAGCAGACGTAAGTCA 
# EEEEEE5IIIIIIIIIIIIIIIIIII
k1 = get_log_prob_for_state_path("EEEEEE5IIIIIIIIIIIIIIIIIII", "CTTCATGTGAAAGCAGACGTAAGTCA") +  math.log (0.1)
print (k1)


# %%
# CTTCATGTGAAAGCAGACGTAAGTCA 
# EEEEEEEE5IIIIIIIIIIIIIIIII
k2 = get_log_prob_for_state_path("EEEEEEEE5IIIIIIIIIIIIIIIII", "CTTCATGTGAAAGCAGACGTAAGTCA") + math.log (0.1)
print (k2)


# %%
# CTTCATGTGAAAGCAGACGTAAGTCA 
# EEEEEEEEEEEE5IIIIIIIIIIIII
k3 = get_log_prob_for_state_path("EEEEEEEEEEEE5IIIIIIIIIIIII", "CTTCATGTGAAAGCAGACGTAAGTCA") + math.log (0.1)
print (k3)


# %%
# CTTCATGTGAAAGCAGACGTAAGTCA 
# EEEEEEEEEEEEEEE5IIIIIIIIII
k4 = get_log_prob_for_state_path("EEEEEEEEEEEEEEE5IIIIIIIIII", "CTTCATGTGAAAGCAGACGTAAGTCA") + math.log (0.1)
print (k4)


# %%
# CTTCATGTGAAAGCAGACGTAAGTCA 
# EEEEEEEEEEEEEEEEEE5IIIIIII
k5 = get_log_prob_for_state_path("EEEEEEEEEEEEEEEEEE5IIIIIII", "CTTCATGTGAAAGCAGACGTAAGTCA") + math.log (0.1)
print (k5)


# %%
# CTTCATGTGAAAGCAGACGTAAGTCA 
# EEEEEEEEEEEEEEEEEEEEEE5III
k6 = get_log_prob_for_state_path("EEEEEEEEEEEEEEEEEEEEEE5III", "CTTCATGTGAAAGCAGACGTAAGTCA") + math.log (0.1)
print (k6)


# %%
# CTTCATGTGAAAGCAGACGTAAGTCA 
# EEEEEEEEEEEEEEEEEEEEEEEEEE
only_E = get_log_prob_for_state_path("EEEEEEEEEEEEEEEEEEEEEEEEEE", "CTTCATGTGAAAGCAGACGTAAGTCA") + math.log (0.1)
print (only_E)


# %% [markdown]
# ### Design of the Viterbi Value matrix
# 
# Rows correspond to the hidden states, and the columns correspond to the emissions that is the observed nucleotide sequences. Here I am showing the calculation for the first two nucletides. 
# 
# ```
#              C                                                          T     T
# s [s-s-C(0.00) max(s-s-C-s-T, s-E-C-s-T, s-5-C-s-T, s-I-C-s-T, s-e-C-s-T)     .] 
# E [s-E-C(0.25) max(s-s-C-E-T, s-E-C-E-T, s-5-C-E-T, s-I-C-E-T, s-e-C-E-T)     .] 
# 5 [s-5-C(0.00) max(s-s-C-5-T, s-E-C-5-T, s-5-C-5-T, s-I-C-5-T, s-e-C-5-T)     .]
# I [s-I-C(0.00) max(s-s-C-I-T, s-E-C-I-T, s-5-C-I-T, s-I-C-I-T  s-e-C-I-T)     .]
# e [s-e-C(0.00) max(s-s-C-e-T, s-E-C-e-T, s-5-C-e-T, s-I-C-e-T, s-e-C-e-T)     .]
# 
# ```
# 
# It is important to remember that you will be working in the log scale.

# %%

num_states = len(states)
seq_len = len(query_sequence)


viterbi_value_matrix = np.zeros((num_states, seq_len))
viterbi_trace_matrix = np.zeros((num_states, seq_len), dtype=int)

viterbi_value_matrix[1, 0] = math.log(0.25) 

print("Scoreboards successfully created!")

# %% [markdown]
# ### Implementation of Viterbi Algorithm
# Write a function `calculate_prob_for_a_node()` that populate a single cell in the matrix. The function will return two values:
# 1. the maximum value, for example, look at the 2nd row, 2nd column in the matrix: `max(s-s-C-E-T, s-E-C-E-T, s-5-C-E-T, s-I-C-E-T, s-e-C-E-T)`. If the probability for `s-E-C-E-T` is highest (lets say X), then the function should return `X`
# 
# **AND** 
# 
# 2. The index of that maximum value described in the first point: so index of X is `1` (recall that Python works on the 0-based index system)
# 
# - Populate `viterbi_value_matrix` with `X` for row 2 and col 2
# 
# - Populate `viterbi_trace_matrix` with `1` for row 2 and col 2

# %%
def calculate_prob_for_a_node(row, col):
    nuc = query_sequence[col]
    nuc_code = emission_nuc_codes[nuc]
    potential_probs = []
    
   
    for prev_state in range(5):
        t_prob = state_transition_prob[prev_state][row]
        e_prob = emission_probs[row][nuc_code]
        
        
        if t_prob > 0 and e_prob > 0:
            log_prob = viterbi_value_matrix[prev_state][col-1] + math.log(t_prob) + math.log(e_prob)
        else:
            log_prob = float('-inf') 
            
        potential_probs.append(log_prob)
        
    
    max_val = max(potential_probs)
    max_index = potential_probs.index(max_val)
    
    return max_val, max_index

# %%

for col in range(1, seq_len):

    for row in range(5):
        max_val, max_index = calculate_prob_for_a_node(row, col)
        
        
        viterbi_value_matrix[row][col] = max_val
        viterbi_trace_matrix[row][col] = max_index

print("Scoreboard fully calculated!")

# %%
def traceback():
    path = []
    
    
    last_col = seq_len - 1
    
    current_state = np.argmax(viterbi_value_matrix[:, last_col])
    path.append(id2state[current_state])
    
    
    for col in range(last_col, 0, -1):
        current_state = viterbi_trace_matrix[current_state][col]
        path.append(id2state[current_state])
        
    
    path.reverse()
    return "".join(path)


predicted_gene_structure = traceback()
print("Final Predicted Path:", predicted_gene_structure)


