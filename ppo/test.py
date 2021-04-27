import numpy as np

def action_mask(valid_actions, log_probs):

    valid_net_actions = []
    only_valid_log_probs = []
    exp_sum_of_val_acts = 0
    valid_log_probs = log_probs
    test = list(log_probs)
    
    #Calculate the sum of valid prob exponents
    for val_act in valid_actions:
         exp_sum_of_val_acts += np.exp(log_probs[val_act])

    print("EXP SUM", exp_sum_of_val_acts)
    #The masking function is Y_k = exp(P_k) / sum(exp(P_valids))
    #Calculate the corrrect masking value for each valid action
 
        

    for indx,log_prob in enumerate(log_probs):
        for val_act in valid_actions:
            if log_prob == log_probs[val_act]:
                test[val_act] = np.exp(log_probs[val_act]) / exp_sum_of_val_acts
            else:
                test[indx] = 0.0
             
   
    #for indx, log_prob in enumerate(log_probs):     
     #   if log_prob in test:
      #      test[indx] = 0.0
    
    return test

P = [0.32, 0.15, 0.33, 0.20]

valid_actions = [1, 3]

print("ACTION MASK", action_mask(valid_actions, P))
