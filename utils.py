

def flip_observation(observation: tuple, n_pieces: int, n_spots: int) -> tuple:

    obs, servation = list(observation)[:7], list(observation)[7:]

    # First reverse the observation
    rev_obs = list(reversed(obs))

    # Then flip the number of players based on the n_pieces variable
    flipped_obs = []

    for o in rev_obs:
        if o > n_pieces:
            flipped_obs.append(o - n_pieces)
        elif o > 0 and o <= n_pieces:
            flipped_obs.append(o + n_pieces)
        else:
            flipped_obs.append(o)
    
    # Now the observation has been flipped
    return tuple(flipped_obs + servation)

def flip_action(action: tuple, n_spots: int) -> tuple:
    # This takes in an action that is based on a flipped observation
    # It then reverses the flipped action into an ordinary action

    rev_board_indices = list(reversed(range(n_spots)))

    if action[0] == 7 and action[1] == 7:
        flipped_action = (7, 7)
    elif action[1] == 7:
        flipped_action = (rev_board_indices[action[0]], 7)
    elif action[0] == 7:
        flipped_action = (7, rev_board_indices[action[1]])
    else:
        flipped_action = (rev_board_indices[action[0]], rev_board_indices[action[1]])

    return tuple(flipped_action)


def deepify_observation(obs):
    res = []
    for o in obs:
        res += [int(i) for i in format(o, "b")]
    
    return res

if __name__ == "__main__":
    obs = [1, 1, 5, 0, 2, 0, 7, 0, 0, 1, 0]

    flipped_observation = flip_observation(obs, 4, 7)

    flipped_flipped_obseration = flip_observation(flipped_observation, 4, 7)

    print("OBS:", obs)
    print("FLIPPED_OBS:", flipped_observation)
    print("FLIPPED_FLIPPED_OBS:", flipped_flipped_obseration)

    print(obs == flipped_flipped_obseration)