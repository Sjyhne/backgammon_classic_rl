

def flip_observation(observation: tuple, n_pieces: int, n_spots: int) -> tuple:

    obs, bar_spots, dice = list(observation)[:7], list(observation)[7:9], list(observation)[9:]

    flipped_obs = []
    for o in obs:
        if o > n_pieces:
            flipped_obs.append(o - n_pieces)
        elif o > 0 and o <= n_pieces:
            flipped_obs.append(o + n_pieces)
        else:
            flipped_obs.append(o)

    rev_flipped_obs = list(reversed(flipped_obs))
    rev_bar_spots = list(reversed(bar_spots))

    result = rev_flipped_obs + rev_bar_spots + dice

    return tuple(result)

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
    obs = [1, 0, 5, 0, 3, 5, 6, 1, 0, 1, 1]

    print(obs)
    print(flip_observation(obs, 4, 7))

    print(flip_observation(flip_observation(obs, 4, 7), 4, 7))

    action = (0, 1)
    flipped_action = flip_action(action, 7)
    flipped_flipped_action = flip_action(flipped_action, 7)

    print("Action:", action)
    print("Flipped action:", flipped_action)
    print("Action:", flipped_flipped_action)
    print(action == flipped_flipped_action)
