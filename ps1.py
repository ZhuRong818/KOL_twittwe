def get_initial_state(m, c):
    
    '''
    Create the initial state for the missionaries and cannibals problem.
    
    Parameters
    ----------    
    m: no. of missionaries
    c: no. of cannibals
    
    Returns
    ----------    
    Return the starting state derived from `m` and `c`. This could be the state representation you described in task 1.1.
    '''
    """ YOUR CODE HERE """
    return (m,c,"R",m,c) #  State = (M_R, C_R, B, m, c)
    raise NotImplementedError
    """ YOUR CODE END HERE """

def get_max_steps(m, c):
    '''
    Calculate a safe upper bound on the number of moves needed to solve the problem.
    
    Parameters
    ----------    
    m: no. of missionaries
    c: no. of cannibals
    
    Returns
    ----------    
    Returns an integer representing the maximum number of steps to explore before giving up. This could be the upper bound you described in task 1.5.
    '''
    """ YOUR CODE HERE """
    return (m+1)*(c+1)*2-1
    raise NotImplementedError
    """ YOUR CODE END HERE """

def is_goal(m, c, state):
    '''
    Check if the given state is a goal state.
    
    Parameters
    ----------    
    state: current state
    m: number of missionaries
    c: number of cannibals
    
    Returns
    ----------    
    Returns True if the state has reached your proposed goal state.
    '''
    """ YOUR CODE HERE """
    M_R, C_R, B, m_total, c_total = state
    if (M_R==0) and (C_R==0) and B=="L":
        return True
    raise NotImplementedError
    """ YOUR CODE END HERE """

def valid_actions(state):
    '''
    Generate all valid actions from the current state.
    
    Parameters
    ----------    
    state: current state
    
    Returns
    ----------    
    Returns a set of valid actions which can be performed at the current state
    '''
    """ YOUR CODE HERE """
    M_R, C_R, B, m, c = state

    # all possible boat loads (capacity 2, at least 1 person)
    candidates = [(1,0), (2,0), (0,1), (0,2), (1,1)]
    actions = set()

    for mb, cb in candidates:
        # Check enough people on the side the boat is currently on
        if B == 'R':
            if mb > M_R or cb > C_R:
                continue
            next_state = (M_R - mb, C_R - cb, 'L', m, c)
        else:  # B == 'L'
            M_L = m - M_R
            C_L = c - C_R
            if mb > M_L or cb > C_L:
                continue
            next_state = (M_R + mb, C_R + cb, 'R', m, c)

        # Only keep actions that lead to a valid (safe) state
        if _is_valid_state(next_state):
            actions.add((mb, cb))

    return actions

    raise NotImplementedError
    """ YOUR CODE END HERE """

def _is_valid_state(state):
    M_R, C_R, B, m, c = state

    if not (0 <= M_R <= m and 0 <= C_R <= c):
        return False
    if B not in ("L", "R"):
        return False

    # right bank safety
    if M_R != 0 and M_R < C_R:
        return False

    # left bank safety (use m,c)
    M_L = m - M_R
    C_L = c - C_R
    if M_L != 0 and M_L < C_L:
        return False

    return True

def transition(state, action):
    '''
    Apply an action to the current state to get the next state.
    
    Parameters
    ----------    
    state: current state
    action: your representation from valid_actions function above for the action to take
    
    Returns
    ----------    
    Returns the state after applying the action.
    '''
    """ YOUR CODE HERE """
    M_R, C_R, B, m, c = state
    mb, cb = action

    # basic action validity (should already hold if using valid_actions, but keep safe)
    if mb < 0 or cb < 0 or mb + cb < 1 or mb + cb > 2:
        return None

    if B == 'R':
        if mb > M_R or cb > C_R:
            return None
        next_state = (M_R - mb, C_R - cb, 'L', m, c)
    else:  # B == 'L'
        M_L = m - M_R
        C_L = c - C_R
        if mb > M_L or cb > C_L:
            return None
        next_state = (M_R + mb, C_R + cb, 'R', m, c)

    return next_state if _is_valid_state(next_state) else None
    raise NotImplementedError
    """ YOUR CODE END HERE """

def mnc_search(m, c):  
    '''
    Solution should be the action taken from the root node (initial state) to 
    the leaf node (goal state) in the search tree.

    Parameters
    ----------    
    m: no. of missionaries
    c: no. of cannibals
    
    Returns
    ----------    
    Returns the solution to the problem as a tuple of steps. Each step is a tuple of two numbers x and y, indicating the number of missionaries and cannibals on the boat respectively as the boat moves from one side of the river to another. If there is no solution, return False.
    '''
    queue = []
    initial_state = get_initial_state(m, c)
    queue.append((initial_state, ()))
    max_steps = get_max_steps(m, c)

    while queue:
        state, steps = queue.pop(0)
        if is_goal(m, c, state):
            return steps
        else:
            for a in valid_actions(state):
                next_state = transition(state, a)
                if next_state is None:
                    continue
                next_steps = steps + (a,)
                if len(next_steps) <= max_steps:
                    queue.append((next_state, next_steps))
    return False