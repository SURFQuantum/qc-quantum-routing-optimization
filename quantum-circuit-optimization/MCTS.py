# main function for the Monte Carlo Tree Search
def monte_carlo_tree_search(root):
    # while resources_left(time, computational power):
    #     leaf = traverse(root)
    #     simulation_result = rollout(leaf)
    #     backpropagate(leaf, simulation_result)
    #
    # return best_child(root)
    return

def selection(parent_node, child_node):
    # receives iteration
    # choosing child node based on Upper Confidence Bound
    # UCB(node i) = (mean node value) + confidence value sqrt(log num visits parent / num visits of node i)
    return

# function for the result of the simulation
def rollout(node):
    # while non_terminal(node):
    #     node = rollout_policy(node)
    # return result(node)
    return


# function for randomly selecting a child node
def rollout_policy(node):
    # return pick_random(node.children)
    return


# function for backpropagation
def backpropagate(node, result):
    # if is_root(node) return
    # node.stats = update_stats(node, result)
    # backpropagate(node.parent)
    return


# function for selecting the best child
# node with highest number of visits
def best_child(node):
    # pick
    # child
    # with highest number of visits
    return