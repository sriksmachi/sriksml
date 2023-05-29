


def agent()
    # initialize Q(s,a) table
    Q = np.zeros([env.observation_space.n,env.action_space.n])
    
    # set learning parameters
    lr = .8
    y = .95
    num_episodes = 2000

# python main method
if __name__ == "__main__":
    agent()