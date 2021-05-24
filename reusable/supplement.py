import matplotlib.pyplot as plt

def plot(returns, mean_returns, algorithm):
    plt.plot(returns, label='Episodic Return')
    plt.plot(mean_returns, label='Mean Return')
    plt.title(f'{algorithm} Training Curve')
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Total Return')
    plt.show()

def is_solved(mean_return, solved, episode, TRAIN_LENGTH):
    if mean_return[-1] >= 195.0 and not solved:
        print(f'Solved CartPole in {episode} episodes :)')
        solved = True
    elif episode + 1 == TRAIN_LENGTH:
        print('You failed.')
    
    return solved