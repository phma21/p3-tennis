import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from deep_rl import Plotter


def plot_ppo():
    plotter = Plotter()
    games = [
        'tennis',
    ]

    patterns = [
        # '190922-195224-seed_18559'  # < the good one while training, solved env, trained from scratch in 1 mio steps
        '190923-085312-seed_699333'   # < evaluation of the good one
    ]

    labels = [
        'PPO'
    ]

    plotter.plot_games(games=games,
                       patterns=patterns,
                       agg='scatter',
                       downsample=0,
                       labels=labels,
                       right_align=False,
                       tag=plotter.RETURN_TEST,
                       root='./tf_log/',
                       interpolation=0,  # 100,
                       window=0,
                       )

    plt.tight_layout()
    # plt.savefig('good_models/PPO-tennis-eval.png', bbox_inches='tight')
    plt.show(bbox_inches='tight')


if __name__ == '__main__':
    plot_ppo()
