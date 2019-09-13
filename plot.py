import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from deep_rl import Plotter


def plot_ppo():
    plotter = Plotter()
    games = [
        'reacher',
    ]

    patterns = [
        # '190913-083243-seed_163894'  # < the good one while training, solved env, trained from scratch in 1 mio steps
        '190913-130706-seed_29164'   # < evaluation of the good one
    ]

    labels = [
        'PPO'
    ]

    plotter.plot_games(games=games,
                       patterns=patterns,
                       agg='mean',
                       downsample=0,
                       labels=labels,
                       right_align=False,
                       tag=plotter.RETURN_TEST,
                       root='/home/philipp/udacity/DeepRL/tf_log/',
                       interpolation=None,  # 100,
                       window=0,
                       )

    plt.tight_layout()
    plt.savefig('good_models/PPO-eval.png', bbox_inches='tight')
    # plt.show(bbox_inches='tight')


if __name__ == '__main__':
    plot_ppo()