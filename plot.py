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
        # 'remark_ppo',
        '171913'
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
                       tag=plotter.RETURN_TRAIN,
                       root='/home/philipp/udacity/DeepRL/tf_log/',
                       interpolation=100,
                       window=0,
                       )

    # plt.show()
    plt.tight_layout()
    # plt.savefig('images/PPO.png', bbox_inches='tight')
    plt.show(bbox_inches='tight')


if __name__ == '__main__':
    plot_ppo()