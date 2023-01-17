import gymnasium as gym
import typer

from modules.dqn import DQNAgent
from modules.utils import plot_results


app = typer.Typer()


@app.command()
def train(n_episodes: int) -> None:
    env = gym.make("CarRacing-v2", continuous=False, render_mode=None)

    agent = DQNAgent(env, gamma=0.99, epsilon_init=1.0, epsilon_min=0.05, epsilon_decay=0.7)
    results = agent.train(n_episodes, 200)
    agent.save_model("models", "dqn_final")

    plot_results(results, 100, "Car Racing: Training")


@app.command()
def play(n_episodes: int, render: bool) -> None:
    if render:
        env = gym.make("CarRacing-v2", continuous=False, render_mode="human")
    else:
        env = gym.make("CarRacing-v2", continuous=False, render_mode=None)

    agent = DQNAgent(env, gamma=0.99, epsilon_init=1.0, epsilon_min=0.05, epsilon_decay=0.7)
    agent.load_model("models", "dqn_final")
    results = agent.play(n_episodes)

    scores = results["score"]
    avg_score = sum(scores) / len(scores)
    print("---------------------")
    print(f"Average score: {avg_score:.2f}")
    print("---------------------")
    input("Press ENTER to exit.")


if __name__ == "__main__":
    app()
