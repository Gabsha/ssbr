import click
import json


@click.group()
def cli():
    pass


@cli.command()
def train(config, dataset, output):
    from ssbr.runners.train import train_experiment
    train_experiment(config, dataset, output)


if __name__ == '__main__':
    cli()